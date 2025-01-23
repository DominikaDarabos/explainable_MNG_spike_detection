import pandas as pd 
import os
import numpy as np
import torch
import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import pickle
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from sklearn.cluster import KMeans
from collections import Counter
import matplotlib.cm as cm


class MicroneurographyDataloader:
    def __init__(self, raw_data_relative_path='../data/raw_data.csv', spikes_relative_path='../data/spike_timestamps.csv', stimulation_relative_path='../data/stimulation_timestamps.csv'):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.raw_data_df = self.read_csv_into_df(raw_data_relative_path)
        self.ground_truth_APs_df = self.read_csv_into_df(spikes_relative_path)
        self.stimulation_df = self.read_csv_into_df(stimulation_relative_path)
        self.track_replacement_dict = self.generate_track_replacement_dict()
        self.all_spikes_df = self.outer_merge_spikes()
        self.raw_data_windows = None
        self.raw_timestamps_windows = None
        self.binary_labels = None
        self.multiple_labels = None
        self.binary_labels_onehot = None
        self.multiple_labels_onehot = None
        self.window_size = None
        self.overlapping = None

    """
    Private, utils
    """

    def read_csv_into_df(self, relative_path):
        csv_path = os.path.abspath(relative_path)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"ERROR: file at {csv_path} does not exist.")
        return pd.read_csv(csv_path)
    
    def generate_track_replacement_dict(self):
        """
        The track types used in the file are numbered for further label processing.
        Their number will be assigned based on the order of their occurence in the file.
        The first occured unique type will get the number 2, and the last will get the number of unique tracks plus 1.
        The number 1 is assigned to the stimulus spikes.
        """
        unique_tracks = self.ground_truth_APs_df['track'].unique()
        r_dict = {track: idx+2 for idx, track in enumerate(unique_tracks)}
        r_dict["X"] = 1
        return r_dict


    def outer_merge_spikes(self):
        """
        Merges the self.ground_truth_APs_df and self.stimulation_df so that all kinds of spikes (from a computational perspective) are in one object.
        The output is a dataframe with columns 'ts' and 'track' where track values are numbers from 1 (stimulus) to the number of unique track types + 1.
        """
        df1 = self.ground_truth_APs_df.rename(columns={'spike_ts': 'ts'})
        df2 = self.stimulation_df.rename(columns={'stimulation_ts': 'ts'})

        all_spike = pd.merge(df1, df2[['ts']], on='ts', how='outer')

        all_spike['track'] = all_spike['track'].fillna('X')

        all_spike = all_spike.sort_values('ts').reset_index(drop=True)
        all_spike = all_spike.drop(columns=['spike_idx'])
        all_spike['track'] = all_spike['track'].replace(self.track_replacement_dict).infer_objects(copy=False)
        return all_spike

    def slice_overlapping_raw_data_windows(self, df, column_name):
        stride = self.window_size - self.overlapping
        # windows_matrix = np.array([df[column_name].values[i:i + self.window_size] for i in range(0, len(df) - self.window_size + 1, stride)])
        # return windows_matrix
        data = df[column_name].values
        windows_matrix = np.lib.stride_tricks.sliding_window_view(data, window_shape=self.window_size)
        return windows_matrix[::stride]

    def one_hot_encode(self, labels, num_classes):
        return torch.nn.functional.one_hot(labels, num_classes=num_classes)

    """
    Public
    """
    
    def generate_raw_windows(self, window_size, overlapping = 0):
        """
        Slice windows for the raw data values and timestamps seperately, but with the same size attributes.
        The given window size and overlapping size are saved as class attribute so that when the class instance is saved, these base parameters can be easily accessed again.
        """
        self.window_size = window_size
        self.overlapping = overlapping
        self.raw_timestamps_windows = self.slice_overlapping_raw_data_windows(self.raw_data_df, 'raw_ts')
        self.raw_data_windows = self.slice_overlapping_raw_data_windows(self.raw_data_df, 'raw_amplitude')


    def generate_labels(self):
        """
        Label generation: if a spike ts is in the current timestamp window, the corresponding label will be positive for the respective value window. Otherwise, the label will be 0.
        Binary labels: Any spike occurence will be labeled as 1.
        Multiclass labels:
            1 - stimulus
            2 - AP for first nerve
            3 - AP for second nerve and so on
        """
        binary_labels_list = []
        multiple_labels_list = []
        for timestamps_window in self.raw_timestamps_windows:
            window_first_ts = timestamps_window[0]
            window_last_ts = timestamps_window[-1]
            
            spike_matching_rows = self.all_spikes_df[
                (self.all_spikes_df['ts'] >= window_first_ts) & 
                (self.all_spikes_df['ts'] <= window_last_ts)]

            if len(spike_matching_rows) == 1:
                multiple_labels_list.append(spike_matching_rows['track'].values[0])
                binary_labels_list.append(1)
            elif len(spike_matching_rows) > 1:
                raise ValueError(f"WARNING: there are {len(spike_matching_rows)}  spike timestamps in one window.")
            else:
                multiple_labels_list.append(0)
                binary_labels_list.append(0)


        self.binary_labels = torch.tensor(binary_labels_list, dtype=torch.int64)
        self.multiple_labels = torch.tensor(multiple_labels_list, dtype=torch.int64)
        self.binary_labels_onehot = self.one_hot_encode(self.binary_labels, 2).to(self.device).to(dtype=torch.float32)
        self.multiple_labels_onehot = self.one_hot_encode(self.multiple_labels, len(torch.unique(self.multiple_labels))).to(self.device).to(dtype=torch.float32)
        value_counts = Counter(multiple_labels_list)
        print("Multi class labels count: ", value_counts)


    def generate_labels_stimuli_relabel(self):
        """
        The stimulus "arrives" approximately 40 datapoints later than it is marked. If the stimulus extreme values appear within
        round(40 / (self.window_size - self.overlapping)) number of windows after the marked timestamp, it is relabeled as class 1.
        If there are no high values in the window originally labeled as stimulus, it is relabeled to class 0.

        Label generation: if a spike ts is in the current timestamp window, the corresponding label will be positive for the respective value window. Otherwise, the label will be 0.
        Binary labels: Any spike occurence will be labeled as 1.
        Multiclass labels:
            1 - stimulus
            2 - AP for first nerve
            3 - AP for second nerve and so on
        """
        binary_labels_list = []
        multiple_labels_list = []

        counter_orig_1_but_0 = 0
        counter_orig_0_but_1 = 0
        counter_orig_1_and_1 = 0
        counter_no_stimulation = 0
        counter_stimuliby_filter_but_not_in_threshold = 0
        last_X_tracks = []
        relabel_threshold = round(40 / (self.window_size - self.overlapping))
    
        print("Original spikes count:", self.all_spikes_df['track'].value_counts())

        for i, (timestamp_window, data_window) in enumerate(zip(self.raw_timestamps_windows, self.raw_data_windows)):
            window_first_ts = timestamp_window[0]
            window_last_ts = timestamp_window[-1]
            
            spike_matching_rows = self.all_spikes_df[
                (self.all_spikes_df['ts'] >= window_first_ts) & 
                (self.all_spikes_df['ts'] <= window_last_ts)
            ]

            #relabel filter
            has_value_below_neg10 = any(value <= -10 for value in data_window)
            has_value_above_9 = any(value >= 9 for value in data_window)
            is_stimulation_by_filter = has_value_below_neg10 and has_value_above_9

            if len(spike_matching_rows) == 1: #originally label 1 2 3
                track_value = spike_matching_rows['track'].values[0]
                last_X_tracks.append(track_value)
                if len(last_X_tracks) > relabel_threshold:
                    last_X_tracks.pop(0)
                if track_value == 1 and is_stimulation_by_filter:
                    counter_orig_1_and_1 += 1
                    multiple_labels_list.append(1)
                    binary_labels_list.append(1)
                elif track_value  == 1 and not is_stimulation_by_filter:
                    counter_orig_1_but_0 += 1
                    multiple_labels_list.append(0)
                    binary_labels_list.append(0)
                elif track_value != 0:
                    multiple_labels_list.append(track_value)
                    binary_labels_list.append(1)
                if track_value != 0 and track_value != 1 and is_stimulation_by_filter:
                    print("STIMULATION BUT OTHER SPIKE")
            elif len(spike_matching_rows) > 1:
                raise ValueError(f"WARNING: there are {len(spike_matching_rows)}  spike timestamps ({spike_matching_rows}) in window {i}.")
            else: # originally label 0
                last_X_tracks.append(0)
                if len(last_X_tracks) > relabel_threshold:
                    last_X_tracks.pop(0)
                if is_stimulation_by_filter:
                    if 1 in last_X_tracks:
                        counter_orig_0_but_1 += 1
                        multiple_labels_list.append(1)
                        binary_labels_list.append(1)
                    else:
                        # Do not label as 1 since '1' not in last X 'track' values
                        multiple_labels_list.append(0)
                        binary_labels_list.append(0)
                        counter_stimuliby_filter_but_not_in_threshold += 1
                else:
                    multiple_labels_list.append(0)
                    binary_labels_list.append(0)
                    counter_no_stimulation += 1
        print("Relabel statistics:")
        print("Originally stimulation label but not stimulation: ", counter_orig_1_but_0)
        print("Originally stimulation label and stimulation indeed: ", counter_orig_1_and_1)
        print("Originally not stimulation label but is stimulation: ", counter_orig_0_but_1)
        print("Originally not stimulation label and not stimulation: ", counter_no_stimulation)
        print(f"Stimulus according to the filter, but stimulus ts is not in the last {relabel_threshold} track: ", counter_stimuliby_filter_but_not_in_threshold)
        value_counts = Counter(multiple_labels_list)

        print("Multi class labels count: ", value_counts)

        self.binary_labels = torch.tensor(binary_labels_list, dtype=torch.int64)
        self.multiple_labels = torch.tensor(multiple_labels_list, dtype=torch.int64)
        self.binary_labels_onehot = self.one_hot_encode(self.binary_labels, 2).to(self.device).to(dtype=torch.float32)
        self.multiple_labels_onehot = self.one_hot_encode(self.multiple_labels, len(torch.unique(self.multiple_labels))).to(self.device).to(dtype=torch.float32)

    def sequential_split_with_resampling(self, batch_size, minor_upsample_count=25000, major_downsample_count=75000):
        """
        Split the whole data into training (the first 60% of the data), validation (the next 20% of the data), and test (the last 20%of the data) sets.
        The training set's positive class is upsamled to the amount of minor_upsample_count, and the negative class to the amount of major_downsample_count.
        The output is a dictionary, contianing 
            "train_loader": the training dataset in the original class ratio.
            "val_loader": the validation dataset in the original class ratio.
            "test_loader": the test dataset in the original class ratio.
            "train_loader_under": the resampled training dataset.
            "val_timestamps": timestamp windows corresponding to the samples in the validation dataset.
            "test_timestamps": timestamp windows corresponding to the samples in the test dataset.
        """
        if self.raw_data_windows is None or self.binary_labels is None or self.multiple_labels is None:
            print("ERROR: raw data windows and labels should be generated first.")
            return

        # Convert raw data and labels to tensors
        samples = torch.tensor(self.raw_data_windows, dtype=torch.float64).unsqueeze(1).to(self.device)

        total_size = len(samples)
        train_size = int(0.6 * total_size)  # 60% for training
        val_size = int(0.2 * total_size)    # 20% for validation

        torch.manual_seed(4)
        indices = torch.arange(total_size)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        train_indices = train_indices[torch.randperm(len(train_indices))]  # Shuffle only the training indices

        train_samples, val_samples, test_samples = samples[train_indices], samples[val_indices], samples[test_indices]
        train_binary_labels, val_binary_labels, test_binary_labels = self.binary_labels_onehot[train_indices], self.binary_labels_onehot[val_indices], self.binary_labels_onehot[test_indices]
        train_multiple_labels, val_multiple_labels, test_multiple_labels = self.multiple_labels_onehot[train_indices], self.multiple_labels_onehot[val_indices], self.multiple_labels_onehot[test_indices]

        train_samples_under, train_binary_labels_under, train_multiple_labels_under = self.apply_complex_oversampling_and_undersampling(
            train_samples, train_binary_labels, train_multiple_labels, minor_upsample_count, major_downsample_count
        )

        train_dataset = TensorDataset(train_samples, train_binary_labels, train_multiple_labels)
        train_dataset_under = TensorDataset(train_samples_under, train_binary_labels_under, train_multiple_labels_under)
        val_dataset = TensorDataset(val_samples, val_binary_labels, val_multiple_labels)
        test_dataset = TensorDataset(test_samples, test_binary_labels, test_multiple_labels)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        train_loader_under = DataLoader(train_dataset_under, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        result = {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "train_loader_under": train_loader_under,
        "val_timestamps": self.raw_timestamps_windows[val_indices],
        "test_timestamps": self.raw_timestamps_windows[test_indices],
        }

        return result


    def apply_random_undersampling(self, samples, binary_labels_onehot, multiple_labels_onehot):
        """
        Undersample the negative class to the amount of the positive class for a balanced dataset.
        """
        binary_labels_flat = binary_labels_onehot.argmax(dim=1).cpu().numpy()
        all_samples = samples.cpu().numpy()

        class_0_indices = np.where(binary_labels_flat == 0)[0]
        class_1_indices = np.where(binary_labels_flat == 1)[0]

        # Keep all samples from class 1 and randomly sample from class 0
        len_class_1 = len(class_1_indices)
        sampled_class_0_indices = np.random.choice(class_0_indices, len_class_1, replace=False)

        resampled_indices = np.concatenate([class_1_indices, sampled_class_0_indices])

        redrawn_samples = all_samples[resampled_indices]
        redrawn_bin_labels = binary_labels_onehot[resampled_indices].cpu().numpy()
        redrawn_multi_labels = multiple_labels_onehot[resampled_indices].cpu().numpy()

        samples_tensor = torch.tensor(redrawn_samples, dtype=torch.float64).to(self.device)
        binary_labels_tensor = torch.tensor(redrawn_bin_labels, dtype=torch.float32).to(self.device)
        multiple_labels_tensor = torch.tensor(redrawn_multi_labels, dtype=torch.float32).to(self.device)

        return samples_tensor, binary_labels_tensor, multiple_labels_tensor
    
    def apply_oversampling_and_undersampling(self, samples, binary_labels_onehot, multiple_labels_onehot, upsample_count, downsample_count):
        """
        Upsample the positive class to the amount of upsample_count and downsample the negative class to the amount of downsample_count.
        """
        binary_labels_flat = binary_labels_onehot.argmax(dim=1).cpu().numpy()
        all_samples = samples.cpu().numpy()

        class_0_indices = np.where(binary_labels_flat == 0)[0]
        class_1_indices = np.where(binary_labels_flat == 1)[0]

        # Oversample class 1 to upsample_count
        if len(class_1_indices) < upsample_count:
            oversampled_indices = np.random.choice(class_1_indices, upsample_count, replace=True)

        # Undersample class 0 to downsample_count
        if len(class_0_indices) > downsample_count:
            undersampled_indices = np.random.choice(class_0_indices, downsample_count, replace=False)

        resampled_indices = np.concatenate([oversampled_indices, undersampled_indices])

        redrawn_samples = all_samples[resampled_indices]
        redrawn_bin_labels = binary_labels_onehot[resampled_indices].cpu().numpy()
        redrawn_multi_labels = multiple_labels_onehot[resampled_indices].cpu().numpy()

        samples_tensor = torch.tensor(redrawn_samples, dtype=torch.float64).to(samples.device)
        binary_labels_tensor = torch.tensor(redrawn_bin_labels, dtype=torch.float32).to(samples.device)
        multiple_labels_tensor = torch.tensor(redrawn_multi_labels, dtype=torch.float32).to(samples.device)

        return samples_tensor, binary_labels_tensor, multiple_labels_tensor
    

    def apply_complex_oversampling_and_undersampling(self, samples, binary_labels_onehot, multiple_labels_onehot, upsample_count, downsample_count):
        """
        Upsample the positive class to the amount of upsample_count, but the repetitive samples are only from the action potentials excluding the stimulus samples.
        Downsample the negative class to the amount of downsample_count.
        """

        binary_labels_flat = binary_labels_onehot.argmax(dim=1).cpu().numpy()
        all_samples = samples.cpu().numpy()
        multiple_labels_flat = multiple_labels_onehot.argmax(dim=1).cpu().numpy()

        ### exclude stimulation indices
        # exclude_indices = np.where(multiple_labels_flat == 1)[0]
        # oversampled_indices = list(np.where((binary_labels_flat == 1) & (~np.isin(np.arange(len(binary_labels_flat)), exclude_indices)))[0])

        # include every positive indice once
        oversampled_indices = list(np.where(binary_labels_flat == 1)[0])

        # Calculate how many more samples are needed to reach the specified upsample_count
        num_class_1 = len(oversampled_indices)
        num_needed_remaining_indices = upsample_count - num_class_1

        if num_needed_remaining_indices > 0:
            AP_indices = np.where((multiple_labels_flat != 0) & (multiple_labels_flat != 1))[0]
            # indices_multiple_2 = np.where(multiple_labels_flat == 2)[0]
            # indices_multiple_3 = np.where(multiple_labels_flat == 3)[0]
            # combined_indices = np.concatenate([indices_multiple_2, indices_multiple_3])
            # probabilities = np.array([0.5 if i in indices_multiple_2 else 0.5 for i in combined_indices])
            # probabilities /= probabilities.sum()

            sampled_additional_indices = np.random.choice(AP_indices, num_needed_remaining_indices, replace=True) #, p = probabilities)
            oversampled_indices.extend(sampled_additional_indices)

        class_0_indices = np.where(binary_labels_flat == 0)[0]
        if len(class_0_indices) > downsample_count:
            sampled_class_0_indices = np.random.choice(class_0_indices, downsample_count, replace=False)

        resampled_indices = np.concatenate([list(oversampled_indices), sampled_class_0_indices])

        # Select balanced data
        redrawn_samples = all_samples[resampled_indices]
        redrawn_bin_labels = binary_labels_onehot[resampled_indices].cpu().numpy()
        redrawn_multi_labels = multiple_labels_onehot[resampled_indices].cpu().numpy()

        samples_tensor = torch.tensor(redrawn_samples, dtype=torch.float64).to(samples.device)
        binary_labels_tensor = torch.tensor(redrawn_bin_labels, dtype=torch.float32).to(samples.device)
        multiple_labels_tensor = torch.tensor(redrawn_multi_labels, dtype=torch.float32).to(samples.device)

        return samples_tensor, binary_labels_tensor, multiple_labels_tensor


    def apply_clustered_undersampling(self, samples, binary_labels_onehot, multiple_labels_onehot, num_clusters=10):
        """
        Downsampling the negative class to the amount of positive class. The selected samples are drawn from predefined clusters equally.
        """
        binary_labels_flat = binary_labels_onehot.argmax(dim=1).cpu().numpy()
        all_samples = samples.cpu().numpy()


        class_0_indices = np.where(binary_labels_flat == 0)[0]
        class_1_indices = np.where(binary_labels_flat == 1)[0]

        # Clustering class 0 samples
        class_0_samples = all_samples[class_0_indices]
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(class_0_samples)

        cluster_indices = {}
        for cluster_id in range(num_clusters):
            cluster_indices[cluster_id] = np.where(kmeans.labels_ == cluster_id)[0]

        sampled_class_0_indices = []
        samples_per_cluster = len(class_1_indices) // num_clusters

        for cluster_id, indices in cluster_indices.items():
            if len(indices) > samples_per_cluster:
                samples_from_cluster = np.random.choice(indices, samples_per_cluster, replace=False)
            else:
                samples_from_cluster = indices
            sampled_class_0_indices.extend(class_0_indices[samples_from_cluster])

        undersampled_indices = np.concatenate([class_1_indices, sampled_class_0_indices])

        redrawn_samples = all_samples[undersampled_indices]
        redrawn_bin_labels = binary_labels_onehot[undersampled_indices].cpu().numpy()
        redrawn_multi_labels = multiple_labels_onehot[undersampled_indices].cpu().numpy()

        samples_tensor = torch.tensor(redrawn_samples, dtype=torch.float64).to(self.device)
        binary_labels_tensor = torch.tensor(redrawn_bin_labels, dtype=torch.float32).to(self.device)
        multiple_labels_tensor = torch.tensor(redrawn_multi_labels, dtype=torch.float32).to(self.device)

        return samples_tensor, binary_labels_tensor, multiple_labels_tensor

    
    def write_samples_and_labels_into_file(self, file_path):
        full_path = os.path.join('../data', file_path)
        params = {
                'raw_data_windows': self.raw_data_windows,
                'raw_timestamps_windows': self.raw_timestamps_windows,
                'binary_labels': self.binary_labels,
                'multiple_labels': self.multiple_labels,
                'binary_labels_onehot': self.binary_labels_onehot,
                'multiple_labels_onehot': self.multiple_labels_onehot,
                'window_size': self.window_size,
                'overlapping': self.overlapping
            }
        with open(full_path, 'wb') as file:
            pickle.dump(params, file)


    def load_samples_and_labels_from_file(self, file_path):
        full_path = os.path.join('../data', file_path)

        if os.path.exists(full_path):
            with open(full_path, 'rb') as file:
                params = pickle.load(file)
            self.raw_data_windows = params['raw_data_windows']
            self.raw_timestamps_windows = params['raw_timestamps_windows']
            self.binary_labels = params['binary_labels']
            self.multiple_labels = params['multiple_labels']
            self.binary_labels_onehot = params['binary_labels_onehot']
            self.multiple_labels_onehot = params['multiple_labels_onehot']
            self.window_size = params['window_size']
            self.overlapping = params['overlapping']
        else:
            raise FileNotFoundError(f"The file {full_path} does not exist.")

    
    """
    Plotting
    """

    def plot_raw_data_window_by_label(self, track_id, subplot_length):
        """
        Plot the first {subplot_length} windows that are labeled as class {track_id}.
        """
        windows_to_plot, ts_windows, spike_timestamps = [], [], []

        for i in range(len(self.multiple_labels)):
            if self.multiple_labels[i] == track_id:
                windows_to_plot.append(self.raw_data_windows[i])
                ts_windows.append(self.raw_timestamps_windows[i])
            if len(ts_windows) == subplot_length:
                break
    
        for i in range(len(self.all_spikes_df['track'])):
            if self.all_spikes_df['track'][i] == track_id:
                spike_timestamps.append(self.all_spikes_df['ts'][i])
            if len(spike_timestamps) == subplot_length:
                break

        fig, axs = plt.subplots(subplot_length, 1, figsize=(8, 10))
        all_handles_labels = []

        for i in range(len(windows_to_plot)):
            raw_data = windows_to_plot[i]
            times_data = ts_windows[i]

            line, = axs[i].plot(times_data, raw_data, marker='o', color='gray', linewidth=2, markersize=6)

            all_handles_labels.append((line, f"Raw data"))

            for ts in spike_timestamps:
                if ts in times_data:
                    vline = axs[i].axvline(x=ts, color='red', linestyle='--', label=f'Stimulus Timestamp', linewidth=3)
                    all_handles_labels.append((vline, 'Stimulus Timestamp'))

            axs[i].set_ylabel("Amplitude")
            axs[i].set_ylim(-10,10)
            axs[i].grid()
            axs[i].ticklabel_format(useOffset=False)
            if i == subplot_length - 1:
                axs[i].set_xlabel("Timestamp")

        handles, labels = zip(*all_handles_labels)
        unique_handles_labels = dict(zip(labels, handles))
        fig.legend(unique_handles_labels.values(), unique_handles_labels.keys(), loc='upper center', ncol=2)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
    
    """
    Statistics
    """

    def get_statistics_of_spikes(self):
        time_diffs = self.all_spikes_df['ts'].diff().dropna()
        min_gap = time_diffs.min()
        max_gap = time_diffs.max()

        indices = self.raw_data_df.index[self.raw_data_df['raw_ts'].isin(self.all_spikes_df['ts'])].tolist()

        index_diffs = np.diff(indices)
        min_index_gap = index_diffs.min()
        max_index_gap = index_diffs.max()

        print(f'Minimum index gap between spike timestamps: {min_index_gap}')
        print(f'Maximum index gap between spike timestamps: {max_index_gap}')

        print(f"Minimum time gap between spike timestamps: {min_gap}")
        print(f"Maximum time gap between spike timestamps: {max_gap}")

        raw_time_diffs = self.raw_data_df['raw_ts'].diff().dropna()
        print(f"Minimum gap between sampled values: {raw_time_diffs.min()}") # 0.00009999999929277692
        print(f"Spike min gap / sample freq gap: {min_gap / raw_time_diffs.min()}")
    
    def get_value_statistics_for_classes(self):
        # multiple_labels = np.array(self.multiple_labels)
        # raw_data_windows = np.array(self.raw_data_windows)
        print("=" * 40)
        print("Value statistics")
        print("=" * 40)
        num_of_classes = len(torch.unique(self.multiple_labels))
        for label in range(num_of_classes):
            label_filter = self.multiple_labels == label
            filtered_windows = self.raw_data_windows[label_filter]
            
            if filtered_windows.size > 0:
                overall_min = np.min(filtered_windows)
                overall_max = np.max(filtered_windows)
                print(f"Label {label}: Overall Min = {overall_min}, Overall Max = {overall_max}")
            else:
                print(f"No windows found for label {label}.")
        print("=" * 40)