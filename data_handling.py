import pandas as pd 
import os
import numpy as np
import torch
import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split
import pickle
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class NeurographyDataset:
    def __init__(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        directory_path = os.getcwd()
        self.raw_data = self.read_csv_into_df([directory_path, '..', 'data', 'raw_data.csv'])
        self.ground_truth_spikes = self.read_csv_into_df([directory_path, '..', 'data', 'spike_timestamps.csv'])
        self.stimulation = self.read_csv_into_df([directory_path, '..', 'data', 'stimulation_timestamps.csv'])
        self.all_differentiated_spikes = self.outer_merge_spikes()
        self.raw_data_windows = None
        self.raw_timestamps_windows = None
        self.binary_labels = None
        self.multiple_labels = None
        self.binary_labels_onehot = None
        self.multiple_labels_onehot = None
        self.samples = None
        self.window_size = None
        self.overlapping = None

    """
    Private, utils
    """

    def read_csv_into_df(self, path_list):
        csv_path = os.path.join(*path_list)
        csv_path = os.path.abspath(csv_path)
        return pd.read_csv(csv_path)


    def outer_merge_spikes(self):
        df1 = self.ground_truth_spikes.rename(columns={'spike_ts': 'ts'})
        df2 = self.stimulation.rename(columns={'stimulation_ts': 'ts'})

        all_spike = pd.merge(df1, df2[['ts']], on='ts', how='outer')

        # Fill missing track values with 'X'
        all_spike['track'] = all_spike['track'].fillna('X')

        all_spike = all_spike.sort_values('ts').reset_index(drop=True)
        all_spike = all_spike.drop(columns=['spike_idx'])
        replacement_dict = {'X': 1, 'Track3': 2, 'Track4': 3}
        all_spike['track'] = all_spike['track'].replace(replacement_dict)
        return all_spike

    def slice_continuous_raw_data_windows(self, df, column_name):
        num_windows = len(df) // self.window_size
        matrix = df[column_name].values[:num_windows * self.window_size].reshape(num_windows, self.window_size)
        return matrix

    def slice_overlapping_raw_data_windows(self, df, column_name):
        stride = self.window_size - self.overlapping  # Define the stride (how much the window shifts)
        num_windows = (len(df) - self.window_size) // stride + 1  # Calculate number of windows
        matrix = np.array([df[column_name].values[i:i + self.window_size] for i in range(0, len(df) - self.window_size + 1, stride)])
        return matrix

    def one_hot_encode(self, labels, num_classes):
        return torch.nn.functional.one_hot(labels, num_classes=num_classes)

    """
    Public
    """
    
    def generate_raw_windows(self, window_size, overlapping = None):
        self.window_size = window_size
        self.overlapping = overlapping

        if type(self.window_size) == type(4):
            if overlapping is None:
                self.raw_timestamps_windows = self.slice_continuous_raw_data_windows(self.raw_data, 'raw_ts')
                self.raw_data_windows = self.slice_continuous_raw_data_windows(self.raw_data,  'raw_amplitude')
            else:
                self.raw_timestamps_windows = self.slice_overlapping_raw_data_windows(self.raw_data, 'raw_ts')
                self.raw_data_windows = self.slice_overlapping_raw_data_windows(self.raw_data, 'raw_amplitude')
        else:
            print("Window size must be of integer type.")

    def generate_labels(self):
        binary_labels_list = []
        multiple_labels_list = []
        for i, window in enumerate(self.raw_timestamps_windows):
            matching_rows = self.all_differentiated_spikes[self.all_differentiated_spikes['ts'].isin(window)]
            
            if len(matching_rows) == 1:
                multiple_labels_list.append(matching_rows['track'].values[0])
                binary_labels_list.append(1)
            elif len(matching_rows) > 1:
                print("MORE SPIKE IN ONE WINDOW")
                raise ValueError(f"{len(matching_rows)} timestamps matched in window {i} with details: {matching_rows}")
            else:
                multiple_labels_list.append(0)
                binary_labels_list.append(0)
    
        replacement_dict = {'X': 1, 'Track3': 2, 'Track4': 3}
        filtered_list = [replacement_dict.get(item, item) for item in multiple_labels_list]
        self.binary_labels = torch.tensor(binary_labels_list, dtype=torch.int64)
        self.multiple_labels = torch.tensor(filtered_list, dtype=torch.int64)
        binary_labels_onehot = self.one_hot_encode(self.binary_labels, len(torch.unique(self.binary_labels)))
        multiple_labels_onehot = self.one_hot_encode(self.multiple_labels, len(torch.unique(self.multiple_labels)))
        self.binary_labels_onehot = binary_labels_onehot.to(self.device).to(dtype=torch.float32)
        self.multiple_labels_onehot = multiple_labels_onehot.to(self.device).to(dtype=torch.float32)

    def random_split_binary_dataloader(self):
        if self.raw_data_windows is None or self.binary_labels is None:
            print("First generate the raw windows and labels.")
            return
        samples = torch.tensor(self.raw_data_windows, dtype=torch.float64)
        self.samples = samples.unsqueeze(1).to(self.device)

        dataset = TensorDataset(self.samples, self.binary_labels_onehot)
        total_size = len(dataset)
        train_size = int(0.8 * total_size)
        test_size = total_size - train_size

        torch.manual_seed(4)
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        return train_loader, test_loader

    def random_split_binary_and_multiple_dataloader(self):
        if self.raw_data_windows is None or self.binary_labels is None or self.multiple_labels is None:
            print("First generate the raw windows and labels (binary and multiple).")
            return
        samples = torch.tensor(self.raw_data_windows, dtype=torch.float64)

        self.samples = samples.unsqueeze(1).to(self.device)

        dataset = TensorDataset(self.samples, self.binary_labels_onehot, self.multiple_labels_onehot)
        total_size = len(dataset)
        train_size = int(0.6 * total_size)   # 60% for training
        val_size = int(0.2 * total_size)     # 20% for validation
        test_size = total_size - train_size - val_size  # The rest for testing (20%)

        # Split the dataset into train, validation, and test sets
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

        # Create DataLoaders for each set
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        self.print_dataset_info("Training", train_dataset)
        self.print_dataset_info("Validation", val_dataset)
        self.print_dataset_info("Testing", test_dataset)

        return train_loader, val_loader, test_loader
    
    def write_samples_and_labels_into_file(self, file_path):
        full_path = os.path.join('../data', file_path)
        params = {
                'raw_data_windows': self.raw_data_windows,
                'raw_timestamps_windows': self.raw_timestamps_windows,
                'binary_labels': self.binary_labels,
                'multiple_labels': self.multiple_labels,
                'binary_labels_onehot': self.binary_labels_onehot,
                'multiple_labels_onehot': self.multiple_labels_onehot,
                'samples': self.samples,
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
            self.samples = params['samples']
            self.window_size = params['window_size']
            self.overlapping = params['overlapping']
        else:
            raise FileNotFoundError(f"The file {full_path} does not exist.")

    
    """
    Plotting
    """

    def plot_raw_data_and_spikes(self, df_part):
        # Get the range of the timestamps from the first 2000 points
        min_ts = df_part['raw_ts'].min()
        max_ts = df_part['raw_ts'].max()

        # 2. Filter timestamps that fall within the plotted range
        df_timestamps_in_range = self.stimulation[
            (self.stimulation['stimulation_ts'] >= min_ts) & (self.stimulation['stimulation_ts'] <= max_ts)
        ]

        df_timestamps_in_range_gt = self.ground_truth_spikes[
            (self.ground_truth_spikes['spike_ts'] >= min_ts) & (self.ground_truth_spikes['spike_ts'] <= max_ts)
        ]

        plt.figure(figsize=(20, 5))
        plt.scatter(df_part['raw_ts'], df_part['raw_amplitude'], label='Raw Data', color='gray', s=3)

        # 4. Plot the filtered timestamps as dots
        #plt.scatter(df_timestamps_in_range['stimulation_ts'], 
        #            [df_part['raw_amplitude'].mean()] * len(df_timestamps_in_range),  # Positioning dots at mean amplitude level
        #            color='r', marker='o', label='Stimulus Timestamps', zorder=5)

        plt.vlines(df_timestamps_in_range['stimulation_ts'], 
                ymin=df_part['raw_amplitude'].min(), 
                ymax=df_part['raw_amplitude'].max(), 
                color='r', linestyle='--', label='Stimulus Timestamps', linewidth=1)

        df_value1 = df_timestamps_in_range_gt[df_timestamps_in_range_gt['track'] == 'Track3']  # For yellow lines
        df_value2 = df_timestamps_in_range_gt[df_timestamps_in_range_gt['track'] == 'Track4']


        plt.vlines(df_value1['spike_ts'], 
                ymin=df_part['raw_amplitude'].min(), 
                ymax=df_part['raw_amplitude'].max(), 
                color='y', linestyle='--', label='Track3', linewidth=1)

        # 5. Plot vertical lines for the second group (green)
        plt.vlines(df_value2['spike_ts'], 
                ymin=df_part['raw_amplitude'].min(), 
                ymax=df_part['raw_amplitude'].max(), 
                color='g', linestyle='--', label='Track4', linewidth=0.5)

        # Customize plot
        plt.xlabel('Timestamp')
        plt.ylabel('Amplitude')
        plt.legend()

        # Show the plot
        plt.show()

    def plot_raw_data_window_by_label(self, track_id, subplot_length):
        x_windows, x_times, special_timestamps = [], [], []

        for i in range(len(self.multiple_labels)):
            if self.multiple_labels[i] == track_id:
                x_windows.append(self.raw_data_windows[i])
                x_times.append(self.raw_timestamps_windows[i])
            if len(x_times) == subplot_length:
                break
    
        for i in range(len(self.all_differentiated_spikes['track'])):
            if self.all_differentiated_spikes['track'][i] == track_id:
                special_timestamps.append(self.all_differentiated_spikes['ts'][i])

            # Stop if we have collected enough elements
            if len(special_timestamps) == subplot_length:
                break

        # Set up the figure and axes for subplots
        fig, axs = plt.subplots(subplot_length, 1, figsize=(10, 15))
        fig.suptitle(f"First {subplot_length} Windows Labeled as {track_id}", fontsize=16)

        # Plot each of the first 5 windows labeled 'X'
        for i in range(len(x_windows)):
            window_data = x_windows[i]
            times_data = x_times[i]

            # Plot amplitudes vs. timestamps
            axs[i].plot(times_data, window_data, marker='o', label=f"Window {i + 1}")

            # Mark special timestamps
            for ts in special_timestamps:
                if ts in times_data:
                    axs[i].axvline(x=ts, color='red', linestyle='--', label=f'Spike Timestamp')

            axs[i].set_xlabel("Timestamp")
            axs[i].set_ylabel("Amplitude")
            axs[i].legend()
            axs[i].grid()

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
    
    """
    Statistics
    """

    def get_statistics_of_spikes(self):
        time_diffs = self.all_differentiated_spikes['ts'].diff().dropna()
        min_gap = time_diffs.min()
        max_gap = time_diffs.max()

        # Calculate the index differences between consecutive indices
        indices = [i for i, track in enumerate(self.binary_labels) if track == 1]
        indices = self.raw_data.index[self.raw_data['raw_ts'].isin(self.all_differentiated_spikes['ts'])].tolist()

        index_diffs = np.diff(indices)
        min_index_gap = index_diffs.min()
        max_index_gap = index_diffs.max()

        print(f'Minimum index gap: {min_index_gap}')
        print(f'Maximum index gap: {max_index_gap}')

        print(f"Minimum gap between spike timestamps: {min_gap}")
        print(f"Maximum gap between spike timestamps: {max_gap}")

        raw_time_diffs = self.raw_data['raw_ts'].diff().dropna()
        print(f"Minimum gap between sampled values: {raw_time_diffs.min()}") # 0.00009999999929277692
        print(f"Spike min gap / sample freq gap: {min_gap / raw_time_diffs.min()}")
    

#df_part = raw_data.iloc[38000:42000]
#plot_raw_data_and_spikes(df_part)

    def print_dataset_info(self, name, dataset):
        # Create DataLoader for easier iteration
        loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        data, binary_labels, multiple_labels = next(iter(loader))
        
        print(f"{name} Set:")
        print(f" - Shape of samples: {data.shape}")
        
        # Get unique values and counts for binary and multiple labels
        binary_label_classes = binary_labels.argmax(dim=-1).cpu().numpy()
        unique_binary, binary_counts = torch.unique(binary_labels, return_counts=True, dim=0)
        
        multiple_label_classes = multiple_labels.argmax(dim=-1).cpu().numpy()
        unique_multiple, multiple_counts = torch.unique(multiple_labels, return_counts=True, dim=0)
        
        print(f" - Unique binary labels and their counts: {list(zip(unique_binary, binary_counts))}")
        print(f" - Unique multiple labels and their counts: {list(zip(unique_multiple, multiple_counts))}\n")