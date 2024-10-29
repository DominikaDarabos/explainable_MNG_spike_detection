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
from sklearn.cluster import KMeans
from collections import Counter


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
            window_start = window[0]  # Start of the window
            window_end = window[-1]    # End of the window
            
            # Find timestamps that fall within this range
            matching_rows = self.all_differentiated_spikes[
                (self.all_differentiated_spikes['ts'] >= window_start) & 
                (self.all_differentiated_spikes['ts'] <= window_end)
            ]
            #matching_rows = self.all_differentiated_spikes[self.all_differentiated_spikes['ts'].isin(window)]
            
            if len(matching_rows) == 1:
                multiple_labels_list.append(matching_rows['track'].values[0])
                binary_labels_list.append(1)
            elif len(matching_rows) > 1:
                print("MORE SPIKE IN ONE WINDOW")
                raise ValueError(f"{len(matching_rows)} timestamps matched in window {i} with details: {matching_rows}")
            else:
                multiple_labels_list.append(0)
                binary_labels_list.append(0)
    
        # replacement_dict = {'X': 1, 'Track3': 2, 'Track4': 3}
        # filtered_list = [replacement_dict.get(item, item) for item in multiple_labels_list]
        self.binary_labels = torch.tensor(binary_labels_list, dtype=torch.int64)
        self.multiple_labels = torch.tensor(multiple_labels_list, dtype=torch.int64)
        binary_labels_onehot = self.one_hot_encode(self.binary_labels, len(torch.unique(self.binary_labels)))
        multiple_labels_onehot = self.one_hot_encode(self.multiple_labels, len(torch.unique(self.multiple_labels)))
        self.binary_labels_onehot = binary_labels_onehot.to(self.device).to(dtype=torch.float32)
        self.multiple_labels_onehot = multiple_labels_onehot.to(self.device).to(dtype=torch.float32)
        value_counts = Counter(multiple_labels_list)

        print("done multiple list counter: ", value_counts)


    def generate_labels_stimuli_relabel(self):
        binary_labels_list = []
        multiple_labels_list = []
        counter_match_1_but_0 = 0
        counter_match_0_but_1 = 0
        counter_match_1_but_1 = 0
        counter_no_stimulation = 0
        counter_other_spike = 0
        is_stimuli_but_not_last_two_track = 0
        last_X_tracks = []

        last_X_track_num = 11
        print("COUNTS", self.all_differentiated_spikes['track'].value_counts())

        for i, (window, data_window) in enumerate(zip(self.raw_timestamps_windows, self.raw_data_windows)):
            window_start = window[0]  # Start of the window
            window_end = window[-1]    # End of the window
            
            # Find timestamps that fall within this range
            matching_rows = self.all_differentiated_spikes[
                (self.all_differentiated_spikes['ts'] >= window_start) & 
                (self.all_differentiated_spikes['ts'] <= window_end)
            ]
            #is_stimulation = any(-10 >= value for value in data_window)
            has_value_below_neg10 = any(value <= -10 for value in data_window)
            has_value_above_9 = any(value >= 9 for value in data_window)
            is_stimulation = has_value_below_neg10 and has_value_above_9
            if len(matching_rows) == 1: #label 1 2 3
                track_value = matching_rows['track'].values[0]
                last_X_tracks.append(track_value)
                if len(last_X_tracks) > last_X_track_num:
                    last_X_tracks.pop(0)
                if track_value == 1 and is_stimulation:
                    counter_match_1_but_1 += 1
                    multiple_labels_list.append(1)
                    binary_labels_list.append(1)
                elif track_value  == 1 and not is_stimulation:
                    counter_match_1_but_0 += 1
                    multiple_labels_list.append(0)
                    binary_labels_list.append(0)
                elif track_value  != 0:
                    multiple_labels_list.append(track_value )
                    binary_labels_list.append(1)
                    counter_other_spike += 1
                if track_value != 0 and track_value != 1 and is_stimulation:
                    print("STIMULATION BUT OTHER SPIKE")
            elif len(matching_rows) > 1:
                print("MORE SPIKE IN ONE WINDOW")
                raise ValueError(f"{len(matching_rows)} timestamps matched in window {i} with details: {matching_rows}")
            else: # label 0
                last_X_tracks.append(0)
                if len(last_X_tracks) > last_X_track_num:
                    last_X_tracks.pop(0)
                if is_stimulation:
                    if 1 in last_X_tracks:
                        counter_match_0_but_1 += 1
                        multiple_labels_list.append(1)
                        binary_labels_list.append(1)
                    else:
                        # Do not label as 1 since '1' not in last five 'track' values
                        multiple_labels_list.append(0)
                        binary_labels_list.append(0)
                        is_stimuli_but_not_last_two_track += 1
                else:
                    multiple_labels_list.append(0)
                    binary_labels_list.append(0)
                    counter_no_stimulation += 1
        
        print("stimulation label but no stimulation: ", counter_match_1_but_0)
        print("stimulation label and stimulation: ", counter_match_1_but_1)
        print("no stimulation label but stimulation: ", counter_match_0_but_1)
        print("no stimulation label and no stimulation: ", counter_no_stimulation)
        print("is stimulus, but stimulus ts is not in the last 2 track: ", is_stimuli_but_not_last_two_track)
        print("other spike:", counter_other_spike)
        value_counts = Counter(multiple_labels_list)

        print("done multiple list counter: ", value_counts)

        self.binary_labels = torch.tensor(binary_labels_list, dtype=torch.int64)
        self.multiple_labels = torch.tensor(multiple_labels_list, dtype=torch.int64)
        binary_labels_onehot = self.one_hot_encode(self.binary_labels, 2)
        multiple_labels_onehot = self.one_hot_encode(self.multiple_labels, 4)
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
        torch.manual_seed(4)
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

        # Create DataLoaders for each set
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # self.print_dataset_info("Training", train_dataset)
        # self.print_dataset_info("Validation", val_dataset)
        # self.print_dataset_info("Testing", test_dataset)

        return train_loader, val_loader, test_loader


    def sequential_split_with_resampling(self):
        if self.raw_data_windows is None or self.binary_labels is None or self.multiple_labels is None:
            print("First generate the raw windows and labels (binary and multiple).")
            return

        # Convert raw data and labels to tensors
        self.samples = torch.tensor(self.raw_data_windows, dtype=torch.float64).unsqueeze(1).to(self.device)

        total_size = len(self.samples)
        train_size = int(0.6 * total_size)  # 60% for training
        val_size = int(0.2 * total_size)    # 20% for validation
        test_size = total_size - train_size - val_size  # Remaining for testing (20%)

        # Split indices for train, validation, and test
        #torch.manual_seed(4)
        # indices = torch.randperm(total_size)
        # train_indices = indices[:train_size]
        # val_indices = indices[train_size:train_size + val_size]
        # test_indices = indices[train_size + val_size:]



        indices = torch.arange(total_size)
        train_indices = indices[:train_size]  # Keep as is
        val_indices = indices[train_size:train_size + val_size]  # Keep as is
        test_indices = indices[train_size + val_size:]  # Keep as is

        # Change the random shuffling to only affect the training set
        torch.manual_seed(4)
        train_indices = train_indices[torch.randperm(len(train_indices))]  # Shuffle only the training indices

        # Use these shuffled indices to split the samples and labels
        train_samples, val_samples, test_samples = self.samples[train_indices], self.samples[val_indices], self.samples[test_indices]
        train_binary_labels, val_binary_labels, test_binary_labels = self.binary_labels_onehot[train_indices], self.binary_labels_onehot[val_indices], self.binary_labels_onehot[test_indices]
        train_multiple_labels, val_multiple_labels, test_multiple_labels = self.multiple_labels_onehot[train_indices], self.multiple_labels_onehot[val_indices], self.multiple_labels_onehot[test_indices]

        val_timestamps, test_timestamps = self.raw_timestamps_windows[val_indices], self.raw_timestamps_windows[test_indices]
        print("val timestamps shape", val_timestamps.shape)
        print("val_samples shape", val_samples.shape)

        # Split the samples and labels based on the indices
        train_samples, val_samples, test_samples = self.samples[train_indices], self.samples[val_indices], self.samples[test_indices]
        train_binary_labels, val_binary_labels, test_binary_labels = self.binary_labels_onehot[train_indices], self.binary_labels_onehot[val_indices], self.binary_labels_onehot[test_indices]
        train_multiple_labels, val_multiple_labels, test_multiple_labels = self.multiple_labels_onehot[train_indices], self.multiple_labels_onehot[val_indices], self.multiple_labels_onehot[test_indices]

        # Apply undersampling to the training data BEFORE creating DataLoader
        train_samples_under, train_binary_labels_under, train_multiple_labels_under = self.apply_oversampling_and_undersampling_comp_up_down(
            train_samples, train_binary_labels, train_multiple_labels, 25000, 75000
        )
        val_samples_under, val_binary_labels_under, val_multiple_labels_under = self.apply_oversampling_and_undersampling_comp_up_down(
            val_samples, val_binary_labels, val_multiple_labels, 25000, 75000
        )

        # Create separate datasets for train, validation, and test sets
        train_dataset = TensorDataset(train_samples, train_binary_labels, train_multiple_labels)
        train_dataset_under = TensorDataset(train_samples_under, train_binary_labels_under, train_multiple_labels_under)
        val_dataset = TensorDataset(val_samples, val_binary_labels, val_multiple_labels)
        val_dataset_under = TensorDataset(val_samples_under, val_binary_labels_under, val_multiple_labels_under)
        test_dataset = TensorDataset(test_samples, test_binary_labels, test_multiple_labels)

        # Create DataLoaders for each set
        train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
        train_loader_under = DataLoader(train_dataset_under, batch_size=1024, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)
        val_loader_under = DataLoader(val_dataset_under, batch_size=1024, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

        result = {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "train_loader_under": train_loader_under,
        "val_loader_under": val_loader_under,
        "val_timestamps": self.raw_timestamps_windows[val_indices],
        "test_timestamps": self.raw_timestamps_windows[test_indices],
        }

        print("Dataloaders are ready")

        return result


    def apply_random_undersampling(self, samples, binary_labels, multiple_labels):
        # Assuming binary_labels is one-hot encoded, convert it to class indices
        binary_labels_flat = binary_labels.argmax(dim=1).cpu().numpy()  # Get the class indices from one-hot encoding

        # Convert the samples and multiple_labels to numpy for manipulation
        all_samples = samples.cpu().numpy()
        all_multiple_labels = multiple_labels.cpu().numpy()

        # Separate class 0 and class 1
        class_0_indices = np.where(binary_labels_flat == 0)[0]
        class_1_indices = np.where(binary_labels_flat == 1)[0]

        # Keep all samples from class 1 and randomly sample from class 0
        num_class_1 = len(class_1_indices)
        sampled_class_0_indices = np.random.choice(class_0_indices, num_class_1, replace=False)

        # Combine the indices of class 1 and the sampled class 0
        undersampled_indices = np.concatenate([class_1_indices, sampled_class_0_indices])

        # Select undersampled data
        X_res = all_samples[undersampled_indices]
        y_res_binary = binary_labels[undersampled_indices].cpu().numpy()
        y_res_multiple = all_multiple_labels[undersampled_indices]

        # Convert back to tensors
        samples_tensor = torch.tensor(X_res, dtype=torch.float64).to(self.device)
        binary_labels_tensor = torch.tensor(y_res_binary, dtype=torch.float32).to(self.device)
        multiple_labels_tensor = torch.tensor(y_res_multiple, dtype=torch.float32).to(self.device)

        print("undersampled shapes: ", samples_tensor.shape, binary_labels_tensor.shape, multiple_labels_tensor.shape)

        return samples_tensor, binary_labels_tensor, multiple_labels_tensor
    
    def apply_oversampling_and_undersampling(self, samples, binary_labels, multiple_labels, upsample_count, downsample_count):
        # Assuming binary_labels is one-hot encoded, convert it to class indices
        binary_labels_flat = binary_labels.argmax(dim=1).cpu().numpy()  # Get the class indices from one-hot encoding

        # Convert the samples and multiple_labels to numpy for manipulation
        all_samples = samples.cpu().numpy()
        all_multiple_labels = multiple_labels.cpu().numpy()

        # Separate class 0 and class 1
        class_0_indices = np.where(binary_labels_flat == 0)[0]
        class_1_indices = np.where(binary_labels_flat == 1)[0]

        # Oversample class 1 to target_size
        if len(class_1_indices) < upsample_count:
            # If there are fewer instances than target_size, we can replicate some instances
            oversampled_indices = np.random.choice(class_1_indices, upsample_count, replace=True)
        else:
            print("WARNING: target size for oversampling is smaller than actual size")
            # Otherwise, randomly select target_size instances from class 1
            oversampled_indices = np.random.choice(class_1_indices, upsample_count, replace=False)

        print("class1 count", len(oversampled_indices))
        # Undersample class 0 to downsample_count
        class_0_indices = np.where(binary_labels_flat == 0)[0]
        if len(class_0_indices) > downsample_count:
            sampled_class_0_indices = np.random.choice(class_0_indices, downsample_count, replace=False)
        else:
            sampled_class_0_indices = class_0_indices  # Keep all if fewer than downsample_count
        print("class0 count", len(sampled_class_0_indices))

        # Combine the indices of oversampled class 1 and undersampled class 0
        balanced_indices = np.concatenate([oversampled_indices, sampled_class_0_indices])

        # Select balanced data
        X_balanced = all_samples[balanced_indices]
        y_balanced_binary = binary_labels[balanced_indices].cpu().numpy()
        y_balanced_multiple = all_multiple_labels[balanced_indices]

        # Convert back to tensors
        samples_tensor = torch.tensor(X_balanced, dtype=torch.float64).to(samples.device)
        binary_labels_tensor = torch.tensor(y_balanced_binary, dtype=torch.float32).to(samples.device)
        multiple_labels_tensor = torch.tensor(y_balanced_multiple, dtype=torch.float32).to(samples.device)

        print("Balanced shapes: ", samples_tensor.shape, binary_labels_tensor.shape, multiple_labels_tensor.shape)

        return samples_tensor, binary_labels_tensor, multiple_labels_tensor
    

    def apply_oversampling_and_undersampling_comp_up_down(self, samples, binary_labels, multiple_labels, upsample_count, downsample_count):
        # Assuming binary_labels is one-hot encoded, convert it to class indices
        binary_labels_flat = binary_labels.argmax(dim=1).cpu().numpy()  # Get the class indices from one-hot encoding

        # Convert the samples and multiple_labels to numpy for manipulation
        all_samples = samples.cpu().numpy()
        multiple_labels_flat = multiple_labels.argmax(dim=1).cpu().numpy()

        ### exclude stimulation indices
        # exclude_indices = np.where(multiple_labels_flat == 1)[0]
        # oversampled_indices = list(np.where((binary_labels_flat == 1) & (~np.isin(np.arange(len(binary_labels_flat)), exclude_indices)))[0])

        ### include every positive indice once
        oversampled_indices = list(np.where(binary_labels_flat == 1)[0])

        # Calculate how many more samples are needed to reach the specified upsample_count
        num_class_1 = len(oversampled_indices)
        remaining_needed = upsample_count - num_class_1

        if remaining_needed > 0:
            # Get indices for multiple_labels == 2 or multiple_labels == 3
            indices_multiple_2 = np.where(multiple_labels_flat == 2)[0]
            indices_multiple_3 = np.where(multiple_labels_flat == 3)[0]

            # Combine indices where multiple_labels are 2 or 3
            combined_indices = np.concatenate([indices_multiple_2, indices_multiple_3])
            # Normalize the probabilities so they sum to 1
            
            probabilities = np.array([0.2 if i in indices_multiple_2 else 0.8 for i in combined_indices])
            probabilities /= probabilities.sum()

            # Randomly sample from combined_indices with higher probability for multiple_labels == 3
            sampled_additional_indices = np.random.choice(combined_indices, remaining_needed, replace=True, p = probabilities)

            # Add these additional indices to the oversampled set
            oversampled_indices.extend(sampled_additional_indices)
        print("class1 count", len(oversampled_indices))
        # Undersample class 0 to downsample_count
        class_0_indices = np.where(binary_labels_flat == 0)[0]
        if len(class_0_indices) > downsample_count:
            sampled_class_0_indices = np.random.choice(class_0_indices, downsample_count, replace=False)
        else:
            sampled_class_0_indices = class_0_indices  # Keep all if fewer than downsample_count
        print("class0 count", len(sampled_class_0_indices))
        # Combine the indices of oversampled class 1 and undersampled class 0
        balanced_indices = np.concatenate([list(oversampled_indices), sampled_class_0_indices])

        # Select balanced data
        X_balanced = all_samples[balanced_indices]
        y_balanced_binary = binary_labels[balanced_indices].cpu().numpy()
        y_balanced_multiple = multiple_labels[balanced_indices].cpu().numpy()

        # Convert back to tensors
        samples_tensor = torch.tensor(X_balanced, dtype=torch.float64).to(samples.device)
        binary_labels_tensor = torch.tensor(y_balanced_binary, dtype=torch.float32).to(samples.device)
        multiple_labels_tensor = torch.tensor(y_balanced_multiple, dtype=torch.float32).to(samples.device)

        print("Balanced shapes: ", samples_tensor.shape, binary_labels_tensor.shape, multiple_labels_tensor.shape)

        return samples_tensor, binary_labels_tensor, multiple_labels_tensor



    def apply_clustered_undersampling(self, samples, binary_labels, multiple_labels, num_clusters=10):
        # Assuming binary_labels is one-hot encoded, convert it to class indices
        binary_labels_flat = binary_labels.argmax(dim=1).cpu().numpy()

        # Convert the samples and multiple_labels to numpy for manipulation
        all_samples = samples.squeeze(1).cpu().numpy()
        all_multiple_labels = multiple_labels.cpu().numpy()

        # Separate class 0 and class 1
        class_0_indices = np.where(binary_labels_flat == 0)[0]
        class_1_indices = np.where(binary_labels_flat == 1)[0]

        # Get class 1 samples
        class_1_samples = all_samples[class_1_indices]

        # Clustering class 0 samples
        class_0_samples = all_samples[class_0_indices]
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(class_0_samples)

        # For each cluster, get the indices of the samples that belong to that cluster
        cluster_indices = {}
        for cluster_id in range(num_clusters):
            cluster_indices[cluster_id] = np.where(kmeans.labels_ == cluster_id)[0]

        # Sample from each cluster
        sampled_class_0_indices = []
        samples_per_cluster = len(class_1_indices) // num_clusters  # Determine how many samples to take from each cluster

        for cluster_id, indices in cluster_indices.items():
            if len(indices) > samples_per_cluster:  # Ensure we don't sample more than available
                sampled_from_cluster = np.random.choice(indices, samples_per_cluster, replace=False)
            else:
                sampled_from_cluster = indices  # If there are fewer samples than requested, take all
            sampled_class_0_indices.extend(class_0_indices[sampled_from_cluster])

        # Combine the indices of class 1 and the sampled class 0
        undersampled_indices = np.concatenate([class_1_indices, sampled_class_0_indices])

        # Select undersampled data
        all_samples = np.expand_dims(all_samples, axis=1) 
        X_res = all_samples[undersampled_indices]
        y_res_binary = binary_labels[undersampled_indices].cpu().numpy()
        y_res_multiple = all_multiple_labels[undersampled_indices]

        # Convert back to tensors
        samples_tensor = torch.tensor(X_res, dtype=torch.float64).to(self.device)
        binary_labels_tensor = torch.tensor(y_res_binary, dtype=torch.float32).to(self.device)
        multiple_labels_tensor = torch.tensor(y_res_multiple, dtype=torch.float32).to(self.device)

        print("undersampled shapes: ", samples_tensor.shape, binary_labels_tensor.shape, multiple_labels_tensor.shape)

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