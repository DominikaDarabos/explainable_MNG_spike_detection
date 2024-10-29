import os, sys
module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

from Weighted_VP_model import *
sys.path.append(os.path.abspath('../Weighted_VP_model'))
from data_handling import *

from vpnet import *
from vpnet.vp_functions import *
from spike_classification import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker

import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
dtype = torch.float64

class OutputVisualizer:
    def __init__(self, model_name, dtype=torch.float32):
        """Initialize OutputVisualizer with model, dataset, and evaluation parameters."""
        self.window_size = 15
        self.overlapping_size = 11
        self.data_set = NeurographyDataset()
        model_path = os.path.join('../data', model_name)
        self.data_set.load_samples_and_labels_from_file(model_name)

        self.dataloaders = self.data_set.sequential_split_with_resampling()
        self._initialize_model()
        
        # Evaluate model on validation set
        decision_boundary = 0.5
        class_weights = torch.tensor([0.003, 0.997]).to(device)
        weighted_criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        criterion = VPLoss(weighted_criterion, 0.1)
        
        self.val_accuracy, self.val_loss, self.test_labels, self.test_predictions, self.test_probabilities = test(
            self.model, self.dataloaders['val_loader'], criterion, decision_boundary
        )
        compute_metrics(self.test_labels, self.test_predictions)
        
        # Reconstruct original validation samples and timestamps
        self.all_val_samples = torch.cat([x.cpu() for x, _, _ in self.dataloaders['val_loader']]).squeeze(1)
        self.original_val_samples = self.reconstruct_original_sequence(self.all_val_samples, self.window_size, self.overlapping_size)
        self.original_val_ts = self.reconstruct_original_sequence(self.dataloaders['val_timestamps'], self.window_size, self.overlapping_size)
        
        # Filter data within validation timestamp range
        start_ts, end_ts = self.original_val_ts[0], self.original_val_ts[-1]
        self.df_val_range = self._filter_data_within_range(self.data_set.all_differentiated_spikes, start_ts, end_ts)


    def _initialize_model(self):
        """Initialize and load the pre-trained model."""
        n_channels, n_in = self.data_set.samples[0].shape
        n_out = len(self.data_set.binary_labels_onehot[0])
        hidden1, weight_num = 3, 2
        affinities = [6 / n_in, -0.3606]
        weights = ((torch.rand(weight_num) - 0.5) * 8).tolist()

        self.model = VPNet(
            n_in, n_channels, hidden1, VPTypes.FEATURES,
            affinities + weights,
            WeightedHermiteSystem(n_in, hidden1, weight_num),
            [hidden1], n_out, device=device, dtype=dtype
        )
        model_file = 'trained_models/widnow_15_overlapping_11_hidden_3_nweight_2_id_1'
        self.model.load_state_dict(torch.load(model_file, weights_only=True))


    @staticmethod
    def _filter_data_within_range(data_frame, start_ts, end_ts):
        """Filter dataframe rows within the specified timestamp range."""
        start_index = data_frame['ts'].searchsorted(start_ts, side='left')
        end_index = data_frame['ts'].searchsorted(end_ts, side='right')
        return data_frame.iloc[start_index:end_index]


    @staticmethod
    def reconstruct_original_sequence(overlapping_windows, window_size, overlap_size):
        """Reconstruct the original sequence from overlapping windows."""
        overlapping_windows = np.asarray(overlapping_windows)
        stride = window_size - overlap_size
        original_length = (len(overlapping_windows) - 1) * stride + window_size

        reconstructed_sequence = np.zeros(original_length)
        count_array = np.zeros(original_length)

        for i in range(len(overlapping_windows)):
            start_index, end_index = i * stride, i * stride + window_size
            reconstructed_sequence[start_index:end_index] += overlapping_windows[i]
            count_array[start_index:end_index] += 1

        # Avoid division by zero and fill NaNs if present
        reconstructed_sequence[count_array > 0] /= count_array[count_array > 0]
        return np.nan_to_num(reconstructed_sequence)


    def plot_output_with_ground_truth(self):
        ts_windows_array = np.array(self.dataloaders['val_timestamps'])
        total_windows = len(self.original_val_ts)
        probabilities = self.test_probabilities[:, 1]


        # Initialize a list to store the highest probabilities
        # Define the index of the first matching timestamp window
        sample_size = 3000  # Number of windows to plot at a time
        custom_colors =  ['#808080', '#C0C0C0', 'khaki', '#FFA500', '#FF0000'] # Adjust colors as needed

        # Adjust bins for custom ranges
        bins = [0, 25, 50, 80, 90, 100]  # Custom bin edges

        # Set up labels for the new ranges
        percentage_labels = ['0-25%', '25-50%', '50-80%', '80-90%', '90-100%']
        legend_elements = [Line2D([0], [0], marker='o', color='w', label=label,
                                markerfacecolor=custom_colors[i], markersize=10) for i, label in enumerate(percentage_labels)]
        track_colors = { 1: 'black', 2: '#04BFAD', 3: 'green'}

        #for start_index in range(21300, 21300+sample_size*10, sample_size):
        for start_index in range(21300, 21300+sample_size*1, sample_size):
            highest_probabilities = []
            end_index = start_index + sample_size
            # Get relevant timestamps and their corresponding probabilities
            for ts in self.original_val_ts[start_index:end_index]:
                # Create a boolean mask for the current timestamp
                window_mask = np.any((ts_windows_array == ts), axis=1)
                window_indices = np.where(window_mask)[0]

                if window_indices.size > 0:
                    probs = probabilities[window_indices].numpy()
                    highest_probability = np.mean(probs)
                else:
                    highest_probability = np.nan

                highest_probabilities.append(highest_probability)

            # Create a DataFrame to store the results
            df_orig = pd.DataFrame({
                'Timestamp': self.original_val_ts[start_index:end_index],
                'Samples': self.original_val_samples[start_index:end_index],
                'Probability': highest_probabilities
            })

            # Plotting
            plt.figure(figsize=(20, 6))

            # Define color bins based on probabilities
            color_bins = np.digitize(df_orig['Probability'] * 100, bins=bins) - 1

            spike_labels = ['stimulus', 'Track3', 'Track4']
            track_legend_elements = [Line2D([0], [0], color=track_colors[i + 1], lw=2, label=spike_labels[i]) for i in range(len(spike_labels))]

            # Scatter plot for this window
            plt.scatter(df_orig['Timestamp'],
                        df_orig['Samples'],
                        c=[custom_colors[color_bins[i]] for i in range(len(color_bins))],  # Color according to bins
                        alpha=0.6, s=10)

            # Add vertical lines based on DataFrame timestamps and track values
            window_timestamps = df_orig['Timestamp'].values
            print("Plotting range:", window_timestamps.min(), window_timestamps.max())
            filtered_df_val_range = self.df_val_range[(self.df_val_range['ts'] >= window_timestamps.min()) & 
                                            (self.df_val_range['ts'] <= window_timestamps.max())]

            # Iterate over the filtered DataFrame
            y_top = df_orig['Samples'].max() * 1.1  # Slightly above the max sample value
            y_bottom = df_orig['Samples'].min() * 1.1
            for index, row in filtered_df_val_range.iterrows():
                color = track_colors.get(int(row['track']), 'black')  # Default to black if track color is not found
                
                # Plot a circle at the top position
                plt.scatter(row['ts'], y_top, color=color, s=100, alpha=0.8, edgecolor='black', linewidth=0.5)

                # Plot a circle at the bottom position
                plt.scatter(row['ts'], y_bottom, color=color, s=100, alpha=0.8, edgecolor='black', linewidth=0.5)
            #for index, row in filtered_df_val_range.iterrows():
                plt.axvline(x=row['ts'], color=track_colors[int(row['track'])], linestyle='--', lw=0.5)

            # Add legends
            plt.legend(handles=legend_elements + track_legend_elements, title="Probability and Track", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xlabel('Timestamp')
            plt.ylabel('Amplitude')
            plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(nbins=50))  # Increase the number of x-ticks
            plt.gca().xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.4f}'))
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show(block=True)
            plt.close()
if __name__ == '__main__':
    visualizer = OutputVisualizer('window_15_overlap_11_corrected.pkl')
    visualizer.plot_output_with_ground_truth()

