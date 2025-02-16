import os, sys
from typing import Callable
import torch
from MicroneurographyDataloader import *
from metrics import *
import time

start_time = time.time()

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

from _external.WHVPNet_pytorch.networks import *


def train(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, \
          n_epoch: int, optimizer: torch.optim.Optimizer, \
          criterion: Callable[[torch.Tensor | list[torch.Tensor]], torch.Tensor], decision_boundary: float = 0.5) -> None:
    n_digits = len(str(n_epoch))
    for epoch in range(n_epoch):
        total_loss = 0
        total_accuracy = 0
        total_length = 0

        total_true_positives = 0
        total_false_positives = 0
        total_false_negatives = 0
        total_true_negatives = 0

        for data in data_loader:
            input_data, binary_labels, multiple_labels = data
            optimizer.zero_grad()
            outputs = model(input_data)

            binary_classes = binary_labels.argmax(dim=-1)
            loss = criterion(outputs, binary_labels)
           
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            predicted_labels = outputs[0] if isinstance(outputs, tuple) else outputs
            predicted_classes = (predicted_labels[:, 1] > decision_boundary).float()

            total_accuracy += (binary_classes == predicted_classes).sum().item()
            total_length += binary_labels.size(0)
            total_true_positives += ((binary_classes == 1) & (predicted_classes == 1)).sum().item()
            total_false_positives += ((binary_classes == 0) & (predicted_classes == 1)).sum().item()
            total_false_negatives += ((binary_classes == 1) & (predicted_classes == 0)).sum().item()
            total_true_negatives += ((binary_classes == 0) & (predicted_classes == 0)).sum().item()


        precision = total_true_positives / (total_true_positives + total_false_positives) if total_true_positives + total_false_positives > 0 else 0.0
        fpr = total_false_positives / (total_false_positives + total_true_negatives) if (total_false_positives + total_true_negatives) > 0 else 0.0
        recall = total_true_positives / (total_true_positives + total_false_negatives) if total_true_positives + total_false_negatives > 0 else 0.0
        total_accuracy /= total_length / 100
        avg_loss = total_loss / len(data_loader)


        sensitivity = total_true_positives / (total_true_positives + total_false_negatives) if total_true_positives + total_false_negatives > 0 else 0.0
        specificity = total_true_negatives / (total_true_negatives + total_false_positives) if total_true_negatives + total_false_positives > 0 else 0.0
        balanced_accuracy = 0.5 * (sensitivity + specificity)

        print(f'Epoch: {epoch+1:0{n_digits}d} / {n_epoch}, '
              f'accuracy: {total_accuracy:.2f}%, loss: {total_loss:.4f}, FPR: {fpr:.4f} '
              f'Precision: {precision:.4f}, Recall: {recall:.4f}, '
              f'Balanced Accuracy: {balanced_accuracy:.4f}')

def test(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, \
         criterion: Callable[[torch.Tensor | list[torch.Tensor]], torch.Tensor], decision_boundary: float = 0.5) -> tuple[float, float]:
    with torch.no_grad():
        total_loss = 0
        total_accuracy = 0
        total_number = 0

        all_binary_labels = []
        all_multiple_labels = []
        all_predicted_classes = []
        all_predicted_probabilities = []

        total_true_positives = 0
        total_false_positives = 0
        total_false_negatives = 0
        total_true_negatives = 0
        for data in data_loader:
            input_data, binary_labels, multiple_labels = data
            outputs = model(input_data)
            loss = criterion(outputs, binary_labels)
            total_loss += loss.item()
            
            binary_classes = binary_labels.argmax(dim=-1)
            predicted_labels = outputs[0] if isinstance(outputs, tuple) else outputs
            predicted_classes = (predicted_labels[:, 1] > decision_boundary).float()

            total_accuracy += (binary_classes == predicted_classes).sum().item()
            total_number += binary_labels.size(0)
            total_true_positives += ((binary_classes == 1) & (predicted_classes == 1)).sum().item()
            total_false_positives += ((binary_classes == 0) & (predicted_classes == 1)).sum().item()
            total_false_negatives += ((binary_classes == 1) & (predicted_classes == 0)).sum().item()
            total_true_negatives += ((binary_classes == 0) & (predicted_classes == 0)).sum().item()


            multiple_classes = multiple_labels.argmax(dim=-1)
            all_multiple_labels.append(multiple_classes.cpu())
            all_binary_labels.append(binary_classes.cpu())
            all_predicted_classes.append(predicted_classes.cpu())
            all_predicted_probabilities.append(predicted_labels.cpu())

        all_binary_labels = torch.cat(all_binary_labels)
        all_multiple_labels = torch.cat(all_multiple_labels)
        all_predicted_classes = torch.cat(all_predicted_classes)
        all_predicted_probabilities = torch.cat(all_predicted_probabilities)

        sensitivity = total_true_positives / (total_true_positives + total_false_negatives) if total_true_positives + total_false_negatives > 0 else 0.0
        specificity = total_true_negatives / (total_true_negatives + total_false_positives) if total_true_negatives + total_false_positives > 0 else 0.0
        balanced_accuracy = 0.5 * (sensitivity + specificity)

        compare_predictions_to_multilabels(all_binary_labels, all_multiple_labels, all_predicted_classes)

        total_accuracy /= total_number / 100
        print("=" * 40)
        print(f'Accuracy: {total_accuracy:.2f}%, loss: {total_loss:.4f}')
        print(f'Weighted Balanced Accuracy: {balanced_accuracy:.4f}')
        return total_accuracy, total_loss, all_binary_labels, all_multiple_labels, all_predicted_classes, all_predicted_probabilities

def full_VPNet_training():
    decision_boundary = 0.8
    epoch = 10
    lr = 0.01
    dtype = torch.float64
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    window_size = 15
    overlapping_size = 11

    MNG_dataloader = MicroneurographyDataloader(raw_data_relative_path='../data/raw_data.csv',
                                                spikes_relative_path='../data/spike_timestamps.csv',
                                                stimulation_relative_path='../data/stimulation_timestamps.csv')
    MNG_dataloader.get_statistics_of_spikes()
    MNG_dataloader_filepath = f'window_{window_size}_overlap_{overlapping_size}_corrected.pkl'
    full_path = os.path.join('../data', MNG_dataloader_filepath)
    if os.path.exists(full_path):
        print("Dataset loading.")
        MNG_dataloader.load_samples_and_labels_from_file(MNG_dataloader_filepath)
    else:
        print("Dataset generating.")
        MNG_dataloader.generate_raw_windows(window_size=window_size, overlapping=overlapping_size)
        #MNG_dataloader.generate_labels()
        MNG_dataloader.generate_labels_stimuli_relabel(logigal_operator="or")
        MNG_dataloader.write_samples_and_labels_into_file(MNG_dataloader_filepath)


    dataloaders = MNG_dataloader.sequential_split_with_resampling(batch_size=1024, minor_upsample_count=25000, major_downsample_count=75000)

    MNG_dataloader.get_value_statistics_for_classes()
    MNG_dataloader.get_statistics_of_labels()

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    # WHVPNet params
    samples = torch.tensor(MNG_dataloader.raw_data_windows, dtype=torch.float64).unsqueeze(1)#.to(MNG_dataloader.device)
    n_channels, n_in = samples[0].shape
    n_out = len(MNG_dataloader.binary_labels_onehot[0])
    num_VP_features = 6
    num_weights = 4
    fcn_neurons = 6
    affin = torch.tensor([6 / n_in, -0.3606]).tolist()
    weight = ((torch.rand(num_weights)-0.5)*8).tolist()


    model = VPNet(n_in, n_channels, num_VP_features, VPTypes.FEATURES, affin + weight, WeightedHermiteSystem(n_in, num_VP_features, num_weights), [fcn_neurons], n_out, device=device, dtype=dtype)
    class_weights = torch.tensor([0.3, 0.7]).to(device)
    weighted_criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    criterion = VPLoss(weighted_criterion, 0.1)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train(model, dataloaders['train_loader_under'], epoch, optimizer, criterion)

    if isinstance(model, VPNet):
        print(*list(model.vp_layer.parameters()))

    class_weights = torch.tensor([0.003, 0.997]).to(device)
    weighted_criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    criterion = VPLoss(weighted_criterion, 0.1)
    print("=" * 40)
    print("VALIDATION/TESTING:")
    accuracy, loss, all_binary_labels, all_multiple_labels, all_predicted_classes, all_predicted_probabilities = test(model, dataloaders['val_loader'], criterion, decision_boundary)
    compute_common_metrics(all_binary_labels, all_predicted_classes)
    compute_merged_metrics(all_binary_labels, all_predicted_classes, all_multiple_labels, proximity_rule=False, latency_threshold=0)
    #create_decision_ceratinty_boxplots(all_binary_labels, all_multiple_labels, all_predicted_classes, all_predicted_probabilities)
    print()
    #torch.save(model.state_dict(), f'trained_models/widnow_{window_size_}_overlapping_{overlapping_size_}_hidden_{hidden1}_nweight_{weight_num}_neuron_24')


if __name__ == '__main__':
    full_VPNet_training()
    # MNG_dataloader = MicroneurographyDataloader()
    # filename = 'window_15_overlap_11_corrected.pkl'
    # MNG_dataloader.load_samples_and_labels_from_file(filename)
    # MNG_dataloader.generate_raw_windows(window_size=20, overlapping=15)
    # MNG_dataloader.generate_labels()
    # MNG_dataloader.generate_labels_stimuli_relabel()
    # MNG_dataloader.write_samples_and_labels_into_file(filename)

    # MNG_dataloader.get_statistics_of_spikes()

    # MNG_dataloader.plot_raw_data_window_by_label(0, 5)
    # MNG_dataloader.plot_raw_data_window_by_label(1, 5)
    # MNG_dataloader.plot_raw_data_window_by_label(2, 5)
    # MNG_dataloader.plot_raw_data_window_by_label(3, 5)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Running time: {elapsed_time:.4f} seconds")