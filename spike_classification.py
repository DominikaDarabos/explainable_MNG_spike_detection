import os, sys
from typing import Callable
import torch
from data_handling import *
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix,
    roc_auc_score, precision_recall_curve, auc, average_precision_score
)
from collections import Counter
from focal_loss import *
import seaborn as sns
import torch.nn as nn
import time
import json
start_time = time.time()

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

from Weighted_VP_model import *
sys.path.append(os.path.abspath('../Weighted_VP_model'))

from vpnet import *
from vpnet.vp_functions import *

def train(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, \
          n_epoch: int, optimizer: torch.optim.Optimizer, \
          criterion: Callable[[torch.Tensor | list[torch.Tensor]], torch.Tensor], decision_boundary: float = 0.5) -> None:
    n_digits = len(str(n_epoch))
    for epoch in range(n_epoch):
        total_loss = 0
        total_accuracy = 0
        total_number = 0
        all_binary_labels = []
        all_predictions = []
        total_true_positives = 0
        total_false_positives = 0
        total_false_negatives = 0
        total_true_negatives = 0
        for data in data_loader:
            x, labels, multiple = data
            optimizer.zero_grad()
            outputs = model(x)

            classes = labels.argmax(dim=-1)
            loss = criterion(outputs, labels)
           
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            y = outputs[0] if isinstance(outputs, tuple) else outputs
            # y_classes = y.argmax(dim=-1)
            y_classes = (y[:, 1] > decision_boundary).float()
            # print(y[:10], y_classes[:10])
            total_accuracy += (classes == y_classes).sum().item()
            total_number += labels.size(0)

            total_true_positives += ((classes == 1) & (y_classes == 1)).sum().item()  # True Positives
            total_false_positives += ((classes == 0) & (y_classes == 1)).sum().item()  # False Positives
            total_false_negatives += ((classes == 1) & (y_classes == 0)).sum().item()  # False negatives
            total_true_negatives += ((classes == 0) & (y_classes == 0)).sum().item()  # True Negatives
        
            all_binary_labels.append(classes.cpu())
            all_predictions.append(y_classes.cpu())

        all_binary_labels = torch.cat(all_binary_labels)
        all_predictions = torch.cat(all_predictions)

        # final metrics
        precision = total_true_positives / (total_true_positives + total_false_positives) if total_true_positives + total_false_positives > 0 else 0.0
        recall = total_true_positives / (total_true_positives + total_false_negatives) if total_true_positives + total_false_negatives > 0 else 0.0
        total_accuracy /= total_number / 100
        avg_loss = total_loss / len(data_loader)


        sensitivity = total_true_positives / (total_true_positives + total_false_negatives) if total_true_positives + total_false_negatives > 0 else 0.0
        specificity = total_true_negatives / (total_true_negatives + total_false_positives) if total_true_negatives + total_false_positives > 0 else 0.0
        balanced_accuracy = 0.5 * (sensitivity + specificity)

        print(f'Epoch: {epoch+1:0{n_digits}d} / {n_epoch}, '
              f'accuracy: {total_accuracy:.2f}%, loss: {total_loss:.4f}, '
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
        all_predictions = []
        all_probabilities = []
        total_true_positives = 0
        total_false_positives = 0
        total_false_negatives = 0
        total_true_negatives = 0
        for data in data_loader:
            x, labels, multiple = data
            outputs = model(x)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            classes = labels.argmax(dim=-1)
            y = outputs[0] if isinstance(outputs, tuple) else outputs
            #y_classes = y.argmax(dim=-1)
            y_classes = (y[:, 1] > decision_boundary).float()
            total_accuracy += (classes == y_classes).sum().item()
            total_number += labels.size(0)

            total_true_positives += ((classes == 1) & (y_classes == 1)).sum().item()
            total_false_positives += ((classes == 0) & (y_classes == 1)).sum().item()
            total_false_negatives += ((classes == 1) & (y_classes == 0)).sum().item()
            total_true_negatives += ((classes == 0) & (y_classes == 0)).sum().item()


            multiple_classes = multiple.argmax(dim=-1)
            all_multiple_labels.append(multiple_classes.cpu())


            all_binary_labels.append(classes.cpu())
            all_predictions.append(y_classes.cpu())
            all_probabilities.append(y.cpu())

        all_binary_labels = torch.cat(all_binary_labels)
        all_multiple_labels = torch.cat(all_multiple_labels)
        all_predictions = torch.cat(all_predictions)
        all_probabilities = torch.cat(all_probabilities)

        # Weighted Balanced Accuracy
        sensitivity = total_true_positives / (total_true_positives + total_false_negatives) if total_true_positives + total_false_negatives > 0 else 0.0
        specificity = total_true_negatives / (total_true_negatives + total_false_positives) if total_true_negatives + total_false_positives > 0 else 0.0
        balanced_accuracy = 0.5 * (sensitivity + specificity)

        # Print comparison and probability statistics
        print("=" * 40)
        compare_predictions_to_multilabels(all_binary_labels, all_multiple_labels, all_predictions)
        print("=" * 40)
        probability_statistics_(all_binary_labels, all_multiple_labels, all_predictions, all_probabilities)

        # Final Accuracy and Loss
        total_accuracy /= total_number / 100
        print("=" * 40)
        print(f'Val accuracy: {total_accuracy:.2f}%, loss: {total_loss:.4f}')
        print(f'Weighted Balanced Accuracy: {balanced_accuracy:.4f}')
        print("=" * 80)
        return total_accuracy, total_loss, all_binary_labels, all_predictions, all_probabilities

def compare_predictions_to_multilabels(all_binary_labels, all_multiple_labels, all_predictions):
    TP_counts = torch.zeros(4)  # True Positives for each multilabel
    FN_counts = torch.zeros(4)  # False Negatives for each multilabel
    FP_counts = torch.zeros(4)  # False Positives for each multilabel
    TN_counts = torch.zeros(4)  # True Negatives for each multilabel

    # Loop through the multilabels
    for i in range(4):
        # Get the indices where multilabel is i
        label_indices = torch.where(all_multiple_labels == i)[0]
        
        # Filter the binary labels and predictions for this multilabel
        binary_labels_for_label = all_binary_labels[label_indices]
        predictions_for_label = all_predictions[label_indices]
        
        # True Positives: binary label = 1, prediction = 1
        TP_counts[i] = torch.sum((binary_labels_for_label == 1) & (predictions_for_label == 1))
        
        # False Negatives: binary label = 1, prediction = 0
        FN_counts[i] = torch.sum((binary_labels_for_label == 1) & (predictions_for_label == 0))
        
        # False Positives: binary label = 0, prediction = 1
        FP_counts[i] = torch.sum((binary_labels_for_label == 0) & (predictions_for_label == 1))
        
        # True Negatives: binary label = 0, prediction = 0
        TN_counts[i] = torch.sum((binary_labels_for_label == 0) & (predictions_for_label == 0))

    # Display the results for each multilabel
    for i in range(4):
        print(f"Label {i}:")
        print(f"  True Positives (TP): {TP_counts[i].item()}")
        print(f"  False Negatives (FN): {FN_counts[i].item()}")
        print(f"  False Positives (FP): {FP_counts[i].item()}")
        print(f"  True Negatives (TN): {TN_counts[i].item()}\n")


def probability_statistics_(all_binary_labels, all_multiple_labels, all_predictions, all_probabilities):
    binary_labels_np = all_binary_labels.cpu().numpy()
    predictions_np = all_predictions.cpu().numpy()
    
    # Check the shape of all_multiple_labels
    all_multiple_labels_np = all_multiple_labels.cpu().numpy()
    print(f"Shape of all_multiple_labels: {all_multiple_labels_np.shape}")

    probabilities_np_0 = all_probabilities[:, 0].cpu().numpy()  # Extract probabilities for class 0
    probabilities_np_1 = all_probabilities[:, 1].cpu().numpy()  # Extract probabilities for class 1

    # Initialize containers for statistics for each label
    num_labels = 4
    stats = {f'Label {i}': {
        'TP': [],
        'FP': [],
        'TN': [],
        'FN': [],
    } for i in range(num_labels)}

    # Iterate through all samples and classify them into TP, FP, TN, FN for each label
    for i in range(len(binary_labels_np)):
        true_label = all_multiple_labels_np[i]  # Multi-label true labels for this sample
        pred_label = predictions_np[i]
        prob = probabilities_np_1[i]

        for label_index in range(num_labels):
            # True Positives (TP)
            if label_index == 0:
                if true_label == label_index and pred_label == 0:
                    stats[f'Label {label_index}']['TN'].append(prob)
                
                # False Positives (FP)
                elif true_label == label_index and pred_label == 1:
                    stats[f'Label {label_index}']['FP'].append(prob)
                    
            else:
                if true_label == label_index and pred_label == 1:
                    stats[f'Label {label_index}']['TP'].append(prob)
                
                # False Negatives (FN)
                elif true_label == label_index and pred_label == 0:
                    stats[f'Label {label_index}']['FN'].append(prob)
    
    data = []
    labels = []
    colors = []
    # Calculate and print statistics for each label
    for label_index in range(num_labels):
        for category in stats[f'Label {label_index}']:
            if len(stats[f'Label {label_index}'][category]) > 0:
                data.extend(stats[f'Label {label_index}'][category])
                labels.extend([f'Label {label_index} - {category}'] * len(stats[f'Label {label_index}'][category]))
                if 'N' in category:
                    colors.extend(['blue'] * len(stats[f'Label {label_index}'][category]))
                elif 'P' in category:
                    colors.extend(['red'] * len(stats[f'Label {label_index}'][category]))
                else:
                    colors.extend(['gray'] * len(stats[f'Label {label_index}'][category]))  # Default color

    df = pd.DataFrame({'Labels': labels, 'Data': data, 'Colors': colors})
    # Plot the boxplot
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    sns.boxplot(
        x='Labels',
        y='Data',
        hue='Colors',  # Assigning the color column as hue
        data=df,
        palette={'blue': 'blue', 'red': 'red', 'gray': 'gray'},  # Mapping colors explicitly
        dodge=False,  # Ensures the boxes do not separate by hue
        #showfliers=False
    )
    plt.xticks(rotation=90)
    plt.title("Boxplot of Probabilities for Each Label and Category")
    plt.ylabel("Probability")
    plt.xlabel("Categories")
    plt.legend([], [], frameon=False)  # Disable legend
    plt.tight_layout()
    plt.show()


    tp_indices = [
        i for i, label in enumerate(labels)
        if any(f'Label {lbl} - TP' in label for lbl in [1, 2, 3])
    ]

    filtered_tp_data = [data[i] for i in tp_indices]
    filtered_tp_labels = [labels[i] for i in tp_indices]
    filtered_tp_colors = ['red'] * len(filtered_tp_data)  # All TP categories will be red

    tp_palette = {
    'Label 1 - TP': 'red',
    'Label 2 - TP': 'red',
    'Label 3 - TP': 'red'
    }

    # Create the boxplot for filtered TP data
    plt.figure(figsize=(8, 6))
    sns.set_style("whitegrid")
    sns.boxplot(
        x=filtered_tp_labels,
        y=filtered_tp_data,
        palette=tp_palette,  # Provide a complete palette mapping
        #showfliers=False  # Remove outlier circles
    )
    plt.xticks(rotation=90)
    plt.title("Boxplot of Probabilities for TP Categories (Labels 1, 2, 3)")
    plt.ylabel("Probability")
    plt.tight_layout()
    plt.show()

def group_indices(indices):
    if len(indices) == 0:
        return []

    groups = []
    group_start = indices[0]
    group_len = 1

    for i in range(1, len(indices)):
        # Check if current index is adjacent to the previous one (i.e., 1 or 2 indices apart)
        if indices[i] - indices[i-1] <= 2:
            group_len += 1
        else:

            groups.append((group_start, group_len))
            group_start = indices[i]
            group_len = 1

    groups.append((group_start, group_len))
    
    return groups

def ranges_overlap(x_start, x_end, y_start, y_end):
    return (x_start < y_end and x_end > y_start) #  or (y_start < x_end and y_end > x_start)

def merge_predictions(y_pred, false_factor=0):
    y_pred = np.array(y_pred)
    merged_indices = []

    i = 0
    while i < len(y_pred):
        if y_pred[i] == 1:
            start = i
            length = 0
            gap = 0
            while i < len(y_pred) and (y_pred[i] == 1 or gap < false_factor):
                if y_pred[i] == 1:
                    length += 1
                    gap = 0  # Reset the gap counter on encountering a `1`
                else:
                    gap += 1
                i += 1
            merged_indices.append((start, length))
        else:
            i += 1

    return merged_indices



def len_stats(group):
    print("Overall length: ", len(group))
    length_counts = {}

    for idx, group_len in group:
        if group_len in length_counts:
            length_counts[group_len] += 1
        else:
            length_counts[group_len] = 1
    
    print("-----")
    print("Distinct length : count")
    for length in sorted(length_counts.keys()):  # Sorting the keys
        print(f"{length}: {length_counts[length]}")


def compute_metrics(y_true, y_pred):
    # Convert tensors to NumPy arrays if necessary
    y_true = y_true.cpu().numpy()
    y_pred_prob = y_pred.cpu().numpy()
    
    y_pred = (y_pred_prob >= 0.8).astype(int)  # Convert to binary outcome
    false_negatives_indexes = np.where((y_true == 1) & (y_pred == 0))[0]

    # Print false negative indexes
    print(f'False Negative Indexes: {false_negatives_indexes}')

    # Precision, Recall, F1 for class 1 (signal)
    precision = precision_score(y_true, y_pred, pos_label=1)
    recall = recall_score(y_true, y_pred, pos_label=1)
    f1 = f1_score(y_true, y_pred, pos_label=1)

    # Confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = conf_matrix.ravel()

    # False Positive Rate
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # ROC-AUC
    roc_auc = roc_auc_score(y_true, y_pred_prob)

    #F-2 score = (1 + 2^2) * (precision * recall) / (2^2 * precision + recall)
    f2 = (5 * precision * recall) / (4  * precision + recall) if (4  * precision + recall) > 0 else 0.0
    
    # Print metrics in a structured format
    print("=" * 40)
    print("           MODEL METRICS          ")
    print("=" * 40)
    print(f"Precision     : {precision:.4f}")
    print(f"Recall        : {recall:.4f}")
    print(f"F2-Score      : {f2:.4f}")
    print(f"FPR           : {fpr:.4f}")
    print("=" * 40)
    print("       CONFUSION MATRIX           ")
    print("=" * 40)
    print(f"              Predicted")
    print(f"          {conf_matrix[0][0]}    {conf_matrix[0][1]}")
    print(f"Actual    {conf_matrix[1][0]}    {conf_matrix[1][1]}")
    print("=" * 40)
    print(f"ROC-AUC       : {roc_auc:.4f}")
    print("=" * 40)
    
       
    # # Find the indices for TP, FN, FP
    # TP_indices = np.where((y_true == 1) & (y_pred == 1))[0]
    # FN_indices = np.where((y_true == 1) & (y_pred == 0))[0]
    # FP_indices = np.where((y_true == 0) & (y_pred == 1))[0]
    # GP_indices = np.where(y_true == 1)[0]

    # # Group the indices for TP, FN, and FP
    # TP_groups = group_indices(TP_indices)
    # FN_groups = group_indices(FN_indices)
    # FP_groups = group_indices(FP_indices)
    # GT_groups = group_indices(GP_indices)


    # # print("True Positive Groups:", TP_groups)
    # # print("False Negative Groups:", FN_groups)
    # # print("False Positive Groups:", FP_groups)
    # # print("Ground Truth Groups:", GT_groups)


    # # Print results
    # print("=" * 40)
    # print("MERGED METRICS")
    # print("=" * 40)

    # print("GROUND TRUTH POSITIVE")
    # len_stats(GT_groups)
    # print("=" * 40)
    # print("TRUE POSITIVE")
    # len_stats(TP_groups)
    # print("=" * 40)
    # print("FALSE POSITIVE")
    # len_stats(FP_groups)
    # print("=" * 40)
    # print("FALSE NEGATIVE")
    # len_stats(FN_groups)
    # print("=" * 40)
    # print("=" * 40)

    pred_intervals = merge_predictions(y_pred)
    true_intervals = merge_predictions(y_true)
    
    tp = 0
    fp = 0
    fn = 0
    pred_length_counts = Counter()
    true_length_counts = Counter()

    FP_indices = []
    TP_indices = []
    FN_indices = []



    # Iterate over predictions. If it overlaps with ground truth, that is a TP. If not, that is a FP.
    for p_start, p_length in pred_intervals:
        pred_length_counts[p_length] += 1
        p_end = p_start + p_length
        overlap = False
        for t_start, t_length in true_intervals:
            t_end = t_start+t_length
            if ranges_overlap(p_start, p_end, t_start, t_end):
                overlap = True
        if overlap:
            tp += 1
            TP_indices.append((p_start, p_length))
        else:
            fp += 1
            FP_indices.append((p_start, p_length))

    for t_start, t_length in true_intervals:
        true_length_counts[t_length] += 1
        t_end = t_start + t_length

        overlap = False
        for p_start, p_length in pred_intervals:
            p_end = p_start+p_length
            if ranges_overlap(t_start, t_end, p_start, p_end):
                overlap = True

        if not overlap:
            fn += 1
            FN_indices.append((t_start, t_length))

    recall_merged = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    print("=" * 40)
    print("MERGED METRICS")
    print("=" * 40)
    print("RECALL", recall_merged)
    print("GROUND TRUTH POSITIVE")
    GT_counter = dict(Counter(true_intervals))
    len_stats(GT_counter)

    print("=" * 40)
    print("TRUE POSITIVE")
    TP_counter = dict(Counter(TP_indices))
    len_stats(TP_counter)

    print("=" * 40)
    print("FALSE POSITIVE")
    FP_counter = dict(Counter(FP_indices))
    len_stats(FP_counter)

    print("=" * 40)
    print("FALSE NEGATIVE")
    FN_counter = dict(Counter(FN_indices))
    len_stats(FN_counter)

    print("=" * 40)
    print("=" * 40)

    return f"precision: {precision}, recall: {recall}, matrix: {conf_matrix}, roc_auc: {roc_auc}"

def full_training():
    decision_boundary = 0.8
    epoch = 10
    lr = 0.01
    dtype = torch.float64
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    window_range = [15]
    overlapping_range = [11]
    for i in range(len(window_range)):
        window_size_ = int(window_range[i])
        overlapping_size_ = int(overlapping_range[i])
        print("PARAMS: ", window_size_, overlapping_size_)
        dataSet = NeurographyDataset()
        path = f'window_{window_size_}_overlap_{overlapping_size_}_corrected.pkl'
        full_path = os.path.join('../data', path)
        if os.path.exists(full_path):
            print("Loading the dataset")
            dataSet.load_samples_and_labels_from_file(path)
        else:
            print("Generating the dataset")
            dataSet.generate_raw_windows(window_size=window_size_, overlapping=overlapping_size_)
            #dataSet.generate_labels()
            dataSet.generate_labels_stimuli_relabel()
            dataSet.write_samples_and_labels_into_file(f'window_{window_size_}_overlap_{overlapping_size_}_corrected.pkl')
        # dataSet.plot_raw_data_window_by_label(0, 5)
        # dataSet.plot_raw_data_window_by_label(1, 5)
        # dataSet.plot_raw_data_window_by_label(2, 5)
        # dataSet.plot_raw_data_window_by_label(3, 5)


        dataloaders = dataSet.sequential_split_with_resampling()
        multiple_labels = np.array(dataSet.multiple_labels)
        raw_data_windows = np.array(dataSet.raw_data_windows)
        labels_of_interest = [0, 1, 2, 3]
        # Loop over each label and calculate the min and max values
        for label in labels_of_interest:
            # Filter windows where the current label matches
            label_filter = multiple_labels == label
            filtered_windows = raw_data_windows[label_filter]
            
            if filtered_windows.size > 0:
                # Calculate the overall min and max for the filtered windows
                overall_min = np.min(filtered_windows)
                overall_max = np.max(filtered_windows)
                print(f"Label {label}: Overall Min = {overall_min}, Overall Max = {overall_max}")
            else:
                print(f"No windows found for label {label}.")
        

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)

        n_channels, n_in = dataSet.samples[0].shape
        n_out = len(dataSet.binary_labels_onehot[0])
        hidden1 = 6
        weight_num = 4
        affin = torch.tensor([6 / n_in, -0.3606]).tolist()
        weight = ((torch.rand(weight_num)-0.5)*8).tolist()


        model = VPNet(n_in, n_channels, hidden1, VPTypes.FEATURES, affin + weight, WeightedHermiteSystem(n_in, hidden1, weight_num), [hidden1], n_out, device=device, dtype=dtype)
        class_weights = torch.tensor([0.3, 0.7]).to(device)
        print("loss weights: ", class_weights)
        weighted_criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        criterion = VPLoss(weighted_criterion, 0.1)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        train(model, dataloaders['train_loader_under'], epoch, optimizer, criterion)

        if isinstance(model, VPNet):
            print(*list(model.vp_layer.parameters()))

        class_weights = torch.tensor([0.003, 0.997]).to(device)
        weighted_criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        criterion = VPLoss(weighted_criterion, 0.1)
        val_accuracy, val_loss, test_labels, test_predictions, test_probabilities = test(model, dataloaders['val_loader'], criterion, decision_boundary)
        print("VALIDATION ON ORIGINAL:")
        metrics = compute_metrics(test_labels, test_predictions)
        # TESTING
        # test_accuracy, test_loss = test(model, test_loader, criterion)
        print()
        # torch.save(model.state_dict(), f'trained_models/widnow_{window_size_}_overlapping_{overlapping_size_}_hidden_{hidden1}_nweight_{weight_num}_id_1')


if __name__ == '__main__':
    #dataSet = NeurographyDataset()
    #dataSet.load_samples_and_labels_from_file(f'window_15_overlap_11.pkl')
    #dataSet.generate_raw_windows(window_size=20, overlapping=15)
    #dataSet.generate_labels()
    #dataSet.generate_labels_stimuli_relabel()
    #dataSet.write_samples_and_labels_into_file('window_15_overlap_11_corrected.pkl')
    #dataSet.load_samples_and_labels_from_file('window_25_overlap_15.pkl')
    #train_loader, val_loader, test_loader = dataSet.random_split_binary_and_multiple_dataloader()
    #print("unique count: ", dataSet.multiple_labels.unique(return_counts=True))
    #dataSet.get_statistics_of_spikes()
    #dataSet.plot_raw_data_window_by_label(0, 5)
    #dataSet.plot_raw_data_window_by_label(1, 5)
    #dataSet.plot_raw_data_window_by_label(2, 5)
    #dataSet.plot_raw_data_window_by_label(3, 5)
    full_training()
    # end_time = time.time()  # Record end time
    # elapsed_time = end_time - start_time  # Calculate elapsed time
    # print(f"Elapsed time: {elapsed_time:.4f} seconds")
