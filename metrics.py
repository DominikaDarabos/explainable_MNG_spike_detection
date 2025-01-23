from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import torch
from collections import Counter
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def compare_predictions_to_multilabels(all_binary_labels, all_multiple_labels, all_predictions):
    """
    Calculate how many TP, FN, FP, TN instances are for every label.
    """
    multiple_num = len(torch.unique(all_multiple_labels))
    TP_counts = torch.zeros(multiple_num)
    FN_counts = torch.zeros(multiple_num)
    FP_counts = torch.zeros(multiple_num)
    TN_counts = torch.zeros(multiple_num)

    for i in range(multiple_num):
        label_indices = torch.where(all_multiple_labels == i)[0]
        binary_labels_for_label = all_binary_labels[label_indices]
        predictions_for_label = all_predictions[label_indices]

        TP_counts[i] = torch.sum((binary_labels_for_label == 1) & (predictions_for_label == 1))
        FN_counts[i] = torch.sum((binary_labels_for_label == 1) & (predictions_for_label == 0))
        FP_counts[i] = torch.sum((binary_labels_for_label == 0) & (predictions_for_label == 1))
        TN_counts[i] = torch.sum((binary_labels_for_label == 0) & (predictions_for_label == 0))

    print("=" * 40)
    for i in range(multiple_num):
        print(f"Label {i}:")
        print(f"  TP: {TP_counts[i].item()}")
        print(f"  FN: {FN_counts[i].item()}")
        print(f"  FP: {FP_counts[i].item()}")
        print(f"  TN: {TN_counts[i].item()}\n")
    print("=" * 40)


def create_decision_ceratinty_boxplots(all_binary_labels, all_multiple_labels, all_predictions, all_probabilities):
    """
    Boxplots for model decision certainty examination.
    """
    binary_labels_np = all_binary_labels.cpu().numpy()
    predictions_np = all_predictions.cpu().numpy()
    
    all_multiple_labels_np = all_multiple_labels.cpu().numpy()

    probabilities = all_probabilities[:, 1].cpu().numpy()

    num_labels = len(torch.unique(all_multiple_labels))
    stats = {f'Label {i}': {
        'TP': [],
        'FP': [],
        'TN': [],
        'FN': [],
    } for i in range(num_labels)}


    for i in range(len(binary_labels_np)):
        true_label = all_multiple_labels_np[i]
        pred_label = predictions_np[i]
        prob = probabilities[i]

        for label in range(num_labels):
            if label == 0:
                if true_label == label and pred_label == 0:
                    stats[f'Label {label}']['TN'].append(prob)
                elif true_label == label and pred_label == 1:
                    stats[f'Label {label}']['FP'].append(prob)    
            else:
                if true_label == label and pred_label == 1:
                    stats[f'Label {label}']['TP'].append(prob)
                elif true_label == label and pred_label == 0:
                    stats[f'Label {label}']['FN'].append(prob)

    data = []
    labels = []
    colors = []

    for label in range(num_labels):
        for pred_quality in stats[f'Label {label}']:
            if len(stats[f'Label {label}'][pred_quality]) > 0:
                data.extend(stats[f'Label {label}'][pred_quality])
                formatted_label = f'Label {label} - {pred_quality}'.replace(' - ', '\n')
                labels.extend([formatted_label] * len(stats[f'Label {label}'][pred_quality]))
                if 'N' in pred_quality:
                    colors.extend(['blue'] * len(stats[f'Label {label}'][pred_quality]))
                elif 'P' in pred_quality:
                    colors.extend(['red'] * len(stats[f'Label {label}'][pred_quality]))

    df = pd.DataFrame({'Labels': labels, 'Data': data, 'Colors': colors})
    # Plot the boxplot
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    sns.boxplot(
        x='Labels',
        y='Data',
        hue='Colors',
        data=df,
        palette={'blue': 'blue', 'red': 'red'},
        dodge=False,
    )
    plt.title("Boxplot of probabilities for each label and category", fontsize=20)
    plt.ylabel("Probability", fontsize=20)
    plt.xlabel("Categories", fontsize=20)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.legend([], [], frameon=False)
    plt.tight_layout()
    plt.show()

    tp_indices = [i for i, label in enumerate(labels) if "TP" in label]
    filtered_tp_data = [data[i] for i in tp_indices]
    filtered_tp_labels = [labels[i] for i in tp_indices]

    plt.figure(figsize=(8, 6))
    sns.set_style("whitegrid")
    sns.boxplot(
        x=filtered_tp_labels,
        y=filtered_tp_data,
        palette=['red', 'red', 'red'],
        #showfliers=False  # Remove outlier circles
    )
    plt.xticks(rotation=90)
    plt.title("Boxplot of Probabilities for TP Categories")
    plt.ylabel("Probability")
    plt.tight_layout()
    plt.show()


def compute_common_metrics(y_true, y_pred):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()


    precision = precision_score(y_true, y_pred, pos_label=1)
    recall = recall_score(y_true, y_pred, pos_label=1)
    conf_matrix = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    roc_auc = roc_auc_score(y_true, y_pred)
    #f2 = (5 * precision * recall) / (4  * precision + recall) if (4  * precision + recall) > 0 else 0.0
    
    print("=" * 40)
    print("           MODEL METRICS          ")
    print("=" * 40)
    print(f"Precision     : {precision:.4f}")
    print(f"Recall        : {recall:.4f}")
    print(f"FPR           : {fpr:.4f}")
    print(f"ROC-AUC       : {roc_auc:.4f}")
    #print(f"F2-Score      : {f2:.4f}")
    print("=" * 40)
    print("       CONFUSION MATRIX           ")
    print("=" * 40)
    print(f"              Predicted")
    print(f"          {conf_matrix[0][0]}    {conf_matrix[0][1]}")
    print(f"Actual    {conf_matrix[1][0]}    {conf_matrix[1][1]}")
    
       
    return f"precision: {precision}, recall: {recall}, matrix: {conf_matrix}, roc_auc: {roc_auc}"



def compute_merged_metrics(y_true, y_pred):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    def ranges_overlap(x_start, x_end, y_start, y_end):
        return (x_start < y_end and x_end > y_start) #  or (y_start < x_end and y_end > x_start)

    def merge_positive_predictions(y_pred, false_factor=0):
        """
        Create a list filled with pairs. The first element of the pair shows where a positive window is after a negative one, and the second is for how long the consecutive windows are posititive afterwards.
        """
        merged_indices = []
        idx = 0
        while idx < len(y_pred):
            if y_pred[idx] == 1:
                start = idx
                length = 0
                gap = 0
                while idx < len(y_pred) and (y_pred[idx] == 1 or gap < false_factor):
                    if y_pred[idx] == 1:
                        length += 1
                        gap = 0
                    else:
                        gap += 1
                    idx += 1
                merged_indices.append((start, length))
            else:
                idx += 1
        return merged_indices


    def len_stats(group):
        print("Quantity: ", len(group))
        length_counts = {}

        for idx, group_len in group:
            if group_len in length_counts:
                length_counts[group_len] += 1
            else:
                length_counts[group_len] = 1
        
        print("Length : count")
        for length in sorted(length_counts.keys()):
            print(f"{length}: {length_counts[length]}")




    pred_intervals = merge_positive_predictions(y_pred)
    true_intervals = merge_positive_predictions(y_true)
    
    tp = 0
    fp = 0
    fn = 0
    pred_length_counts = Counter()
    true_length_counts = Counter()

    FP_indices = []
    TP_indices = []
    FN_indices = []

    for p_start, p_length in pred_intervals:
        pred_length_counts[p_length] += 1
        p_end = p_start + p_length
        overlap = False
        for t_start, t_length in true_intervals:
            t_end = t_start+t_length
            if ranges_overlap(p_start, p_end, t_start, t_end):
                overlap = True
                break
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
                break
        if not overlap:
            fn += 1
            FN_indices.append((t_start, t_length))

    recall_merged = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    print("=" * 40)
    print("MERGED METRICS")
    print("=" * 40)
    print("recall", recall_merged)
    print("-" * 40)
    print("GROUND TRUTH POSITIVE")
    GT_counter = dict(Counter(true_intervals))
    len_stats(GT_counter)

    print("-" * 40)
    print("TRUE POSITIVE")
    TP_counter = dict(Counter(TP_indices))
    len_stats(TP_counter)

    print("-" * 40)
    print("FALSE POSITIVE")
    FP_counter = dict(Counter(FP_indices))
    len_stats(FP_counter)

    print("-" * 40)
    print("FALSE NEGATIVE")
    FN_counter = dict(Counter(FN_indices))
    len_stats(FN_counter)

    print("-" * 40)
    print("-" * 40)



    # # Find the indices for TP, FN, FP
    # TP_indices = np.where((y_true == 1) & (y_pred == 1))[0]
    # FN_indices = np.where((y_true == 1) & (y_pred == 0))[0]
    # FP_indices = np.where((y_true == 0) & (y_pred == 1))[0]
    # GP_indices = np.where(y_true == 1)[0]

    # def group_indices(indices):
    #     if len(indices) == 0:
    #         return []

    #     groups = []
    #     group_start = indices[0]
    #     group_len = 1

    #     for i in range(1, len(indices)):
    #         if indices[i] - indices[i-1] <= 2:
    #             group_len += 1
    #         else:

    #             groups.append((group_start, group_len))
    #             group_start = indices[i]
    #             group_len = 1

    #     groups.append((group_start, group_len))
        
    #     return groups

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