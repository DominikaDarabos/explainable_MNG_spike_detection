from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import torch
from collections import Counter
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json


def compare_predictions_to_multilabels(all_binary_labels, all_multiple_labels, all_predictions):
    """
    Calculate how many TP, FN, FP, TN instances are for every label and return results as JSON.
    """
    multiple_num = len(torch.unique(all_multiple_labels))
    results = {}

    for i in range(multiple_num):
        label_indices = torch.where(all_multiple_labels == i)[0]
        binary_labels_for_label = all_binary_labels[label_indices]
        predictions_for_label = all_predictions[label_indices]

        if i == 0:  # For class 0: only FP and TN
            FP = torch.sum((binary_labels_for_label == 0) & (predictions_for_label == 1)).item()
            TN = torch.sum((binary_labels_for_label == 0) & (predictions_for_label == 0)).item()
            results[f"Label {i}"] = {
                "TN": TN,
                "FP": FP
            }
        else:  # For other classes: only TP and FN
            TP = torch.sum((binary_labels_for_label == 1) & (predictions_for_label == 1)).item()
            FN = torch.sum((binary_labels_for_label == 1) & (predictions_for_label == 0)).item()
            results[f"Label {i}"] = {
                "TP": TP,
                "FN": FN
            }
    print("=" * 40)
    print("Window-wise multi-class comparison")
    print("=" * 40)
    print(json.dumps(results, indent=4))
    print("=" * 40)
    return results


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

    common_metrics = {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "fpr": round(fpr, 4),
        "roc_auc": round(roc_auc, 4),
        "confusion_matrix": {
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
        }
    }

    print("=" * 40)
    print("Common Metrics")
    print("=" * 40)
    print(json.dumps(common_metrics, indent=4))
    print("=" * 40)
    return common_metrics



def generate_filtered_intervals(y_pred, multiple_labels, majority_voting = False, relative_distance_threshold = 0):
    def split_into_stimulus_pieces(multiple_labels):
        """
        Split the data by stimulus labels. If multiple are next to each other, the first one is used.
        """
        pieces = []
        start = 0
        i = 0

        while i < len(multiple_labels):
            if multiple_labels[i] == 1:
                pieces.append((start, i))
                start = i
                while i < len(multiple_labels) and multiple_labels[i] == 1:
                    i += 1
                continue
            i += 1
        pieces.append((start, i))
        return pieces

    def merge_positive_predictions(y_pred, false_factor=0):
        """
        Identify (start, length) pairs for positive predictions.
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

                if majority_voting and length == 1:
                    continue
                
                merged_indices.append((start, length))
            else:
                idx += 1
        return merged_indices

    def filter_intervals_by_pieces(intervals, pieces):
        """
        Keep only positive prediction intervals that are relatively close in consecutive pieces defined by stimulus.
        """
        valid_intervals = []
        
        for i in range(len(pieces) - 1):  
            start1, end1 = pieces[i]  
            start2, end2 = pieces[i + 1]

            intervals_piece_one = []
            intervals_piece_two = []

            for p_start, p_length in intervals:
                p_end = p_start + p_length

                # Interval in first piece
                if start1 <= p_start < end1:
                    rel_start = p_start - start1  
                    rel_end = p_end - start1  
                    intervals_piece_one.append((p_start, p_length, rel_start, rel_end))

                # Interval in second piece
                elif start2 <= p_start < end2:
                    rel_start = p_start - start2  
                    rel_end = p_end - start2  
                    intervals_piece_two.append((p_start, p_length, rel_start, rel_end))
                

            for p1_start, p1_length, rel1_start, rel1_end in intervals_piece_one:
                for p2_start, p2_length, rel2_start, rel2_end in intervals_piece_two:
                    # Check if intervals are within the max_gap OR overlap
                    if (rel1_end >= rel2_start and rel1_start <= rel2_end) or \
                        (abs(rel1_end - rel2_start) <= relative_distance_threshold) or \
                        (abs(rel2_end - rel1_start) <= relative_distance_threshold):
                        if (p1_start, p1_length) not in valid_intervals:
                            valid_intervals.append((p1_start, p1_length))
                        if (p2_start, p2_length) not in valid_intervals:
                            valid_intervals.append((p2_start, p2_length))

        valid_intervals.sort(key=lambda x: x[0])
        return valid_intervals

    pieces = split_into_stimulus_pieces(multiple_labels)
    pred_intervals = merge_positive_predictions(y_pred)

    if relative_distance_threshold != 0:
        pred_intervals = filter_intervals_by_pieces(pred_intervals, pieces)

    return pred_intervals



def compute_merged_metrics(y_true, y_pred, multiple_labels, majority_voting = False, relative_distance_threshold = 0):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    multiple_labels = multiple_labels.cpu().numpy()
    
    def ranges_overlap(x_start, x_end, y_start, y_end):
        return (x_start < y_end and x_end > y_start)

    
    def len_stats(group):
        length_counts = Counter([group_len for _, group_len in group])
        total_count = sum(length_counts.values())
        return {"sum": total_count, "length_counter": dict(length_counts)}

    pred_intervals = generate_filtered_intervals(y_pred, multiple_labels, majority_voting=majority_voting, relative_distance_threshold=relative_distance_threshold)
    true_intervals = generate_filtered_intervals(y_true, multiple_labels, majority_voting=False, relative_distance_threshold=0)
    
    tp, fp, fn = 0, 0, 0
    FP_indices, TP_indices, FN_indices = [], [], []
    
    for p_start, p_length in pred_intervals:
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
    
    results = {
        "metrics": {
            "recall": recall_merged,
        },
        "ground_truth_positive": len_stats(true_intervals),
        "true_positive": len_stats(TP_indices),
        "false_positive": len_stats(FP_indices),
        "false_negative": len_stats(FN_indices),
    }
    
    print("=" * 40)
    print("Merged Metrics")
    print("=" * 40)
    print(json.dumps(results, indent=4))
    print("=" * 40)
    
    return results


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