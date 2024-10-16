import os, sys
from typing import Callable
import torch
from data_handling import *
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from collections import Counter
from sklearn.metrics import precision_recall_curve
from focal_loss import *

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter(log_dir="runs/tensorboard")

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
        for data in data_loader:
            x, labels, _ = data
            optimizer.zero_grad()
            outputs = model(x)

            loss = criterion(outputs, labels)
            classes = labels.argmax(dim=-1)
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
        
            all_binary_labels.append(classes.cpu())
            all_predictions.append(y_classes.cpu())

        all_binary_labels = torch.cat(all_binary_labels)
        all_predictions = torch.cat(all_predictions)

        # final metrics
        precision = total_true_positives / (total_true_positives + total_false_positives) if total_true_positives + total_false_positives > 0 else 0.0
        recall = total_true_positives / (total_true_positives + total_false_negatives) if total_true_positives + total_false_negatives > 0 else 0.0
        total_accuracy /= total_number / 100
        avg_loss = total_loss / len(data_loader)
        accuracy = total_accuracy / total_number
        # writer.add_pr_curve('Precision-Recall Curve', all_binary_labels, all_predictions, global_step=epoch)
        # writer.add_scalar("Loss/epoch", avg_loss, epoch)
        # writer.add_scalar("Accuracy/train", accuracy, epoch)
        print(f'Epoch: {epoch+1:0{n_digits}d} / {n_epoch}, accuracy: {total_accuracy:.2f}%, loss: {total_loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')

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

            multiple_classes = multiple.argmax(dim=-1)
            all_multiple_labels.append(multiple_classes.cpu())


            all_binary_labels.append(classes.cpu())
            all_predictions.append(y_classes.cpu())
            all_probabilities.append(y.cpu())

        all_binary_labels = torch.cat(all_binary_labels)
        all_multiple_labels = torch.cat(all_multiple_labels)
        all_predictions = torch.cat(all_predictions)
        all_probabilities = torch.cat(all_probabilities)

        print("=" * 40)
        compare_predictions_to_multilabels(all_binary_labels, all_multiple_labels, all_predictions)
        print("=" * 40)
        probability_statistics(all_binary_labels, all_multiple_labels, all_predictions, all_probabilities)

        total_accuracy /= total_number / 100
        print("=" * 40)
        print(f'Val accuracy: {total_accuracy:.2f}%, loss: {total_loss:.4f}')
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

def probability_statistics(all_binary_labels, all_multiple_labels, all_predictions, all_probabilities):
    binary_labels_np = all_binary_labels.cpu().numpy()
    predictions_np = all_predictions.cpu().numpy()
    probabilities_np_0 = all_probabilities[:, 0].cpu().numpy()  # Extract probabilities for class 1
    probabilities_np_1 = all_probabilities[:, 1].cpu().numpy()  # Extract probabilities for class 1

    # Initialize containers for statistics
    stats = {
        'True Positives': [],
        'False Positives': [],
        'True Negatives': [],
        'False Negatives': [],
    }

    # Iterate through all samples and classify them into TP, FP, TN, FN
    for i in range(len(binary_labels_np)):
        true_label = binary_labels_np[i]
        pred_label = predictions_np[i]
        prob_0 = probabilities_np_0[i]  # This is the model's predicted probability for class 1
        prob_1 = probabilities_np_1[i]  # This is the model's predicted probability for class 1
        
        # True Positives (TP): Ground truth is 1, and the prediction is 1
        if true_label == 1 and pred_label == 1:
            stats['True Positives'].append(prob_1)
        
        # False Positives (FP): Ground truth is 0, but the prediction is 1
        elif true_label == 0 and pred_label == 1:
            stats['False Positives'].append(prob_1)
        
        # True Negatives (TN): Ground truth is 0, and the prediction is 0
        elif true_label == 0 and pred_label == 0:
            stats['True Negatives'].append(prob_1)
        
        # False Negatives (FN): Ground truth is 1, but the prediction is 0
        elif true_label == 1 and pred_label == 0:
            stats['False Negatives'].append(prob_1)
    for category in stats:
        if len(stats[category]) > 0:
            print(f"\nStatistics for {category}:")
            probs_tensor = torch.tensor(stats[category])
            print(f"Min: {probs_tensor.min():.4f}")
            print(f"Max: {probs_tensor.max():.4f}")
            print(f"mean: {probs_tensor.mean():.4f}")
            print(f"avg: {(probs_tensor.sum() / len(probs_tensor)):.4f}")
            print(f"median: {probs_tensor.median():.4f}")
            print(f"Std Dev: {probs_tensor.std():.4f}")
    

    categories = list(stats.keys())
    data = [torch.tensor(stats[category]) for category in categories]

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot a boxplot for each category
    ax.boxplot([d.numpy() for d in data], labels=categories, patch_artist=True)
    # Customize the plot
    ax.set_title('Statistics for Model Predictions')
    ax.set_ylabel('Probability')
    ax.legend(loc='upper right', fontsize='small')

    # Show plot
    plt.tight_layout()
    plt.show()

def compute_metrics(y_true, y_pred):
    # Convert tensors to NumPy arrays if necessary
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    
    # Precision, Recall, F1 for class 1 (signal)
    precision = precision_score(y_true, y_pred, pos_label=1)
    recall = recall_score(y_true, y_pred, pos_label=1)
    f1 = f1_score(y_true, y_pred, pos_label=1)
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    # ROC-AUC
    roc_auc = roc_auc_score(y_true, y_pred)
    
    #print(f'Precision: {precision:.4f}')
    #print(f'Recall: {recall:.4f}')
    #print(f'F1-Score: {f1:.4f}')
    #print(f'Confusion Matrix:\n{conf_matrix}')  # TN FP
    #                                            # FN TP
    #print(f'ROC-AUC: {roc_auc:.4f}')
    # Print metrics in a structured format
    print("=" * 40)
    print("           MODEL METRICS          ")
    print("=" * 40)
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print("=" * 40)
    print("       CONFUSION MATRIX           ")
    print("=" * 40)
    print(f"              Predicted")
    print(f"          {conf_matrix[0][0]}    {conf_matrix[0][1]}")
    print(f"Actual    {conf_matrix[1][0]}    {conf_matrix[1][1]}")
    print("=" * 40)
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print("=" * 40)

def full_training():
    decision_boundary = 0.95
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
        path = f'window_{window_size_}_overlap_{overlapping_size_}.pkl'
        full_path = os.path.join('../data', path)
        if os.path.exists(full_path):
            dataSet.load_samples_and_labels_from_file(f'window_{window_size_}_overlap_{overlapping_size_}.pkl')
        else:
            print("Generating the dataset")
            dataSet.generate_raw_windows(window_size=window_size_, overlapping=overlapping_size_)
            dataSet.generate_labels()
            dataSet.write_samples_and_labels_into_file(f'window_{window_size_}_overlap_{overlapping_size_}.pkl')

        # dataSet.plot_raw_data_window_by_label(0, 5)
        # dataSet.plot_raw_data_window_by_label(1, 5)
        # dataSet.plot_raw_data_window_by_label(2, 5)
        # dataSet.plot_raw_data_window_by_label(3, 5)
        
        #train_loader, val_loader, test_loader = dataSet.random_split_binary_and_multiple_dataloader()
        train_loader, val_loader, test_loader, train_loader_under, val_loader_under = dataSet.random_split_undersampling()
        #print("Multiple labels unique count: ", dataSet.multiple_labels.unique(return_counts=True))

        # torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)

        n_channels, n_in = dataSet.samples[0].shape
        n_out = len(dataSet.binary_labels_onehot[0])
        hidden1 = 3
        weight_num = 2
        affin = torch.tensor([6 / n_in, -0.3606]).tolist()
        #affin = torch.tensor([6 / n_in, -0.3606]).tolist()  #semioptimal
        weight = ((torch.rand(weight_num)-0.5)*8).tolist()

        #model = VPNet(n_in, n_channels, 4, VPTypes.FEATURES, [0.1, 0], HermiteSystem(n_in, 4), [16], n_out, device=device, dtype=dtype)
        #criterion = VPLoss(torch.nn.CrossEntropyLoss(), 0.1)

        model = VPNet(n_in, n_channels, hidden1, VPTypes.FEATURES, affin + weight, WeightedHermiteSystem(n_in, hidden1, weight_num), [hidden1], n_out, device=device, dtype=dtype)
        # total_count = dataSet.binary_labels.size(0)
        # class_0_count = (dataSet.binary_labels == 0).sum().item()
        # class_1_count = (dataSet.binary_labels == 1).sum().item()
        #for BCELoss
        #weights_tensor = torch.tensor([total_count / (class_0_count * 2), total_count / (class_1_count * 2)]).to(device)
        #for CrossEntrophy
        #weights_tensor = torch.tensor([total_count / class_0_count, total_count / class_1_count]).to(device)
        class_weights = torch.tensor([0.15, 0.85]).to(device)
        print("loss weights: ", class_weights)
        weighted_criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        criterion = VPLoss(weighted_criterion, 0.1)

        #focal = FocalLoss(gamma=5, alpha=[0.04, 0.96], size_average=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        #train(model, train_loader, epoch, optimizer, criterion)
        train(model, train_loader_under, epoch, optimizer, criterion)
        # writer.flush()
        if isinstance(model, VPNet):
            print(*list(model.vp_layer.parameters()))
        #val_accuracy, val_loss, test_labels, test_predictions, test_probabilities = test(model, val_loader, criterion)
        val_accuracy, val_loss, test_labels, test_predictions, test_probabilities = test(model, val_loader_under, criterion, decision_boundary)
        print("VALIDATION:")
        compute_metrics(test_labels, test_predictions)
        class_weights = torch.tensor([0.003, 0.997]).to(device)
        weighted_criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        criterion = VPLoss(weighted_criterion, 0.1)
        val_accuracy, val_loss, test_labels, test_predictions, test_probabilities = test(model, val_loader, criterion, decision_boundary)
        print("VALIDATION ON ORIGINAL:")
        compute_metrics(test_labels, test_predictions)
        # TESTING
        # test_accuracy, test_loss = test(model, test_loader, criterion)
        print()
        #torch.save(model.state_dict(), f'trained_models/widnow_{window_size_}_overlapping_{overlapping_size_}_hidden_{hidden1}_nweight_{weight_num}_id_1')

if __name__ == '__main__':
    #dataSet = NeurographyDataset()
    #dataSet.load_samples_and_labels_from_file(f'window_30_overlap_22.pkl')
    #dataSet.generate_raw_windows(window_size=48, overlapping=36)
    #dataSet.generate_labels()
    #dataSet.write_samples_and_labels_into_file('window_48_overlap_36.pkl')
    #dataSet.load_samples_and_labels_from_file('window_25_overlap_15.pkl')
    #train_loader, val_loader, test_loader = dataSet.random_split_binary_and_multiple_dataloader()
    #print("unique count: ", dataSet.multiple_labels.unique(return_counts=True))
    #dataSet.get_statistics_of_spikes()
    #dataSet.plot_raw_data_window_by_label(0, 5)
    #dataSet.plot_raw_data_window_by_label(1, 5)
    #dataSet.plot_raw_data_window_by_label(2, 5)
    #dataSet.plot_raw_data_window_by_label(3, 5)

    full_training()
    # writer.close()
