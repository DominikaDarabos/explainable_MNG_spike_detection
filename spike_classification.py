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
writer = SummaryWriter(log_dir="runs/tensorboard")

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

from Weighted_VP_model import *
sys.path.append(os.path.abspath('../Weighted_VP_model'))

from vpnet import *
from vpnet.vp_functions import *

def train(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, \
          n_epoch: int, optimizer: torch.optim.Optimizer, \
          criterion: Callable[[torch.Tensor | list[torch.Tensor]], torch.Tensor]) -> None:
    n_digits = len(str(n_epoch))
    for epoch in range(n_epoch):
        total_loss = 0
        total_accuracy = 0
        total_number = 0
        all_labels = []
        all_predictions = []
        total_true_positives = 0
        total_false_positives = 0
        total_false_negatives = 0
        for data in data_loader:
            x, labels, _ = data
            optimizer.zero_grad()
            outputs = model(x)
            # print("model output shapes: ", outputs[0].shape, labels.shape, classes.shape, _.shape)
            loss = criterion(outputs, labels)
            
            classes = labels.argmax(dim=-1)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            y = outputs[0] if isinstance(outputs, tuple) else outputs
            y_classes = y.argmax(dim=-1)
            total_accuracy += (classes == y_classes).sum().item()
            total_number += labels.size(0)
            total_true_positives += ((classes == 1) & (y_classes == 1)).sum().item()  # True Positives
            total_false_positives += ((classes == 0) & (y_classes == 1)).sum().item()  # False Positives
            total_false_negatives += ((classes == 1) & (y_classes == 0)).sum().item()  # False negatives
        
            all_labels.append(classes.cpu())
            all_predictions.append(y_classes.cpu())

        all_labels = torch.cat(all_labels)
        all_predictions = torch.cat(all_predictions)

        #precision, recall, _ = precision_recall_curve(all_labels, all_predictions)
        precision = total_true_positives / (total_true_positives + total_false_positives) if total_true_positives + total_false_positives > 0 else 0.0
        recall = total_true_positives / (total_true_positives + total_false_negatives) if total_true_positives + total_false_negatives > 0 else 0.0
        total_accuracy /= total_number / 100
        avg_loss = total_loss / len(data_loader)
        accuracy = total_accuracy / total_number
        writer.add_pr_curve('Precision-Recall Curve', all_labels, all_predictions, global_step=epoch)
        writer.add_scalar("Loss/epoch", avg_loss, epoch)
        writer.add_scalar("Accuracy/train", accuracy, epoch)
        print(f'Epoch: {epoch+1:0{n_digits}d} / {n_epoch}, accuracy: {total_accuracy:.2f}%, loss: {total_loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')

def test(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, \
         criterion: Callable[[torch.Tensor | list[torch.Tensor]], torch.Tensor]) -> tuple[float, float]:
    with torch.no_grad():
        total_loss = 0
        total_accuracy = 0
        total_number = 0
        all_labels = []
        all_predictions = []
        all_probabilities = []
        for data in data_loader:
            x, labels, _ = data
            outputs = model(x)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            classes = labels.argmax(dim=-1)
            y = outputs[0] if isinstance(outputs, tuple) else outputs
            y_classes = y.argmax(dim=-1)
            total_accuracy += (classes == y_classes).sum().item()
            total_number += labels.size(0)

            all_labels.append(classes.cpu())
            all_predictions.append(y_classes.cpu())
            all_probabilities.append(y.cpu())

        all_labels = torch.cat(all_labels)
        all_predictions = torch.cat(all_predictions)
        all_probabilities = torch.cat(all_probabilities)

        total_accuracy /= total_number / 100
        print("=" * 40)
        print(f'Val accuracy: {total_accuracy:.2f}%, loss: {total_loss:.4f}')
        print("=" * 80)
        return total_accuracy, total_loss, all_labels, all_predictions, all_probabilities

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
    epoch = 10
    lr = 0.01
    dtype = torch.float64
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    window_range = [20]
    overlapping_range = [15]

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

        #dataSet.plot_raw_data_window_by_label(0, 5)
        #dataSet.plot_raw_data_window_by_label(1, 5)
        #dataSet.plot_raw_data_window_by_label(2, 5)
        #dataSet.plot_raw_data_window_by_label(3, 5)
        
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
        criterion = VPLoss(torch.nn.CrossEntropyLoss(), 0.1)

        model = VPNet(n_in, n_channels, hidden1, VPTypes.FEATURES, affin + weight, WeightedHermiteSystem(n_in, hidden1, weight_num), [hidden1], n_out, device=device, dtype=dtype)
        total_count = dataSet.binary_labels.size(0)
        class_0_count = (dataSet.binary_labels == 0).sum().item()
        class_1_count = (dataSet.binary_labels == 1).sum().item()
        #for BCELoss
        #weights_tensor = torch.tensor([total_count / (class_0_count * 2), total_count / (class_1_count * 2)]).to(device)
        #for CrossEntrophy
        #weights_tensor = torch.tensor([total_count / class_0_count, total_count / class_1_count]).to(device)
        class_weights = torch.tensor([0.2, 0.8]).to(device)
        weighted_criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        criterion = VPLoss(weighted_criterion, 0.1)

        #focal = FocalLoss(gamma=5, alpha=[0.04, 0.96], size_average=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        #train(model, train_loader, epoch, optimizer, criterion)
        train(model, train_loader_under, epoch, optimizer, criterion)
        writer.flush()
        if isinstance(model, VPNet):
            print(*list(model.vp_layer.parameters()))
        #val_accuracy, val_loss, test_labels, test_predictions, test_probabilities = test(model, val_loader, criterion)
        val_accuracy, val_loss, test_labels, test_predictions, test_probabilities = test(model, val_loader_under, criterion)
        print("VALIDATION:")
        compute_metrics(test_labels, test_predictions)
        class_weights = torch.tensor([0.003, 0.997]).to(device)
        weighted_criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        criterion = VPLoss(weighted_criterion, 0.1)
        val_accuracy, val_loss, test_labels, test_predictions, test_probabilities = test(model, val_loader, criterion)
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
    writer.close()
