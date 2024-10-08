import os, sys
from typing import Callable
import torch
from data_handling import *
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

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
        for data in data_loader:
            x, labels, _ = data
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, labels)
            writer.add_scalar("Loss/train", loss, epoch)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            classes = labels.argmax(dim=-1)
            y = outputs[0] if isinstance(outputs, tuple) else outputs
            y_classes = y.argmax(dim=-1)
            total_accuracy += (classes == y_classes).sum().item()
            total_number += labels.size(0)

        total_accuracy /= total_number / 100
        print(f'Epoch: {epoch+1:0{n_digits}d} / {n_epoch}, accuracy: {total_accuracy:.2f}%, loss: {total_loss:.4f}')

        # Log per-epoch loss to TensorBoard
        avg_loss = total_loss / len(data_loader)
        writer.add_scalar("Loss/epoch", avg_loss, epoch)
        
        # Calculate and log per-epoch accuracy
        accuracy = total_accuracy / total_number
        writer.add_scalar("Accuracy/train", accuracy, epoch)

def test(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, \
         criterion: Callable[[torch.Tensor | list[torch.Tensor]], torch.Tensor]) -> tuple[float, float]:
    with torch.no_grad():
        total_loss = 0
        total_accuracy = 0
        total_number = 0
        all_labels = []
        all_predictions = []
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

        all_labels = torch.cat(all_labels)
        all_predictions = torch.cat(all_predictions)

        total_accuracy /= total_number / 100
        print(f'Val accuracy: {total_accuracy:.2f}%, loss: {total_loss:.4f}')
        return total_accuracy, total_loss, all_labels, all_predictions

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
    
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-Score: {f1:.4f}')
    print(f'Confusion Matrix:\n{conf_matrix}')  # TN FP
                                                # FN TP
    print(f'ROC-AUC: {roc_auc:.4f}')

def full_training():
    batch_size = 256
    epoch = 3
    lr = 0.01
    dtype = torch.float64
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # search
    #window_range = np.linspace(20, 50, 5)
    #overlapping_range = np.linspace(10, 40, 5)
    window_range = [48]
    overlapping_range = [36]

    good_params = []
    bad_params = []

    for i in range(len(window_range)):
        window_size_ = int(window_range[i])
        overlapping_size_ = int(overlapping_range[i])
        try:
            dataSet = NeurographyDataset()
            #dataSet.generate_raw_windows(window_size=window_size_, overlapping=overlapping_size_)
            #dataSet.generate_labels()
            #dataSet.write_samples_and_labels_into_file(f'window_{window_size_}_overlap_{overlapping_size_}.pkl')

            #dataSet.plot_raw_data_window_by_label(0, 5)
            #dataSet.plot_raw_data_window_by_label(1, 5)
            #dataSet.plot_raw_data_window_by_label(2, 5)
            #dataSet.plot_raw_data_window_by_label(3, 5)

            dataSet.load_samples_and_labels_from_file(f'window_{window_size_}_overlap_{overlapping_size_}.pkl')
            train_loader, val_loader, test_loader = dataSet.random_split_binary_and_multiple_dataloader()
            print("Multiple labels unique count: ", dataSet.multiple_labels.unique(return_counts=True))

            # torch.use_deterministic_algorithms(True, warn_only=True)
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = False
            torch.manual_seed(0)
            random.seed(0)
            np.random.seed(0)

            #samples0, labels0 = train_dataset[0]
            n_channels, n_in = dataSet.samples[0].shape
            n_out = len(dataSet.binary_labels_onehot[0])
            hidden1 = 3
            weight_num = 2
            affin = torch.tensor([6 / n_in, -0.3606]).tolist()
            #affin = torch.tensor([6 / n_in, -0.3606]).tolist()  #semioptimal
            weight = ((torch.rand(weight_num)-0.5)*8).tolist()
            #weight = [3]

            #model = VPNet(n_in, n_channels, 4, VPTypes.FEATURES, [0.1, 0], HermiteSystem(n_in, 4), [16], n_out, device=device, dtype=dtype)

            #criterion = VPLoss(torch.nn.CrossEntropyLoss(), 0.1)

            model = VPNet(n_in, n_channels, hidden1, VPTypes.FEATURES, affin + weight, WeightedHermiteSystem(n_in, hidden1, weight_num), [hidden1], n_out, device=device, dtype=dtype)
            class_weights = torch.tensor([0.01, 0.99]).to(device)
            weighted_criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
            criterion = VPLoss(weighted_criterion, 0.1)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            train(model, train_loader, epoch, optimizer, criterion)
            writer.flush()
            if isinstance(model, VPNet):
                print(*list(model.vp_layer.parameters()))
            val_accuracy, val_loss, test_labels, test_predictions = test(model, val_loader, criterion)
            print("Validation  metrics:")
            compute_metrics(test_labels, test_predictions)
            # TESTING
            # test_accuracy, test_loss = test(model, test_loader, criterion)
            print()
            good_params.append((window_size_, overlapping_size_))
        except Exception as e:
            bad_params.append((window_size_, overlapping_size_))
    print("Bad parameters: ", bad_params)
    print("Good parameters: ", good_params)

if __name__ == '__main__':
    dataSet = NeurographyDataset()
    #dataSet.load_samples_and_labels_from_file(f'window_{window_size_}_overlap_{overlapping_size_}.pkl')
    #dataSet.generate_raw_windows(window_size=48, overlapping=36)
    #dataSet.generate_labels()
    #dataSet.write_samples_and_labels_into_file('window_48_overlap_36.pkl')
    #dataSet.load_samples_and_labels_from_file('window_48_overlap_36.pkl')
    #train_loader, val_loader, test_loader = dataSet.random_split_binary_and_multiple_dataloader()
    #print("unique count: ", dataSet.multiple_labels.unique(return_counts=True))
    #dataSet.get_statistics_of_spikes()
    #dataSet.plot_raw_data_window_by_label(0, 5)
    #dataSet.plot_raw_data_window_by_label(1, 5)
    #dataSet.plot_raw_data_window_by_label(2, 5)
    #dataSet.plot_raw_data_window_by_label(3, 5)

    full_training()
    writer.close()
