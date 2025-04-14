import os
import pandas as pd
from random import shuffle
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
import argparse
import time
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import matthews_corrcoef
import torch.nn.functional as F

# Model hyperparameters
HID_SIZE = 8
DROPOUT = 0.2
NUM_LAYERS = 2

# Training hyperparameters
SEQUENCE_LENGTH = 128
STRIDES = 60
NUM_EPOCHS = 5
BATCH_SIZE = 32

LAMBDA = 0.0
COST_MATRIX = torch.tensor([
    [0, 1, 5],     # Costs for true Normal
    [15, 0, 10],   # Costs for true Alert 
    [40, 10, 0]    # Costs for true Fall 
])

DATA_DIR = 'path/to/data'

def load_df():

    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train_data.csv'))
    test_df = pd.read_csv(os.path.join(DATA_DIR, 'test_data.csv'))
    return train_df, test_df

class SisFallDataset(Dataset):
    def __init__(self, df, sequence_length=SEQUENCE_LENGTH, strides=STRIDES):
        """
        Args:
            data_list: List of pandas DataFrames containing sensor data
            sequence_length: Length of each sequence (window size)
            stride: Number of steps to move the sliding window
        """
        self.dfs = []
        self.num_entries = []
        self.cumulative_entries = []
        self.num_windows = []
        self.cumulative_num_windows = []
        self.labels = []
        self.sequence_length = sequence_length
        self.strieds = strides
        
        # Process each DataFrame (each file) separately
        for file_name, sub_df in df.groupby('file_name'):
            m, n = sub_df.shape
            self.dfs.append(sub_df)
            self.num_entries.append(m)
            self.cumulative_entries.append(sum(self.num_entries))
            self.num_windows.append(max(m - sequence_length, 0))
            self.cumulative_num_windows.append(sum(self.num_windows))
            # Get sensor data and labels
        
    def __len__(self):
        return sum(self.num_windows) // self.strieds
    
    def __getitem__(self, idx):
        idx = idx * self.strieds
        # Find which DataFrame the index belongs to
        df_idx = 0
        inner_idx = 0
        for i, cumulative in enumerate(self.cumulative_num_windows):
            if idx < cumulative:
                df_idx = i
                inner_idx = idx - (self.cumulative_num_windows[i-1] if i > 0 else 0)
                break
        df = self.dfs[df_idx][inner_idx:inner_idx + self.sequence_length]
        sensor_data = df[['ADXL345_x', 'ADXL345_y', 'ADXL345_z', 
                        'ITG3200_x', 'ITG3200_y', 'ITG3200_z',
                        'MMA8451Q_x', 'MMA8451Q_y', 'MMA8451Q_z']].values.astype(np.float32)
        labels = df['label'].values.astype(np.int64)
           
        if len([x for x in labels if x == 2]) > 0.1 * self.sequence_length:
            sequence_label = 2
        elif len([x for x in labels if x == 1]) > 0.5 * self.sequence_length:
            sequence_label = 1
        else:
            sequence_label = 0
        
        return torch.FloatTensor(sensor_data), torch.LongTensor([sequence_label])
       

def create_dataloaders(train_data, test_data, batch_size=32, sequence_length=SEQUENCE_LENGTH):
    """
    Create train and test DataLoaders
    """
    train_dataset = SisFallDataset(train_data, sequence_length=sequence_length)
    test_dataset = SisFallDataset(test_data, sequence_length=sequence_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

class LSTMClassifier(nn.Module):
    def __init__(self, input_size=9, hidden_size=64, num_layers=2, num_classes=3):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=DROPOUT)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out
    
    def predict(self, probs):
        cost_matrix = COST_MATRIX.to(probs.device)
        batch_costs = torch.zeros((probs.shape[0], probs.shape[1]), device=probs.device)
#        print(probs.shape)
#        print(cost_matrix.shape)
        for i in range(probs.shape[1]):  # For each possible true class
            batch_costs[:, i] = torch.sum(probs * cost_matrix[i], dim=1)
        # Select class with minimum expected cost
        predicted = torch.argmin(batch_costs, dim=1)
        return predicted

class CostSensitiveLoss(nn.Module):
    def __init__(self, base_loss, l, cost_matrix):
        """
        Custom loss function that incorporates misclassification costs
        
        Args:
            cost_matrix: Tensor of shape (num_classes, num_classes) containing costs
                        where cost_matrix[i,j] is the cost of predicting class j
                        when true class is i
        """
        super().__init__()
        self.base_loss = base_loss
        self.l = l
        self.cost_matrix = cost_matrix
        
    def forward(self, logits, targets):
        base_loss = self.base_loss(logits, targets)
        # Get probabilities
        probs = F.softmax(logits, dim=1)
        
        # Calculate cost for each sample
        batch_size = logits.shape[0]
        costs = torch.zeros(batch_size, device=logits.device)
        
        for i in range(batch_size):
            # Get predicted probabilities for this sample
            sample_probs = probs[i]
            true_class = targets[i]
            
            # Multiply probabilities by costs for true class
            costs[i] = torch.sum(sample_probs * self.cost_matrix[true_class])
            
        return (1.0 - self.l) * base_loss + self.l * torch.mean(costs)

def get_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    size_mb = os.path.getsize("temp.p") / 1e6
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')
    return size_mb

def train_model(model, train_loader, test_loader, criterion, optimizer, 
                num_epochs=50, device='cuda'):
    """
    Train model with optional knowledge distillation from teacher
    Args:
        model: model to train
        train_loader: Training data loader
        test_loader: Validation data loader  
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Number of training epochs
        device: Device to train on
    """
    model = model.to(device)
    best_acc = 0.0
    val_accs = []
    train_losses = []
    val_losses = []
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(enumerate(train_loader), 
                   total=len(train_loader),
                   desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, (inputs, labels) in pbar:
            inputs = inputs.to(device)
            labels = labels.squeeze().to(device)
            
            optimizer.zero_grad()
            logits = model(inputs)
            
            # Regular training without distillation
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            train_losses.append(loss.item())
            probs = F.softmax(logits, dim=1)
            predicted = model.predict(probs)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 100 == 0:
                pbar.set_postfix({
                    'loss': running_loss/(batch_idx+1), 
                    'acc': 100.*correct/total,
                    'batch': f'{batch_idx+1}/{len(train_loader)}'
                })
        
        print("Validation phase")
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.squeeze().to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_losses.append(loss.item())
                probs = F.softmax(outputs, dim=1)
                predicted = model.predict(probs)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100.*val_correct/val_total
        val_accs.append(val_acc)
        print(f'Validation Accuracy: {val_acc:.2f}%')
        
        
    end_time = time.time()
    info = {
        'train_time': end_time - start_time,
        'val_accs': val_accs,
        'train_losses': train_losses,
        'val_losses': val_losses,
    }
    
    return model, info

def calculate_auc_scores(y_true, y_prob, output_file="roc_curves.png"):
    """
    Calculate AUC scores, precision, and recall for each class and plot ROC curves
    """
    # Binarize the labels
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    n_classes = 3
    
    # Calculate ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    precision = dict()
    recall = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Calculate precision and recall for each class
        y_pred_class = (y_prob[:, i] >= 0.5).astype(int)  # Using 0.5 as threshold
        tp = np.sum((y_true_bin[:, i] == 1) & (y_pred_class == 1))
        fp = np.sum((y_true_bin[:, i] == 0) & (y_pred_class == 1))
        fn = np.sum((y_true_bin[:, i] == 1) & (y_pred_class == 0))
        
        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green']
    classes = ['Normal', 'Alert', 'Fall']
    
    for i, color, cls in zip(range(n_classes), colors, classes):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'{cls} (AUC = {roc_auc[i]:0.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curves')
    plt.legend(loc="lower right")
    plt.savefig(output_file)
    plt.close()
    
    # Print metrics
    print("\nClassification Metrics:")
    print("-" * 50)
    print(f"{'Class':<10} {'AUC':>10} {'Precision':>10} {'Recall':>10}")
    print("-" * 50)
    for i, cls in enumerate(classes):
        print(f"{cls:<10} {roc_auc[i]:>10.4f} {precision[i]:>10.4f} {recall[i]:>10.4f}")
    
    metrics = {
        'auc': roc_auc,
        'precision': precision,
        'recall': recall
    }
    
    return metrics

def evaluate_model(model, test_loader, device):
    """
    Evaluate model and return predictions, probabilities and true labels
    with cost-sensitive prediction
    """
    model.eval()
    model.to(device)
    all_preds = []
    all_probs = []
    all_labels = []
    all_costs = []
    
    start_time = time.time()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.squeeze().to(device)
            
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            predicted = model.predict(probs)
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_costs.extend(COST_MATRIX[predicted].cpu().numpy())
    end_time = time.time()

    average_inference_time = (end_time - start_time) / len(test_loader)
    
    print(f"Accuracy: {100 * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels):.2f}%")
    print(f"Average inference time: {average_inference_time:.4f} seconds")
    
    return np.array(all_labels), np.array(all_preds), np.array(all_probs), np.array(all_costs), average_inference_time

def main():
    parser = argparse.ArgumentParser(description="Run experiment")
    parser.add_argument("--out_dir", type=str, default="run_0", help="Output directory")
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Move cost matrix to device
    cost_matrix = COST_MATRIX.to(device)
    
    # Load data
    train_data, test_data = load_df()
    print(train_data.shape)
    print(test_data.shape)
    train_loader, test_loader = create_dataloaders(train_data, test_data, 
                                                 batch_size=BATCH_SIZE, 
                                                 sequence_length=SEQUENCE_LENGTH)
    
    print("Train and test data loaded successfully.")
    model_name = "lstm"

    #class_weights = torch.FloatTensor([1.0, 10.0, 100.0]).to(device)
    base_loss = nn.CrossEntropyLoss()
    criterion = CostSensitiveLoss(base_loss, LAMBDA, cost_matrix)

    model = LSTMClassifier(hidden_size=HID_SIZE, num_layers=NUM_LAYERS)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Train the model
    print(f"Training the {model_name} model...")
    model, train_info = train_model(model, train_loader, test_loader, criterion, optimizer, 
                    num_epochs=NUM_EPOCHS, device=device)
    
    size_mb = get_size_of_model(model)
    
    # Add parameter count to info dictionary
    info = {}
    info['model_size_mb'] = size_mb  # Size in MB
    
    
    # Evaluate model and plot confusion matrix
    y_true, y_pred, y_prob, all_costs, average_inference_time = evaluate_model(model, test_loader, 'cpu')
    #plot_confusion_matrix(y_true, y_pred, 
    #                     output_file=f"{args.out_dir}/confusion_matrix_{model_name}.png")
    metrics = calculate_auc_scores(y_true, y_prob, 
                                 output_file=f"{args.out_dir}/roc_curves_{model_name}.png")
    
    # Add metrics to info dictionary
    info['metrics'] = {
        'normal': {
            'auc': float(metrics['auc'][0]),
            'precision': float(metrics['precision'][0]),
            'recall': float(metrics['recall'][0])
        },
        'alert': {
            'auc': float(metrics['auc'][1]),
            'precision': float(metrics['precision'][1]),
            'recall': float(metrics['recall'][1])
        },
        'fall': {
            'auc': float(metrics['auc'][2]),
            'precision': float(metrics['precision'][2]),
            'recall': float(metrics['recall'][2])
        }
    }
    mcc = matthews_corrcoef(y_true, y_pred)
    print(f"MCC: {mcc:.4f}")
    info['mcc'] = mcc
    
    info['average_cost'] = np.mean(all_costs)

    # Save results
    experiment_name = "sisfall_detection" 

    final_info = {
        experiment_name: {
            "means": {k:v for k, v in info.items()},
            "stderrs": {k:v for k, v in info.items()},
            "final_info_dict": {k: [v] for k, v in info.items()}
        }
    }

    all_results = {
        "sisfall_detection_final_info": final_info,
        "sisfall_detection_train_info": train_info,
        "y_true": y_true,
        "y_pred": y_pred,

    }

    with open(os.path.join(args.out_dir, "final_info.json"), "w") as f:
        json.dump(final_info, f)

    with open(os.path.join(args.out_dir, "all_results.npy"), "wb") as f:
        np.save(f, all_results)

if __name__ == '__main__':
    main()