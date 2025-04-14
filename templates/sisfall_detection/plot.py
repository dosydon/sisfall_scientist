import json
import os
import os.path as osp

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

# LOAD FINAL RESULTS:
datasets = ["sisfall"]
folders = os.listdir("./")
final_info = {}
results_info = {}
for folder in folders:
    if folder.startswith("run") and osp.isdir(folder):
        train_info = {}
        with open(osp.join(folder, "final_info.json"), "r") as f:
            final_info[folder] = json.load(f)

        results_dict = np.load(osp.join(folder, "all_results.npy"), allow_pickle=True).item()

        train_info['train_losses'] = results_dict["sisfall_detection_train_info"]["train_losses"]
        train_info['val_losses'] = results_dict["sisfall_detection_train_info"]["val_losses"]
        train_info['val_accs'] = results_dict["sisfall_detection_train_info"]["val_accs"]
        train_info['y_pred'] = results_dict["y_pred"]
        train_info['y_true'] = results_dict["y_true"]
        results_info[folder] = train_info

# CREATE LEGEND -- ADD RUNS HERE THAT WILL BE PLOTTED
labels = {
    "run_0": "Baselines",
}


# Create a programmatic color palette
def generate_color_palette(n):
    cmap = plt.get_cmap('tab20')
    return [mcolors.rgb2hex(cmap(i)) for i in np.linspace(0, 1, n)]


# Get the list of runs and generate the color palette
runs = list(labels.keys())
colors = generate_color_palette(len(runs))

def smooth_curve(points, factor=0.8):
    """
    Applies exponential moving average smoothing
    
    Args:
        points: Array of points to smooth
        factor: Smoothing factor (0 = no smoothing, 1 = flat line)
    """
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

plt.figure(figsize=(10, 6))
for i, run in enumerate(runs):
    train_info = results_info[run]
    smoothed_losses = smooth_curve(train_info['train_losses'])
    plt.plot(range(len(smoothed_losses)), smoothed_losses, label=run, color=colors[i])
plt.title(f"Training Loss")
plt.xlabel("Iteration")
plt.ylabel("Training Loss")
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.tight_layout()
plt.savefig(f"train_loss.png")
plt.close()


plt.figure(figsize=(10, 6))
for i, run in enumerate(runs):
    train_info = results_info[run]
    plt.plot(range(len(train_info['val_accs'])), train_info['val_accs'], label=labels[run], color=colors[i])
plt.title(f"Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Validation Accuracy")
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.tight_layout()
plt.savefig(f"val_accs.png")
plt.close()

def plot_confusion_matrix(y_true, y_pred, output_file="confustion_matrix.png", classes=['Normal', 'Alert', 'Fall'], label='Baseline'):
    """
    Plot confusion matrix using seaborn
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes,
                yticklabels=classes)
    plt.title(f'Confusion Matrix for {label}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(output_file)
    plt.close()

for i, run in enumerate(runs):
    y_true = results_info[run]['y_true']
    y_pred = results_info[run]['y_pred']
    plot_confusion_matrix(y_true, y_pred, output_file=f"confusion_matrix_{run}.png", label=labels[run])