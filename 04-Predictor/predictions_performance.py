import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    log_loss,
    roc_curve,
    precision_recall_curve,
    auc,
    ConfusionMatrixDisplay
)
import glob
import os

# Set the current working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load predictions and true labels from stored CSV files
predictions_df = pd.read_csv('../objects/FULL_stacked_data_with_preds.csv', parse_dates=True)
predictions_df = predictions_df[['t1_index', 'index', 'prediction', 'probability']]

# Set index in predictions_df
predictions_df.set_index('index', inplace=True)

# Load the actual target data
Y = pd.read_csv('../objects/Y_DATASET.csv')
Y.set_index('index', inplace=True)

# Add true target values and weights to predictions_df based on 'index'
predictions_df['target'] = Y['target']
predictions_df['weight_attr'] = Y['weight_attr']

# Initialize a list to store metrics for each test set and lists to accumulate overall values
performance_summary = []
all_y_true = []
all_y_pred = []
all_y_prob = []
all_weight_attr = []

# Define a function to calculate and store metrics for each test period
def evaluate_performance(y_true, y_pred, y_prob, weight_attr=None):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred, sample_weight=weight_attr),
        'macro_accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary', sample_weight=weight_attr),
        'recall': recall_score(y_true, y_pred, average='binary', sample_weight=weight_attr),
        'f1_score': f1_score(y_true, y_pred, average='binary', sample_weight=weight_attr),
        'macro_f1_score': f1_score(y_true, y_pred, average='macro'),
        'roc_auc': roc_auc_score(y_true, y_prob, sample_weight=weight_attr),
        'log_loss': log_loss(y_true, y_prob, sample_weight=weight_attr),
        'confusion_matrix': confusion_matrix(y_true, y_pred, sample_weight=weight_attr),
    }
    return metrics

# Load individual test set predictions and compute metrics
test_files = glob.glob('../objects/predictions_*.csv')

for counter, test_file in enumerate(test_files):
    test_preds = pd.read_csv(test_file)
    test_preds.rename(columns={'Unnamed: 0': 't1_index', 'Unnamed: 1': 'index'}, inplace=True)
    test_preds.set_index('index', inplace=True)
    
    test_preds['target'] = Y['target']
    test_preds['weight_attr'] = Y['weight_attr']
    
    test_preds = test_preds.dropna(subset=['target', 'prediction', 'probability'])

    y_true = test_preds['target'].values
    y_pred = test_preds['prediction'].values
    y_prob = test_preds['probability'].values
    weight_attr = test_preds['weight_attr'].values  # Use weights if available

    # Accumulate for overall performance calculation
    all_y_true.extend(y_true)
    all_y_pred.extend(y_pred)
    all_y_prob.extend(y_prob)
    all_weight_attr.extend(weight_attr)

    # Evaluate metrics for this test set
    metrics = evaluate_performance(y_true, y_pred, y_prob, weight_attr)
    metrics['test_period'] = f"Test Period {counter + 1}"
    
    tn, fp, fn, tp = metrics['confusion_matrix'].ravel()
    metrics.update({'true_negatives': tn, 'false_positives': fp, 'false_negatives': fn, 'true_positives': tp})
    del metrics['confusion_matrix']

    performance_summary.append(metrics)

# Calculate overall metrics
overall_metrics = evaluate_performance(
    np.array(all_y_true), 
    np.array(all_y_pred), 
    np.array(all_y_prob), 
    weight_attr=np.array(all_weight_attr)
)
overall_metrics['test_period'] = 'Overall'

tn, fp, fn, tp = overall_metrics['confusion_matrix'].ravel()
overall_metrics.update({'true_negatives': tn, 'false_positives': fp, 'false_negatives': fn, 'true_positives': tp})
del overall_metrics['confusion_matrix']

# Append overall metrics to the performance summary
performance_summary.append(overall_metrics)

# Convert the list of dictionaries to a DataFrame for a structured table
performance_df = pd.DataFrame(performance_summary)
performance_df.to_csv("../objects/performance_summary.csv", index=False)
print(performance_df)

# === Saving Descriptive Performance Plots for Overall Performance === #

# Directory for saving plots
plot_dir = "../objects/"

# 1. ROC Curve
fpr, tpr, _ = roc_curve(all_y_true, all_y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid()
plt.savefig(os.path.join(plot_dir, "roc_curve.png"))
plt.close()

# 2. Precision-Recall Curve
precision, recall, _ = precision_recall_curve(all_y_true, all_y_prob)
pr_auc = auc(recall, precision)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='purple', lw=2, label=f'Precision-Recall curve (AUC = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.grid()
plt.savefig(os.path.join(plot_dir, "precision_recall_curve.png"))
plt.close()

# 3. Confusion Matrix Heatmap
cm_display = ConfusionMatrixDisplay.from_predictions(all_y_true, all_y_pred, cmap="Blues")
plt.title('Confusion Matrix')
plt.grid(False)
cm_display.figure_.savefig(os.path.join(plot_dir, "confusion_matrix.png"))
plt.close()

# 4. Log-Loss Over Time
log_losses = []

for test_file in test_files:
    test_preds = pd.read_csv(test_file)
    test_preds.rename(columns={'Unnamed: 0': 't1_index', 'Unnamed: 1': 'index'}, inplace=True)
    test_preds.set_index('index', inplace=True)
    
    test_preds['target'] = Y['target']
    test_preds = test_preds.dropna(subset=['target', 'probability'])
    
    y_true = test_preds['target'].values
    y_prob = test_preds['probability'].values
    log_losses.append(log_loss(y_true, y_prob))

plt.figure(figsize=(8, 6))
plt.plot(range(1, len(log_losses) + 1), log_losses, marker='o', color='darkorange')
plt.xlabel('Test Period')
plt.ylabel('Log Loss')
plt.title('Log Loss Over Time')
plt.grid()
plt.savefig(os.path.join(plot_dir, "log_loss_over_time.png"))
plt.close()
