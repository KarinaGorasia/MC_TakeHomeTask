import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, confusion_matrix, classification_report

# function for data quality review
def data_quality_report(df: pd.DataFrame):
    """
    Returns a summary of missing values, data types, unique values, 
    and basic stats for numeric and categorical columns.
    """
    report = pd.DataFrame({
        'Data Type': df.dtypes,
        'Missing Values': df.isnull().sum(),
        'Missing %': (df.isnull().sum() / len(df)) * 100,
        'Unique Values': df.nunique(),
        'Sample Values': df.apply(lambda x: x.dropna().unique()[:3])
    })

    numeric_cols = df.select_dtypes(include=np.number).columns
    report['Mean'] = df[numeric_cols].mean()
    report['Std'] = df[numeric_cols].std()
    report['Min'] = df[numeric_cols].min()
    report['Max'] = df[numeric_cols].max()

    return report.sort_values(by='Missing %', ascending=False)

# function for stacked bar plot 
def plot_conversion_rate_stacked_bar(df, col, target, figsize=(4,3), colors=['#66c2a5', '#fc8d62']):
    """
    This function creates a horizontal 100% stacked bar chart to visualise 
    the subscription rate (or target proportion) by each category of a specified 
    categorical feature.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing the data.
    - col (str): Name of the categorical column to analyse.
    - target (str): Name of the binary target column indicating subscription 
    status. 
    - figsize (tuple): Figure size for the plot (default (4, 3)).
    - colours (list): List of colours for the bars representing non-subscribed and subscribed groups.
    """
    counts = df.groupby([col, target]).size().unstack(fill_value=0)
    props = counts.div(counts.sum(axis=1), axis=0)
    props = props.sort_values(by=1, ascending=False)

    ax = props.plot(kind='barh', stacked=True, figsize=figsize, color=colors)
    plt.xlabel('Proportion within category')
    plt.ylabel(col)
    plt.title(f'Conversion rate by {col} (ordered by subscriptions)')
    plt.legend().remove()
    plt.tight_layout()
    plt.show()

# evaluate classification performance across thresholds using precision, recall, and F1 score - find optimal cutoff for probs
def evaluate_threshold_metrics(y_true, y_probs, show_plot=False, save_path=None):
    """
    This function computes the precision-recall curve, identifies the optimal threshold 
    based on the highest F1 score, and optionally visualises and saves the precision, recall, 
    and F1 score trends across different thresholds.

    Parameters:
    - y_true (array): True binary labels (0 or 1).
    - y_probs (array): Predicted probabilities for the positive class.
    - show_plot (bool): optional (default=False). If True, displays the threshold evaluation plot. 
    - save_path (str): optional. File path where the precision-recall-F1 plot will be saved, if a path is provided.

    Returns:
    - best_threshold (float): The decision threshold that yields the highest F1 score.
    - best_f1 (float): The maximum F1 score achieved across all thresholds.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)
    best_threshold = thresholds[np.argmax(f1_scores)]
    best_f1 = np.max(f1_scores)

    plt.figure(figsize=(7, 5))
    plt.plot(thresholds, precision[:-1], label='Precision')
    plt.plot(thresholds, recall[:-1], label='Recall')
    plt.plot(thresholds, f1_scores, label='F1 Score', linestyle='--', color='purple')
    plt.axvline(x=best_threshold, color='red', linestyle=':', label=f'Best F1 Threshold = {best_threshold:.2f}')

    plt.xlabel('Decision Threshold')
    plt.ylabel('Score')
    plt.title('Precision, Recall & F1 vs Threshold')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    if show_plot:
        plt.show()

    plt.close()
    
    y_pred_custom = (y_probs >= best_threshold).astype(int)

    print(f"\nBest threshold: {best_threshold:.3f}")
    print(f"Best F1 score: {best_f1:.3f}\n")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred_custom))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_custom))

    return best_threshold, best_f1