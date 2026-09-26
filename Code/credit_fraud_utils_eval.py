import numpy as np
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)

def find_optimal_threshold(y_true, y_probs):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)

    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    return float(best_thresh), float(best_f1)

def evaluate_model(y_true, y_probs, threshold = 0.5):
    y_pred = (y_probs >= threshold).astype(int)

    pr_auc = average_precision_score(y_true, y_probs)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    results = {
        'threshold': threshold,
        'f1_score': f1,
        'pr_auc': pr_auc,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': cm
    }

    return results

def print_evaluation_report(model_name, results):
    print(f'\n === {model_name} Evalutaion Results ===')
    print(f'Applied Threshold:  {results['threshold']:.4f}')
    print(f'F1_Score:           {results['f1_score']:.4f}')
    print(f'PR-AUC:             {results['pr_auc']:.4f}')
    print(f'Precision:          {results['precision']:.4f}')
    print(f'Recall:             {results['recall']:.4f}')
    print('\nConfusion Matrix:')
    print(results['confusion_matrix'])
    print('=' * 40)