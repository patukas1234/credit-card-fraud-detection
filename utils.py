
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve, confusion_matrix
 

def get_eval_metrics(y, y_hat):
    precision, recall, f1, _ = precision_recall_fscore_support(y, y_hat, average = None)
    precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(y, y_hat, average = "weighted")
    return precision, recall, f1, precision_avg, recall_avg, f1_avg

def plot_model_performance_metrics(y, y_hat, y_proba, ax = None, **kwargs):
    cm = confusion_matrix(y, y_hat, labels = kwargs.get("labels", [0, 1]))
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    labels = kwargs.get("label_names", {0:"non fraud", 1:"fraud"}).values()
    sns.heatmap(cmn, annot=True, fmt='.2f', ax = ax[0], xticklabels=labels, yticklabels=labels)
    ax[0].set_ylabel('Actual')
    ax[0].set_xlabel('Predicted')
    ax[0].set_title(kwargs["model"])

    pos_probs = y_proba[:, 1]
    # fraud line as the proportion of the fraud class
    fraud = len(y[y==1]) / len(y)
    precision_arr, recall_arr, _ = precision_recall_curve(y, pos_probs)

    ax[1].plot([0, 1], [fraud, fraud], linestyle='--', label='Fraud')
    ax[1].plot(recall_arr, precision_arr, marker='.', label= kwargs['model'])
    ax[1].set_xlabel('Recall (Label: Fraud)')
    ax[1].set_ylabel('Precision (Label: Fraud)')
    ax[1].set_title(kwargs["model"])
    ax[1].legend()


