from sklearn.metrics import confusion_matrix
import numpy as np


def weighted_precision(gt, pred):
    # for class 1
    tn, fp, fn, tp = confusion_matrix(gt, pred).ravel()
    precision = tp/(tp + fp)
    y = len(gt)
    y1 = np.count_nonzero(gt == 1)
    w1 = (y1/y)
    weighted_precision_1 = w1 * precision

    # for class 0
    tp, fn, fp, tn = confusion_matrix(gt, pred).ravel()
    precision_2 = tp / (tp + fp)
    y1_2 = np.count_nonzero(gt == 0)
    w2 = (y1_2 / y)
    weighted_precision_2 = w2 * precision_2

    weighted_average_precision = weighted_precision_1 + weighted_precision_2

    return weighted_average_precision


def weighted_recall(gt, pred):
    # for class 1
    tn, fp, fn, tp = confusion_matrix(gt, pred).ravel()
    recall = tp/(tp + fn)
    y = len(gt)
    y1 = np.count_nonzero(gt == 1)
    w1 = (y1/y)
    weighted_recall_1 = w1 * recall

    # for class 0
    tp, fn, fp, tn = confusion_matrix(gt, pred).ravel()
    recall_2 = tp/(tp + fn)
    y1_2 = np.count_nonzero(gt == 0)
    w2 = (y1_2 / y)
    weighted_recall_2 = w2 * recall_2

    weighted_average_recall = weighted_recall_1 + weighted_recall_2
    return weighted_average_recall


def weighted_f1score(gt, pred):
    weighted_average_precision = weighted_precision(gt, pred)
    weighted_average_recall = weighted_recall(gt, pred)
    f1score = (2*weighted_average_precision * weighted_average_recall)\
        / (weighted_average_precision + weighted_average_recall)
    return f1score


def accuracy_sc(gt, pred):
    tn, fp, fn, tp = confusion_matrix(gt, pred).ravel()
    accuracy_1 = (tp + tn)/(tp + tn + fp + fn)
    return accuracy_1


def weighted_specificity(gt, pred):
    # for class 1
    tn, fp, fn, tp = confusion_matrix(gt, pred).ravel()
    specificity_1 = tn/(tn + fp)
    y = len(gt)
    y1 = np.count_nonzero(gt == 1)
    w1 = (y1/y)
    weighted_specificity_1 = w1 * specificity_1

    # for class 0
    tp, fn, fp, tn = confusion_matrix(gt, pred).ravel()
    specificity_2 = tn/(tn + fp)
    y1_2 = np.count_nonzero(gt == 0)
    w2 = (y1_2 / y)
    weighted_specificity_2 = w2 * specificity_2

    weighted_average_specificity = weighted_specificity_1 + weighted_specificity_2
    return weighted_average_specificity

