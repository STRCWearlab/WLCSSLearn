import numpy as np
from scipy.special import expit


def isolated_fitness_function_params(matching_scores, labels, thresholds, classes, parameter_to_optimize='f1_acc'):
    num_classes = len(classes)
    num_instances = len(matching_scores)
    true_positive = np.zeros([num_classes])
    false_positive = np.zeros([num_classes])
    true_negative = np.zeros([num_classes])
    false_negative = np.zeros([num_classes])
    for i in range(num_instances):
        for j in range(num_classes):
            act = classes[j]
            label = labels[i]
            test_matching_scores = matching_scores[i, j]
            if test_matching_scores >= thresholds[j] and act == label:
                true_positive[j] += 1
            elif test_matching_scores < thresholds[j] and act == label:
                false_negative[j] += 1
            elif test_matching_scores >= thresholds[j] and act != label:
                false_positive[j] += 1
            elif test_matching_scores < thresholds[j] and act != label:
                true_negative[j] += 1
    if parameter_to_optimize == 'acc':
        # Accuracy
        return (np.sum(true_positive) + np.sum(true_negative)) / (num_classes * num_instances)
    elif parameter_to_optimize == 'prec':
        # Precision
        tps = np.sum(true_positive)
        fps = np.sum(false_positive)
        if tps != 0 or fps != 0:
            return tps / (tps + fps)
        else:
            return 0
    elif parameter_to_optimize == 'recall':
        # Recall
        tps = np.sum(true_positive)
        fns = np.sum(false_negative)
        if tps != 0 or fns != 0:
            return tps / (tps + fns)
        else:
            return 0
    elif parameter_to_optimize == 'f1':
        tps = np.sum(true_positive)
        fps = np.sum(false_positive)
        fns = np.sum(false_negative)
        # F1
        if tps != 0 or fps != 0:
            precision = tps / (tps + fps)
        else:
            precision = 0
        if tps != 0 or fns != 0:
            recall = tps / (tps + fns)
        else:
            recall = 0
        if precision != 0 and recall != 0:
            f1 = 2 / (1 / recall + 1 / precision)
        else:
            f1 = 0
        return f1
    elif parameter_to_optimize == 'f1_acc':
        tps = np.sum(true_positive)
        fps = np.sum(false_positive)
        fns = np.sum(false_negative)
        # F1
        if tps != 0 or fps != 0:
            precision = tps / (tps + fps)
        else:
            precision = 0
        if tps != 0 or fns != 0:
            recall = tps / (tps + fns)
        else:
            recall = 0
        if precision != 0 and recall != 0:
            f1 = 2 / (1 / recall + 1 / precision)
        else:
            f1 = 0
        accuracy = (np.sum(true_positive) + np.sum(true_negative)) / (num_classes * num_instances)
        return f1 * accuracy
