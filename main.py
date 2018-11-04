from utils.data import Data
from utils.classifiers import RandomForest, SVM, AlexNet, AutoEncoder
import itertools
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import time


def prepare_dataset(data, mode, minimum_instances=100, split=0.8, max_instances=None):
    """
    Organize dataset data and labels into testing and training vectors
    No data preprocessing is done in this function
    """

    print("Preparing Training and Testing dataset...")
    # training data and labels
    tr_data = []
    tr_labels = []
    tr_sizes = dict()
    # testing data and labels
    ts_data = []
    ts_labels = []
    ts_sizes = dict()

    if mode == 'anomaly':
        # training data consists of only normal traces
        x = data.get_instances('Normal')
        s = len(x)
        if max_instances is not None:
            x = x[:min(max_instances, s)]
            s = len(x)
        cut = int(s * split)

        tr_data.extend(x[:cut])
        tr_labels.extend(['Normal' for _ in x[:cut]])
        tr_sizes['Normal'] = len(x[:cut])
        ts_data.extend(x[cut:])
        ts_labels.extend(['Normal' for _ in x[cut:]])
        ts_sizes['Normal'] = len(x[cut:])

        # testing data consists of some normal traffic
        # the rest are attacks
        for label in data.get_labels():
            if label != 'Normal':
                x = data.get_instances(label)
                s = len(x)
                if s > minimum_instances:
                    if max_instances is not None:
                        x = x[:min(max_instances, s)]
                        s = len(x)
                    ts_data.extend(x)
                    ts_labels.extend([label for _ in x[:s]])
                    ts_sizes[label] = len(x[:s])

    elif mode == 'misuse':
        # split attack examples between training and testing using a 80%/20% ratio
        for label in data.get_labels():
            x = data.get_instances(label)
            s = len(x)
            if s > minimum_instances:
                if max_instances is not None:
                    x = x[:min(max_instances, s)]
                    s = len(x)
                cut = int(s * split)

                tr_data.extend(x[:cut])
                tr_labels.extend([label for _ in x[:cut]])
                tr_sizes[label] = len(x[:cut])

                ts_data.extend(x[cut:])
                ts_labels.extend([label for _ in x[cut:]])
                ts_sizes[label] = len(x[cut:])

    print("Train:", json.dumps(tr_sizes, indent=4))
    print("Test:", json.dumps(ts_sizes, indent=4))

    return tr_data, tr_labels, ts_data, ts_labels


def misuse(style, tr_data, tr_labels, ts_data, ts_labels):
    """Run the misuse IDS classifier experiment """
    classifier = None
    if style == "neural":
        classifier = AlexNet()
    elif style == "classic":
        classifier = RandomForest()
    assert(classifier is not None)

    train_start = time.time()
    classifier.train(tr_data, tr_labels)
    train_end = time.time()
    print("Training finished in %f" % (train_end - train_start))

    test_start = time.time()
    results = classifier.evaluate(ts_data, ts_labels)
    test_end = time.time()
    print("Testing finished in %f" % (test_end - test_start))

    classes = {cls: {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0} for cls in results.keys()}
    for cls, dict in results.items():
        for label, count in dict.items():
            if label == cls:
                classes[cls]['TP'] += count
                for c in results.keys():
                    if c != cls:
                        classes[c]['TN'] += count
            else:
                classes[cls]['FN'] += count
                classes[label]['FP'] += count
    for cls in classes.keys():
        metrics = classes[cls]
        metrics['Recall'] = metrics['TP'] / (metrics['TP'] + metrics['FN'])
        metrics['Precision'] = metrics['TP'] / (metrics['TP'] + metrics['FP'])
        metrics['F-Measure'] = 2 / ((1/metrics['Recall']) + (1/metrics['Precision']))
        metrics['Accuracy'] = (metrics['TP'] + metrics['TN']) / \
                              (metrics['FP'] + metrics['TP'] + metrics['FN'] + metrics['TN'])
        classes[cls] = metrics
    print("Results:", json.dumps(results, indent=4))
    print("Metrics:", json.dumps(classes, indent=4))

    plt.figure()
    plot_confusion_matrix(classifier.cm, sorted(results.keys()), normalize=True)
    plt.show()


def anomaly(style, tr_data, tr_labels, ts_data, ts_labels):
    """Run the anomaly IDS classifier experiment """
    classifier = None
    if style == "neural":
        classifier = AutoEncoder()
    elif style == "classic":
        classifier = SVM()
    assert(classifier is not None)

    train_start = time.time()
    classifier.train(tr_data, tr_labels)
    train_end = time.time()
    print("Training finished in %f" % (train_end - train_start))

    test_start = time.time()
    results = classifier.evaluate(ts_data, ts_labels)
    test_end = time.time()
    print("Testing finished in %f" % (test_end - test_start))

    # calculate notable metrics
    metrics = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}
    for key, value in results.items():
        if key == 'Normal':
            metrics['TP'] += value.get('Normal', 0)
            metrics['FP'] += value.get('Attack', 0)
        else:
            metrics['TN'] += value.get('Attack', 0)
            metrics['FN'] += value.get('Normal', 0)
    metrics['Recall'] = metrics['TP'] / (metrics['TP'] + metrics['FN'])
    metrics['Precision'] = metrics['TP'] / (metrics['TP'] + metrics['FP'])
    metrics['F-Measure'] = 2 / ((1/metrics['Recall']) + (1/metrics['Precision']))
    metrics['Accuracy'] = (metrics['TP'] + metrics['TN']) / \
                          (metrics['FP'] + metrics['TP'] + metrics['FN'] + metrics['TN'])

    # print comprehensive results
    print("Results:", json.dumps(results, indent=4))
    print("Metrics:", json.dumps(metrics, indent=4))


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    src: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


if __name__ == "__main__":

    # parse argument(s)
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--style', choices=['classic', 'neural'], required=True)
    args = parser.parse_args()

    # load data zip
    path = "data/optimized_attacks_normal.ZIP"
    data = Data(path)
    assert ("Normal" in data.get_labels())

    if args.mode == 'anomaly':
        anomaly(args.style, *prepare_dataset(data, args.mode))

    elif args.mode == 'misuse':
        misuse(args.style, *prepare_dataset(data, args.mode))
