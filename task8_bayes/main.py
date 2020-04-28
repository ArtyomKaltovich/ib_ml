import copy
import random
import re

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import spacy
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from task8_bayes.bag_of_words import BoW, BowStem
from task8_bayes.bayes import NaiveBayes


def read_dataset(filename):
    file = open(filename, encoding="utf-8")
    x = []
    y = []
    for idx, line in enumerate(file):
        #if idx > 1000: break
        cl, sms = re.split("^(ham|spam)[\t\s]+(.*)$", line)[1:3]
        x.append(sms)
        y.append(cl)
    return np.array(x, dtype=np.str), np.array(y, dtype=np.str)


def get_precision_recall_accuracy(y_pred, y_true):
    classes = np.unique(list(y_pred) + list(y_true))
    true_positive = dict((c, 0) for c in classes)
    true_negative = dict((c, 0) for c in classes)
    false_positive = dict((c, 0) for c in classes)
    false_negative = dict((c, 0) for c in classes)
    for c_pred, c_true in zip(y_pred, y_true):
        for c in classes:
            if c_true == c:
                if c_pred == c_true:
                    true_positive[c] = true_positive.get(c, 0) + 1
                else:
                    false_negative[c] = false_negative.get(c, 0) + 1
            else:
                if c_pred == c:
                    false_positive[c] = false_positive.get(c, 0) + 1
                else:
                    true_negative[c] = true_negative.get(c, 0) + 1
    precision = dict((c, true_positive[c] / (true_positive[c] + false_positive[c])) for c in classes)
    recall = dict((c, true_positive[c] / (true_positive[c] + false_negative[c])) for c in classes)
    accuracy = sum([true_positive[c] for c in classes]) / len(y_pred)
    return precision, recall, accuracy


def plot_precision_recall(X_train, y_train, X_test, y_test, bow_method, plots_prefix, voc_sizes=range(4, 200, 5)):
    classes = np.unique(list(y_train) + list(y_test))
    precisions = dict([(c, []) for c in classes])
    recalls = dict([(c, []) for c in classes])
    accuracies = []
    for v in voc_sizes:
        bow = bow_method(X_train, voc_limit=v)
        X_train_transformed = bow.transform(X_train)
        X_test_transformed = bow.transform(X_test)
        classifier = NaiveBayes(0.001)
        classifier.fit(X_train_transformed, y_train)
        y_pred = classifier.predict(X_test_transformed)
        precision, recall, acc = get_precision_recall_accuracy(y_pred, y_test)
        for c in classes:
            precisions[c].append(precision[c])
            recalls[c].append(recall[c])
        accuracies.append(acc)

    def plot(x, ys, ylabel, to_file, legend=True):
        plt.figure(figsize=(12, 3))
        plt.xlabel("Vocabulary size")
        plt.ylabel(ylabel)
        plt.xlim(x[0], x[-1])
        plt.ylim(np.min(list(ys.values())) - 0.01, np.max(list(ys.values())) + 0.01)
        for c in ys.keys():
            plt.plot(x, ys[c], label="Class " + str(c))
        if legend:
            plt.legend()
        plt.tight_layout()
        plt.savefig(to_file)

    plot(voc_sizes, recalls, "Recall", f"{plots_prefix}_recall.png")
    plot(voc_sizes, precisions, "Precision", f"{plots_prefix}_precision.png")
    plot(voc_sizes, {"": accuracies}, "Accuracy", f"{plots_prefix}_accuracy.png", legend=False)


def most_importance(bag_of_words, X, y, plot_file_name):
    plt.clf()
    bow = bag_of_words(X, voc_limit=500)
    X_train_bow = bow.transform(X)
    predictor = NaiveBayes(0.001)
    predictor.fit(X_train_bow, y)
    result = pd.DataFrame()
    result["Word"] = bow.counts.keys()
    result["Probabilities difference"] = predictor.word_probs[0] - predictor.word_probs[1]
    result = result.nlargest(n=5, columns="Probabilities difference").append(
        result.nsmallest(n=5, columns="Probabilities difference")[::-1])
    sns.scatterplot(data=result, x="Word", y="Probabilities difference")
    plt.title(f"Positive difference - mostly in {predictor.labels[0]},\n"
                 f"negative difference - mostly in {predictor.labels[1]}")
    plt.tight_layout()
    plt.savefig(plot_file_name)


def main(bag_of_words, plots_prefix, X_train, X_test, y_train, y_test):
    bow = bag_of_words(X_train, voc_limit=1000)
    X_train_bow = bow.transform(X_train)
    X_test_bow = bow.transform(X_test)
    predictor = NaiveBayes(0.001)
    predictor.fit(X_train_bow, y_train)
    get_precision_recall_accuracy(predictor.predict(X_test_bow), y_test)
    plot_precision_recall(X_train, y_train, X_test, y_test, bag_of_words, plots_prefix)


if __name__ == '__main__':
    X, y = read_dataset("data/spam")
    #most_importance(BoW, X, y, "plot/most_important_words.png")
    #most_importance(BowStem, X, y, "plot/most_important_stems.png")
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
    main(BoW, "plot/non_stemmed", X_train, X_test, y_train, y_test)
    main(BowStem, "plot/stemmed", X_train, X_test, y_train, y_test)
