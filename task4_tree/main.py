from sklearn.model_selection import train_test_split
import numpy as np
import pandas
import random
import matplotlib.pyplot as plt
import matplotlib
import copy
import seaborn as sns
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from time import perf_counter_ns

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

from common.data import read_cancer_dataset, read_spam_dataset
from task3_tree.draw import plot_roc_curve
from task3_tree.tree import DecisionTreeClassifier
from task4_tree.forest import RandomForestClassifier


def main():
    X_train, X_test, y_train, y_test = train_test_split(*read_spam_dataset())
    best_param, best_clr = _cross_val_param(X_train, y_train)
    print(best_param)
    best_clr.fit(X_train, y_train)
    y_pred = best_clr.predict_proba(X_test)
    y_pred_acc = np.array(y_pred > 0.5, dtype=int)
    plt.title(f"Accuracy {accuracy_score(y_test, y_pred_acc)}, roc auc {roc_auc_score(y_test, y_pred)}")
    plt.suptitle(f"Best params {best_param}")
    plot_roc_curve(y_test, y_pred)


def _cross_val_param(X_train, y_train):
    best_roc_auc = 0
    best_param = None
    best_clr = None

    for max_depth in [2, 3, 5, 7, 10]:
        forest = None
        for n_estimators in [100, 50, 30, 20, 10, 5]:
            result = []
            for mask in cross_val_mask(size=len(X_train), n=5):
                current_X_test, current_X_train, current_y_test, current_y_train = _cross_val_split(X_train, y_train, mask)
                if not forest:
                    forest = RandomForestClassifier(min_samples_leaf=10, n_estimators=n_estimators, max_depth=max_depth)
                    forest.fit(current_X_train, current_y_train)
                else:
                    forest.estimators = forest.estimators[:n_estimators]
                y_pred = forest.predict_proba(current_X_test)
                result.append(roc_auc_score(current_y_test, y_pred))
            cur_mean = np.mean(result)
            if cur_mean > best_roc_auc:
                best_param = dict(n_estimators=n_estimators, max_depth=max_depth)
                best_clr = forest
                best_roc_auc = cur_mean
    return best_param, best_clr


def _cross_val_split(X_train, y_train, mask):
    n_mask = ~mask
    current_X_train = X_train[n_mask]
    current_y_train = y_train[n_mask]
    current_X_test = X_train[mask]
    current_y_test = y_train[mask]
    return current_X_test, current_X_train, current_y_test, current_y_train


def cross_val_mask(size, n=5):
    assert n > 1
    mask = np.zeros(size, dtype=bool)
    end = 0
    step = size // n
    for i in range(n - 1):
        start = end
        end = start + step
        mask[start: end] = True
        yield mask
        mask[start: end] = False
    # process tail
    mask[end: size] = True
    yield mask


def feature_importance(rfc, X, y, accuracy):
    result = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        x = np.array(X)
        np.random.shuffle(x[:, i])
        result[i] = accuracy - np.mean(rfc.predict(x) == y)
    return result / sum(result)


def train_frameworks():
    X_train, X_test, y_train, y_test = train_test_split(*read_cancer_dataset())

    cat_acc, cat_time = _train_one_framework(CatBoostClassifier, X_test, X_train, y_test, y_train)
    lgbm_acc, lgbm_time = _train_one_framework(LGBMClassifier, X_test, X_train, y_test, y_train)

    fig, axs = plt.subplots(ncols=2)
    fig.suptitle("Default params")
    axs[0].set_ylabel("Accuracy")
    sns.barplot(x=[f"catboost\n{cat_acc}", f"lightgbm\n{lgbm_acc}"], y=[cat_acc, lgbm_acc], ax=axs[0])
    sns.barplot(x=[f"catboost\n{cat_time} s.", f"lightgbm\n{lgbm_time} s."], y=[cat_time, lgbm_time], ax=axs[1])
    plt.show()


def _train_one_framework(clf, X_test, X_train, y_test, y_train):
    start = perf_counter_ns()
    clf = clf()
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    stop = perf_counter_ns()
    acc = accuracy_score(y_test, y_pred)
    time = stop - start
    return acc, time / 1_000_000_000


def print_feature_importance():
    X, y = read_spam_dataset()
    rfc = RandomForestClassifier(n_estimators=100, max_depth=10)
    rfc.fit(X, y)
    accuracy = np.mean(rfc.predict(X) == y)
    print("Accuracy:", accuracy)
    importances = feature_importance(rfc, X, y, accuracy)
    print("Importance:", importances)
    print(np.take(X.columns, np.argsort(importances)[::-1]))
    # Accuracy: 0.9982425307557118
    # Importance: [0.03076923 0.06153846 0.         0.         0.01538462 0.
    #  0.         0.01538462 0.         0.         0.         0.
    #  0.01538462 0.55384615 0.01538462 0.         0.01538462 0.
    #  0.01538462 0.         0.         0.01538462 0.         0.2
    #  0.         0.01538462 0.01538462 0.01538462 0.         0.        ]
    # [13 23  1  0 18  4  7 12 16 14 27 21 26 25 19  8  2  3  5  6  9 20 10 11
    #  24 22 28 15 17 29]

    # Accuracy: 0.6346446424690285
    # Importance: [ 0.01714286  0.          0.          0.09714286  0.          0.
    #   0.          0.          0.          0.          0.          0.
    #   0.          0.          0.          0.          0.          0.
    #   0.          0.01142857  0.          0.25428571  0.00285714  0.
    #   0.          0.00285714  0.01714286  0.02857143  0.00285714  0.00285714
    #   0.01142857  0.45428571  0.00285714  0.05714286  0.00571429  0.
    #   0.          0.01428571  0.00285714  0.00571429  0.00285714  0.00857143
    #   0.00571429  0.00285714  0.          0.00285714 -0.01714286 -0.00285714
    #   0.          0.          0.          0.          0.          0.
    #   0.00571429  0.          0.        ]
    # Index(['word_freq_857', 'word_freq_font', 'word_freq_3d', 'word_freq_415',
    #        'word_freq_650', 'word_freq_george', 'word_freq_make',
    #        'word_freq_parts', 'word_freq_telnet', 'word_freq_credit',
    #        'word_freq_meeting', 'word_freq_original', 'capital_run_length_average',
    #        'word_freq_direct', 'word_freq_85', 'word_freq_hpl', 'word_freq_000',
    #        'word_freq_labs', 'word_freq_data', 'word_freq_lab', 'word_freq_pm',
    #        'word_freq_cs', 'word_freq_project', 'word_freq_edu', 'word_freq_1999',
    #        'word_freq_internet', 'word_freq_report', 'word_freq_people',
    #        'word_freq_will', 'word_freq_receive', 'word_freq_mail',
    #        'word_freq_order', 'word_freq_over', 'word_freq_remove',
    #        'word_freq_free', 'word_freq_our', 'char_freq_#', 'word_freq_all',
    #        'word_freq_address', 'word_freq_addresses', 'word_freq_you',
    #        'word_freq_business', 'word_freq_email', 'word_freq_technology',
    #        'char_freq_$', 'word_freq_your', 'char_freq_!', 'char_freq_[',
    #        'word_freq_money', 'word_freq_hp', 'char_freq_(', 'char_freq_:',
    #        'capital_run_length_longest', 'word_freq_re',
    #        'capital_run_length_total', 'word_freq_conference', 'word_freq_table'],
    #       dtype='object')


if __name__ == '__main__':
    #main()
    #plt.show()
    #print_feature_importance()
    train_frameworks()
