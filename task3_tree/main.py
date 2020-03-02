from sklearn.model_selection import train_test_split

from common.data import read_cancer_dataset, read_spam_dataset
from task3_tree.draw import plot_roc_curve, draw_tree
from task3_tree.tree import DecisionTreeClassifier

X_train, X_test, y_train, y_test = train_test_split(*read_spam_dataset())
tree = DecisionTreeClassifier(max_depth=6, min_samples_leaf=20)
tree.fit(X_train, y_train)
print(f"train accuracy={sum(t == p for t, p in zip(tree.predict(X_train), y_train)) / len(y_train)}")
print(f"test accuracy={sum(t == p for t, p in zip(tree.predict(X_test), y_test)) / len(y_test)}")
plot_roc_curve(y_test, tree.predict_proba(X_test), save_path=r"plot/spam_rog_aug.png")
draw_tree(tree, save_path=r"plot/spam_tree.png")
