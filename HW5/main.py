import numpy as np
import matplotlib.pyplot as plt
# You can import other methods from sklearn, if needed
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score
from RandomForest import RandomForestClassifier
from AdaBoost import AdaBoostClassifier


def load_dataset():
    X_train = np.genfromtxt('./adult_dataset/X_train.txt', delimiter=' ')
    y_train = np.genfromtxt('./adult_dataset/y_train.txt', delimiter=' ')
    X_test  = np.genfromtxt('./adult_dataset/X_test.txt', delimiter=' ')
    y_test  = np.genfromtxt('./adult_dataset/y_test.txt', delimiter=' ')

    return X_train, y_train, X_test, y_test

def make_plot(X_train, y_train):
    fold_num = 5
    max_clf_num = 20

    auc_adaboost = []
    auc_rf = []
    skf = StratifiedKFold(n_splits=fold_num)

    for i in range(1, max_clf_num + 1):
        ada_clf = AdaBoostClassifier(T = i)
        rf_clf = RandomForestClassifier(T = i)

        auc_adaboost_fold = []
        auc_rf_fold = []

        for train_idx, val_idx in skf.split(X_train, y_train):
            X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
            y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

            ada_clf.fit(X_train_fold, y_train_fold)
            rf_clf.fit(X_train_fold, y_train_fold)

            y_pred_ada = ada_clf.predict_proba(X_val_fold)
            y_pred_rf = rf_clf.predict_proba(X_val_fold)

            auc_adaboost_fold.append(roc_auc_score(y_val_fold, y_pred_ada))
            auc_rf_fold.append(roc_auc_score(y_val_fold, y_pred_rf))

        auc_adaboost.append(np.mean(auc_adaboost_fold))
        auc_rf.append(np.mean(auc_rf_fold))

        # print(auc_adaboost)
        # print(auc_rf)
    # print(max(auc_adaboost))
    # print(max(auc_rf))
    plt.figure(figsize=(9, 6))
    plt.plot(range(1, max_clf_num + 1), auc_adaboost, label="AdaBoost")
    plt.plot(range(1, max_clf_num + 1), auc_rf, label="Random Forest")
    plt.xlabel("Number of Base Classifiers")
    plt.ylabel("Average AUC")
    plt.title("AUC vs. Number of Base Classifiers")
    plt.legend()
    plt.savefig("evaluation.png")

if __name__ == "__main__":

    X_train, y_train, X_test, y_test = load_dataset()


    rf_clf = RandomForestClassifier(T=10)
    result = rf_clf.evaluate(X_train, y_train, X_test, y_test)
    print(f"Random Forest AUC = {result:.3f}")

    ad_clf = AdaBoostClassifier(T=10)
    result = ad_clf.evaluate(X_train, y_train, X_test, y_test)
    print(f"AdaBoost AUC = {result:.3f}")

    # ==== Start of your plotting code ====

    make_plot(X_train, y_train)







