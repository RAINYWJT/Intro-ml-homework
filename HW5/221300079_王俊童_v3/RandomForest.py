import numpy as np

from sklearn.tree import DecisionTreeClassifier as dtclf
from sklearn.metrics import roc_auc_score

class RandomForestClassifier:
    def __init__(self, T, max_depth=4):
        self.T = T
        self.max_depth = max_depth
        self.m = None
        self.trees = []
        # You may add more fields as needed here.

    def fit(self, X, y):
        size, n_features = X.shape 
        if self.m is None:
            self.m = int(np.log2(n_features))
        for _ in range(self.T):
            bootstrap_indices = np.random.choice(size , size = size, replace=True)
            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]
            tree = dtclf(max_depth=self.max_depth, max_features=self.m)
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)

    def predict_proba(self, X):
        """
        Return the probability of each sample being class 1
        Args:
            X: np.array of shape (n_samples, n_features)
        Returns:
            proba: np.array of shape (n_samples,) where each value is the probability of the sample being class 1
        """
        all_probas = np.array([tree.predict_proba(X)[:, 1] for tree in self.trees])
        return np.mean(all_probas, axis=0)


    # ======== DO NOT MODIFY THIS ========
    def evaluate(self, X_train, y_train, X_eval, y_eval):
        self.fit(X_train, y_train)
        proba = self.predict_proba(X_eval)
        return roc_auc_score(y_true=y_eval, y_score=proba)
    # ====================================

    # You may implement some other utility methods if necessary