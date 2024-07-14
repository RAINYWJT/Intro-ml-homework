import numpy as np

from sklearn.tree import DecisionTreeClassifier as dtclf
from sklearn.metrics import roc_auc_score

class AdaBoostClassifier:
    def __init__(self, T, max_depth=4):
        self.T = T
        self.max_depth = max_depth

        # You may add more fields as needed here.
        self.models = []
        self.alphas = []

    def fit(self, X, y):
        size, n_features = X.shape
        weights = np.full(size, 1/size)
        for t in range(self.T):
            model = dtclf(max_depth = self.max_depth)
            model.fit(X, y, sample_weight = weights)
            pre = model.predict(X)
            error = weights.dot(pre != y)
            if error > 0.5:
                break
            alpha = 0.5 * np.log((1 - error) / error) 
            weights *= np.exp(-alpha * y * pre)
            weights /= weights.sum()
            self.models.append(model)
            self.alphas.append(alpha)

    def predict_proba(self, X):
        """
        Return the probability of each sample being class 1
        Args:
            X: np.array of shape (n_samples, n_features)
        Returns:
            proba: np.array of shape (n_samples,) where each value is the probability of the sample being class 1
        """
        predictions = np.zeros(X.shape[0])
        for model, alpha in zip(self.models, self.alphas):
            predictions += alpha * model.predict(X)
        y_pred = np.sign(predictions)
        proba = (y_pred + 1) / 2 
        return proba


    # ======== DO NOT MODIFY THIS ========
    def evaluate(self, X_train, y_train, X_eval, y_eval):
        self.fit(X_train, y_train)
        proba = self.predict_proba(X_eval)
        return roc_auc_score(y_true=y_eval, y_score=proba)
    # ====================================

    # You may implement some other utility methods if necessary