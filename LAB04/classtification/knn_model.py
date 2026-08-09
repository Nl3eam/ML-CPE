"""
Small wrapper around sklearn's KNeighborsClassifier
so main.py can call .fit / .predict / .score just like a normal model.
"""

from sklearn.neighbors import KNeighborsClassifier


class KNNModel:

    def __init__(self, k=5):
        self.k = k
        self.model = KNeighborsClassifier(n_neighbors=k)

    # -----------------------------------------------------------------
    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    # -----------------------------------------------------------------
    def predict(self, X):
        return self.model.predict(X)

    # -----------------------------------------------------------------
    def score(self, X, y):
        """accuracy = proportion of correct predictions"""
        return self.model.score(X, y)
