import numpy as np
import pandas as pd
import numpy as np

class SVM:
    def __init__(self, C=1.0, kernel='linear', degree=3, gamma='auto', coef0=0.0, tol=1e-3, max_iter=-1):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter

        self._X = None
        self._y = None
        self._n_samples = None
        self._n_features = None
        self._alpha = None
        self._b = None
        self._E = None
        self._K = None

    def fit(self, X, y):
        self._X = X
        self._y = y
        self._n_samples, self._n_features = X.shape
        self._alpha = np.zeros(self._n_samples)
        self._b = 0
        self._E = np.zeros(self._n_samples)

        if self.gamma == 'auto':
            self.gamma = 1 / self._n_features

        self._K = self._kernel(self._X)

        iter = 0
        entire_set = True
        alpha_pairs_changed = 0

        while (iter < self.max_iter and ((alpha_pairs_changed > 0) or entire_set)):
            alpha_pairs_changed = 0

            if entire_set:
                for i in range(self._n_samples):
                    alpha_pairs_changed += self._examine_example(i)
                iter += 1
            else:
                non_bound_is = np.nonzero((self._alpha > 0) * (self._alpha < self.C))[0]
                for i in non_bound_is:
                    alpha_pairs_changed += self._examine_example(i)
                iter += 1

            if entire_set:
                entire_set = False
            elif alpha_pairs_changed == 0:
                entire_set = True

        return self

    def predict(self, X):
        y_pred = np.sign(self._decision_function(X))
        y_pred[y_pred == 0] = -1
        return y_pred

    def _decision_function(self, X):
        return np.sum(self._alpha * self._y * self._kernel(X, self._X), axis=1) + self._b

    def _kernel(self, X, Y=None):
        if Y is None:
            Y = X

        if self.kernel == 'linear':
            return np.dot(X, Y.T)
        elif self.kernel == 'poly':
            return (np.dot(X, Y.T) + self.coef0) ** self.degree
        elif self.kernel == 'rbf':
            return np.exp(-self.gamma * np.linalg.norm(X[:, np.newaxis] - Y, axis=2) ** 2)

    def _loss(self, y_true, y_pred):
        return np.maximum(0, 1 - y_true * y_pred)



from sklearn.metrics import accuracy_score 

if __name__ == "__main__":
    train = pd.read_pickle('./processed/corax_train.pkl')
    test = pd.read_pickle('./processed/corax_test.pkl')
    # y_train is the last column
    X_train = train.iloc[:, :-1]
    y_train = train.iloc[:, -1]
    X_test = test.iloc[:, :-1]
    y_test = test.iloc[:, -1]

    clf = SVM(C=1, kernel='rbf', gamma=0.1, max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)