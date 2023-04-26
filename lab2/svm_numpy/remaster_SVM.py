import numpy as np
# to solve import issue
import sys, os
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PROJ_DIR))
from utils import load_cora
from tqdm import tqdm

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def generateMultiClassSVM(self, X, y):
        n_samples, n_features = X.shape

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        self.classifiers_ = []
        self.y_in_classifiers_ = []
        #for i in tqdm(range(n_classes), desc="Training Multi-Class SVM"):
        for i in range(n_classes):
        #for i in range(2):
            y_i = np.where(y == self.classes_[i], 1, -1)
            self.classifiers_.append(SVM(self.lr, self.lambda_param, self.n_iters))
            self.y_in_classifiers_.append(y_i)

    def fitMultiClassSVM(self, X):
        for i, classifier in tqdm(enumerate(self.classifiers_), desc="Training Multi-Class SVM"):
        #for i, classifier in enumerate(self.classifiers_):
            classifier.fit(X, self.y_in_classifiers_[i])

    def fit(self, X, y):
        n_samples, n_features = X.shape

        y_ = np.where(y <= 0, -1, 1)

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]
    
    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)

    def predictMultiClass(self, X):
        predictions = []
        for i in tqdm(range(len(self.classifiers_)), desc="Infering through multi-class SVM"):
        #for i in range(len(self.classifiers_)):
        #for i in range(2):
            pred_i = self.classifiers_[i].predict(X)
            predictions.append(pred_i)
        onehot_result = np.array(predictions).T
        return np.argmax(onehot_result, axis=1)


# Testing
if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_cora()

    clf = SVM()
    clf.generateMultiClassSVM(X_train, y_train)
    clf.fitMultiClassSVM(X_train)
    y_pred = clf.predictMultiClass(X_test)
    # count the number of y_pred == y_test
    accuracy = np.sum(y_pred == y_test) / len(y_test)
    # accuracy = np.mean(y_pred == y_test) # a trick for calculating acc
    print(accuracy)