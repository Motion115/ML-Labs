import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
# to solve import issue
import sys, os
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PROJ_DIR))
from utils import load_cora

class multiclassSVM:
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
        for i in range(n_classes):
            y_i = np.where(y == self.classes_[i], 1, -1)
            self.classifiers_.append(multiclassSVM(self.lr, self.lambda_param, self.n_iters))
            self.y_in_classifiers_.append(y_i)

    def fit(self, X, y):
        self.generateMultiClassSVM(X, y)
        for i, classifier in tqdm(enumerate(self.classifiers_), desc="Training Multi-Class multiclassSVM"):
            classifier.fitSingleClass(X, self.y_in_classifiers_[i])

    def fitSingleClass(self, X, y):
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
    
    def predictSingleClass(self, X):
        approx = np.dot(X, self.w) - self.b
        return approx
        #return np.sign(approx)

    def predict(self, X):
        predictions = []
        for i in tqdm(range(len(self.classifiers_)), desc="Infering through multi-class SVM"):
            pred_i = self.classifiers_[i].predictSingleClass(X)
            predictions.append(pred_i)
        onehot_result = np.array(predictions).T
        #print(onehot_result)
        return np.argmax(onehot_result, axis=1)


# Testing
if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_cora()

    clf = multiclassSVM()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    # use sklearn metrics to calculate accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)
