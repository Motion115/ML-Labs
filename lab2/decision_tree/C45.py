import numpy as np
from tqdm import tqdm

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class C45:
    def __init__(self, min_samples_split=2, max_depth=100):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

    def _entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities))

    def _split(self, X, y, feature, threshold):
        left_mask = X[:, feature] < threshold
        right_mask = X[:, feature] >= threshold
        return X[left_mask], y[left_mask], X[right_mask], y[right_mask]

    def _best_split(self, X, y):
        best_gain_ratio = float('-inf')
        best_feature = None
        best_threshold = None
        for feature in tqdm( range(X.shape[1])):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                X_left, y_left, X_right, y_right = self._split(X, y, feature, threshold)
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                entropy_left = self._entropy(y_left)
                entropy_right = self._entropy(y_right)
                entropy = (len(y_left) * entropy_left + len(y_right) * entropy_right) / len(y)
                gain = self._entropy(y) - entropy
                split_info = -((len(y_left) / len(y)) * np.log2(len(y_left) / len(y)) + (len(y_right) / len(y)) * np.log2(len(y_right) / len(y)))
                gain_ratio = gain / split_info
                if gain_ratio > best_gain_ratio:
                    best_gain_ratio = gain_ratio
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        if n_samples >= self.min_samples_split and depth <= self.max_depth:
            feature, threshold = self._best_split(X, y)
            if feature is not None:
                X_left, y_left, X_right, y_right = self._split(X, y, feature, threshold)
                left_node = self._build_tree(X_left, y_left, depth + 1)
                right_node = self._build_tree(X_right, y_right, depth + 1)
                return Node(feature, threshold, left_node, right_node)
        return Node(value=np.argmax(np.bincount(y)))

    def fit(self, X, y):
        self.root = self._build_tree(X, y, 0)

    def _predict(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] < node.threshold:
            return self._predict(x, node.left)
        return self._predict(x, node.right)

    def predict(self, X):
        return np.array([self._predict(x, self.root) for x in X])
if __name__ == "__main__":
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score
    from utils import load_cora
    # 划分训练集和测试集
    X_train, y_train, X_test, y_test = load_cora()
    # 使用手写实现的C45
    our_nb = C45(max_depth=3)
    our_nb.fit(X_train, y_train)
    our_predictions = our_nb.predict(X_test)

    # 使用sklearn的朴素贝叶斯分类器
    sklearn_nb = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
    sklearn_nb.fit(X_train, y_train)
    sklearn_predictions = sklearn_nb.predict(X_test)

    # 比较准确率
    our_accuracy = accuracy_score(y_test, our_predictions)
    sklearn_accuracy = accuracy_score(y_test, sklearn_predictions)

    print(f"Our C45 DecisionTree Classifier accuracy: {our_accuracy}")
    print(f"Sklearn C45 DecisionTree Classifier accuracy: {sklearn_accuracy}")

    # Our C45 DecisionTree Classifier accuracy: 0.317
    # Sklearn C45 DecisionTree Classifier accuracy: 0.359