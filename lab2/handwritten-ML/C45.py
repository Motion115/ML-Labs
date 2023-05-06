# 手写C45
import numpy as np
from tqdm import tqdm
# to solve import issue
import sys, os
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PROJ_DIR))
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class C45DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None
    # 计算熵
    def _entropy(self, y):
        # y中每个不同的标签出现的次数
        _, counts = np.unique(y, return_counts=True)
        # 每个标签出现的概率
        probabilities = counts / len(y)
        # 使用公式计算熵并返回
        return -np.sum(probabilities * np.log2(probabilities))

    # 根据划分的阈值threshold将数据集划分为左右两个子集
    def _split(self, X, y, feature, threshold):
        left_mask = X[:, feature] < threshold
        right_mask = X[:, feature] >= threshold
        return X[left_mask], y[left_mask], X[right_mask], y[right_mask]

    def _best_split(self, X, y):
        best_gain_ratio = float('-inf')
        best_feature = None
        best_threshold = None
        for feature in tqdm(range(X.shape[1])):
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

    # 递归构建决策树的函数
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
    # 对单个输入样本进行预测
    def _predict(self, x, node):
        # 如果传入的节点是叶子节点，直接返回该叶子节点的值。
        if node.is_leaf_node():
            return node.value
        # 如果输入样本在节点的分裂特征上的取值小于节点的阈值，则递归遍历左子树进行预测。
        if x[node.feature] < node.threshold:
            return self._predict(x, node.left)
        # 否则，递归遍历右子树进行预测。
        return self._predict(x, node.right)
    # 对数据集进行预测
    def predict(self, X):
        return np.array([self._predict(x, self.root) for x in X])
    # 后剪枝方法
    def _prune(self, node, X_val, y_val):
        # 如果传入的节点是叶子节点
        if node.is_leaf_node():
            # 直接返回该叶子节点的值
            return node.value

        # 根据给定的特征数据和传入的节点，预测左右子树的预测结果
        left_val = np.array([self._predict(x, node.left) for x in X_val])
        right_val = np.array([self._predict(x, node.right) for x in X_val])

        # 计算剪枝前的误差，使用错误分类的样本数量作为误差
        error_before = np.sum(left_val != y_val[node.left]) + np.sum(right_val != y_val[node.right])
        if not node.left.is_leaf_node():
            node.left = Node(value=np.argmax(np.bincount(y_val[node.left])))
        if not node.right.is_leaf_node():
            node.right = Node(value=np.argmax(np.bincount(y_val[node.right])))
        # 对左右子树分别进行剪枝
        left_val = np.array([self._predict(x, node.left) for x in X_val])
        right_val = np.array([self._predict(x, node.right) for x in X_val])
        # 并计算剪枝后的误差
        error_after = np.sum(left_val != y_val[node.left]) + np.sum(right_val != y_val[node.right])

        # 如果剪枝后的误差小于等于剪枝前的误差，则保留剪枝后的决策树
        if error_after <= error_before:
            return node.value
        # 否则，恢复原始的决策树结构，递归剪枝左右子树
        node.left = self._prune(node.left, X_val, y_val)
        node.right = self._prune(node.right, X_val, y_val)
        return node

    def fit(self, X, y, X_val=None, y_val=None):
        self.root = self._build_tree(X, y, 0)
        if X_val is not None and y_val is not None:
            self.root = self._prune(self.root, X_val, y_val)

if __name__ == "__main__":
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score
    from utils import load_cora
    # 划分训练集和测试集
    X_train, y_train, X_test, y_test = load_cora()
    # 使用手写实现的C45
    our_nb = C45DecisionTree(max_depth=3)
    our_nb.fit(X_train, y_train)
    our_predictions = our_nb.predict(X_test)

    # 使用sklearn的朴素贝叶斯分类器
    sklearn_nb = DecisionTreeClassifier(criterion='entropy', max_depth=3)
    sklearn_nb.fit(X_train, y_train)
    sklearn_predictions = sklearn_nb.predict(X_test)

    # 比较准确率
    our_accuracy = accuracy_score(y_test, our_predictions)
    sklearn_accuracy = accuracy_score(y_test, sklearn_predictions)

    print(f"Our C45 DecisionTree Classifier accuracy: {our_accuracy}")
    print(f"Sklearn C45 DecisionTree Classifier accuracy: {sklearn_accuracy}")

    # Our C45 DecisionTree Classifier accuracy: 0.317
    # Sklearn C45 DecisionTree Classifier accuracy: 0.359
