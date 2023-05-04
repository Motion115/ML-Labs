import numpy as np
from tqdm import tqdm

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def predict(self, X):
        y_pred = []
        for x in X:
            node = self.tree
            while not node['is_leaf']:
                if x[node['split_feature']] <= node['split_value']:
                    node = node['left']
                else:
                    node = node['right']
            y_pred.append(node['label'].item())

        return y_pred

    def _build_tree(self, X, y, depth):
        if depth == self.max_depth or len(X) < self.min_samples_split:
            return {'is_leaf': True, 'label': self._most_common_label(y)}
        if self._all_same_class(y):
            return {'is_leaf': True, 'label': y[0]}
        split_feature, split_value = self._find_best_split(X, y)
        left_indices = X[:, split_feature] <= split_value
        right_indices = X[:, split_feature] > split_value
        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth=depth+1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth=depth+1)
        return {'is_leaf': False, 'split_feature': split_feature, 'split_value': split_value,
                'left': left_subtree, 'right': right_subtree}

    def _find_best_split(self, X, y):
        best_split_feature = None
        best_split_value = None
        best_info_gain = -float('inf')
        for feature in tqdm(range(X.shape[1])):
            values = sorted(list(set(X[:, feature])))
            for i in tqdm(range(1, len(values))):
                split_value = (values[i-1] + values[i]) / 2
                left_indices = X[:, feature] <= split_value
                right_indices = X[:, feature] > split_value
                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue
                info_gain = self._information_gain(y, y[left_indices], y[right_indices])
                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_split_feature = feature
                    best_split_value = split_value
        return best_split_feature, best_split_value

    def _information_gain(self, y, y_left, y_right):
        p = len(y_left) / len(y)
        entropy_parent = self._entropy(y)
        entropy_children = p * self._entropy(y_left) + (1 - p) * self._entropy(y_right)
        return entropy_parent - entropy_children

    def _entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities))

    def _most_common_label(self,y):
        unique_labels, counts = np.unique(y, return_counts=True)
        index = np.argmax(counts)
        return unique_labels[index]

    def _all_same_class(self, y):
        _, counts = np.unique(y, return_counts=True)
        return len(counts) == 1

if __name__ == "__main__":
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score
    from utils import load_cora
    # 划分训练集和测试集
    X_train, y_train, X_test, y_test = load_cora()
    # 使用我们的C45
    our_nb = DecisionTree(max_depth=3)
    our_nb.fit(X_train, y_train)
    our_predictions = our_nb.predict(X_test)

    # 使用sklearn的朴素贝叶斯分类器
    sklearn_nb = DecisionTreeClassifier(criterion='gini', max_depth=3)
    sklearn_nb.fit(X_train, y_train)
    sklearn_predictions = sklearn_nb.predict(X_test)

    # 比较准确率
    our_accuracy = accuracy_score(y_test, our_predictions)
    sklearn_accuracy = accuracy_score(y_test, sklearn_predictions)

    print(f"Our CART DecisionTree Classifier accuracy: {our_accuracy}")
    print(f"Sklearn CART DecisionTree Classifier accuracy: {sklearn_accuracy}")

    # Our CART DecisionTree Classifier accuracy: 0.359
    # sklearn CART DecisionTree Classifier accuracy: 0.359