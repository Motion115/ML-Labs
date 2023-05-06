# 手写ID3
import numpy as np
from tqdm import tqdm
# to solve import issue
import sys, os
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PROJ_DIR))

class ID3DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=None, pruning=False):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.pruning = pruning
        self.tree = None
        self.validation_data = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def predict(self, X):
        y_pred = []
        for x in X:
            node = self.tree
            while not node['is_leaf']:
                if x[node['feature']] <= node['split_value']:
                    node = node['left']
                else:
                    node = node['right']
            y_pred.append(node['label'])
        return np.array(y_pred)

    def _build_tree(self, X, y, depth=0):
        # 如果所有样本都属于同一类别，返回叶节点
        if len(np.unique(y)) == 1:
            return {'is_leaf': True, 'label': y[0]}
        # 如果样本数量小于最小分割样本数，返回叶节点，将该节点标记为样本数量最多的类别
        if len(X) < self.min_samples_split:
            label = np.bincount(y).argmax()
            return {'is_leaf': True, 'label': label}
        # 如果所有特征都已经被用于分割，返回叶节点，将该节点标记为样本数量最多的类别
        if len(np.unique(X, axis=0)) == 1:
            label = np.bincount(y).argmax()
            return {'is_leaf': True, 'label': label}
        # 如果达到了最大深度，返回叶节点，将该节点标记为样本数量最多的类别
        if self.max_depth is not None and depth == self.max_depth:
            label = np.bincount(y).argmax()
            return {'is_leaf': True, 'label': label}

        # 选择最佳分割特征和分割点
        best_feature, best_value = self._best_split(X, y)

        # 如果最佳分割特征无法分割数据，则返回叶节点，将该节点标记为样本数量最多的类别
        if best_feature is None:
            label = np.bincount(y).argmax()
            return {'is_leaf': True, 'label': label}
        # 分割数据
        left_indices = X[:, best_feature] <= best_value
        right_indices = X[:, best_feature] > best_value
        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth=depth+1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth=depth+1)
        # 构建决策树节点
        node = {'is_leaf': False, 'feature': best_feature, 'split_value': best_value,
                'left': left_subtree, 'right': right_subtree}
        # 预剪枝
        if self.pruning:
            # 计算未剪枝决策树在验证集上的准确率
            X_val, y_val = self.validation_data
            y_pred = self.predict(X_val)
            acc_unpruned = np.mean(y_pred == y_val)
            # 计算剪枝后的决策树在验证集上的准确率
            node['is_leaf'] = True
            node['label'] = np.bincount(y).argmax()
            y_pred = self.predict(X_val)
            acc_pruned = np.mean(y_pred == y_val)
            # 如果剪枝后的决策树准确率更高，则剪枝
            if acc_pruned >= acc_unpruned:
                return node
        return node

    def _best_split(self, X, y):
        # 计算信息增益，选择最佳分割特征和分割点
        best_feature = None
        best_value = None
        best_info_gain = -float('inf')

        for feature in tqdm(range(X.shape[1])):
            values = np.unique(X[:, feature])
            for i in range(1, len(values)):
                split_value = (values[i-1] + values[i]) / 2
                info_gain = self._information_gain(y, X[:, feature], split_value)

                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_feature = feature
                    best_value = split_value

        return best_feature, best_value

    def _information_gain(self, y, feature, split_value):
        # 计算信息增益
        parent_entropy = self._entropy(y)

        left_indices = feature <= split_value
        right_indices = feature > split_value

        if len(left_indices) == 0 or len(right_indices) == 0:
            return 0

        left_entropy = self._entropy(y[left_indices])
        right_entropy = self._entropy(y[right_indices])

        children_entropy = (len(y[left_indices])/len(y)) * left_entropy + (len(y[right_indices])/len(y)) * right_entropy

        return parent_entropy - children_entropy

    def _entropy(self, y):
        # 计算熵
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities))

    def set_validation_data(self, X_val, y_val):
        self.validation_data = X_val, y_val

    def post_pruning(self):
        # 后剪枝
        self._post_pruning(self.tree)

    def _post_pruning(self, node):
        if not node['is_leaf']:
            self._post_pruning(node['left'])
            self._post_pruning(node['right'])

            # 判断是否可以剪枝
            if node['left']['is_leaf'] and node['right']['is_leaf']:
                left_label = node['left']['label']
                right_label = node['right']['label']

                # 计算未剪枝决策树在验证集上的准确率
                X_val, y_val = self.validation_data
                y_pred = self.predict(X_val)
                acc_unpruned = np.mean(y_pred == y_val)

                # 计算剪枝后的决策树在验证集上的准确率
                node['left'] = {'is_leaf': True, 'label': np.bincount(node['left']['y']).argmax()}
                node['right'] = {'is_leaf': True, 'label': np.bincount(node['right']['y']).argmax()}
                y_pred = self.predict(X_val)
                acc_pruned = np.mean(y_pred == y_val)

                # 如果剪枝后的决策树准确率更高，则剪枝
                if acc_pruned >= acc_unpruned:
                    node['is_leaf'] = True
                    node['label'] = np.bincount(node['y']).argmax()
                else:
                    node['left'] = self._build_tree(node['left']['X'], node['left']['y'])
                    node['right'] = self._build_tree(node['right']['X'], node['right']['y'])
if __name__ == "__main__":
    from utils import load_cora
    from sklearn.metrics import accuracy_score
    from sklearn.tree import DecisionTreeClassifier
    X_train, y_train, X_test, y_test = load_cora()

    # 使用手写实现的ID3算法
    id3_tree = ID3DecisionTree(max_depth=3)
    id3_tree.fit(X_train, y_train)
    id3_predictions = id3_tree.predict(X_test)

    # 使用sklearn的DecisionTreeClassifier-ID3
    sklearn_tree = DecisionTreeClassifier(criterion='entropy', max_depth=3)
    sklearn_tree.fit(X_train, y_train)
    sklearn_predictions = sklearn_tree.predict(X_test)

    # 比较准确率
    id3_accuracy = accuracy_score(y_test, id3_predictions)
    sklearn_accuracy = accuracy_score(y_test, sklearn_predictions)

    print(f"Our ID3 DecisionTree Classifier accuracy: {id3_accuracy}")
    print(f"sklearn ID3 DecisionTree Classifier accuracy: {sklearn_accuracy}")

    # Our ID3 DecisionTree Classifier accuracy: 0.359
    # sklearn ID3 DecisionTree Classifier accuracy: 0.359