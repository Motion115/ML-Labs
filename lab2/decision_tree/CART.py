# 手写CART
import numpy as np
from tqdm import tqdm

class CartDecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        # 最大深度
        self.max_depth = max_depth
        # 小拆分样本数
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

    # 递归构建决策树
    def _build_tree(self, X, y, depth):
        # 如果达到最大深度或样本数少于最小分裂样本数，返回叶节点，并以最常见的标签作为该叶节点的预测值
        if depth == self.max_depth or len(X) < self.min_samples_split:
            return {'is_leaf': True, 'label': self._most_common_label(y)}
        # 如果所有样本都属于同一类别，返回叶节点，并以该类别作为该叶节点的预测值
        if self._all_same_class(y):
            return {'is_leaf': True, 'label': y[0]}
        # 选择最佳拆分特征和拆分点，根据拆分点将数据集分成左右两个子集
        split_feature, split_value = self._best_split(X, y)
        left_indices = X[:, split_feature] <= split_value
        right_indices = X[:, split_feature] > split_value
        # 递归构建左子树和右子树
        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth=depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth=depth + 1)
        # 返回非叶节点，并记录拆分特征、拆分点以及左右子树
        return {'is_leaf': False, 'split_feature': split_feature, 'split_value': split_value,
                'left': left_subtree, 'right': right_subtree}

    # 找到最佳拆分特征和拆分值，以最大化信息增益
    def _best_split(self, X, y):
        # 初始化最佳拆分特征、拆分值和信息增益
        best_split_feature = None
        best_split_value = None
        best_info_gain = -float('inf')
        # 遍历所有特征
        for feature in tqdm(range(X.shape[1])):
            # 对当前特征的取值进行排序
            values = sorted(list(set(X[:, feature])))
            # 遍历当前特征的所有取值
            for i in tqdm(range(1, len(values))):
                # 计算当前取值和上一个取值的平均值作为拆分点
                split_value = (values[i - 1] + values[i]) / 2
                # 根据拆分点将数据集分成左右两个子集
                left_indices = X[:, feature] <= split_value
                right_indices = X[:, feature] > split_value
                # 如果有一个子集为空，则跳过这个拆分点
                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue
                # 计算通过当前拆分点得到的信息增益
                info_gain = self._information_gain(y, y[left_indices], y[right_indices])
                # 如果当前信息增益大于最大信息增益，则更新最佳拆分特征、拆分值和信息增益
                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_split_feature = feature
                    best_split_value = split_value
        # 返回最佳拆分特征和拆分值
        return best_split_feature, best_split_value

    # 计算信息增益
    def _information_gain(self, y, y_left, y_right):

        # 左子树样本数占总样本数的比例
        p = len(y_left) / len(y)
        # 总体数据集的熵
        entropy_parent = self._entropy(y)
        # 分裂后两个子节点的加权平均熵
        entropy_children = p * self._entropy(y_left) + (1 - p) * self._entropy(y_right)
        # 返回信息增益 = 总体数据集的熵 - 分裂后两个子节点的加权平均熵
        return entropy_parent - entropy_children

    # 计算输入标签y的熵
    def _entropy(self, y):
        # 统计每个类别出现的次数
        _, counts = np.unique(y, return_counts=True)
        # 计算每个类别的出现概率
        probabilities = counts / len(y)
        # 根据公示计算熵并返回
        return -np.sum(probabilities * np.log2(probabilities))

    # 计算标签y中最常见的类别
    def _most_common_label(self,y):
        unique_labels, counts = np.unique(y, return_counts=True)
        index = np.argmax(counts)
        return unique_labels[index]

    # 判断标签y中的所有样本是否属于同一类别
    def _all_same_class(self, y):
        _, counts = np.unique(y, return_counts=True)
        return len(counts) == 1

if __name__ == "__main__":
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score
    from utils import load_cora
    # 划分训练集和测试集
    X_train, y_train, X_test, y_test = load_cora()
    # 使用手写实现的CART
    our_nb = CartDecisionTree(max_depth=3)
    our_nb.fit(X_train, y_train)
    our_predictions = our_nb.predict(X_test)

    # 使用sklearn的DecisionTreeClassifier-CART
    sklearn_nb = DecisionTreeClassifier(criterion='gini', max_depth=3)
    sklearn_nb.fit(X_train, y_train)
    sklearn_predictions = sklearn_nb.predict(X_test)

    # 比较准确率
    our_accuracy = accuracy_score(y_test, our_predictions)
    sklearn_accuracy = accuracy_score(y_test, sklearn_predictions)

    print(f"Our CART DecisionTree Classifier accuracy: {our_accuracy}")
    print(f"Sklearn CART DecisionTree Classifier accuracy: {sklearn_accuracy}")

    # Our CART DecisionTree Classifier accuracy: 0.359
    # sklearn CART DecisionTree Classifier accuracy: 0.336