# 手写贝叶斯分类算法，并给出和sklearn中库函数中贝叶斯算法的比较
import numpy as np

class NaiveBayesClassifier:
    def __init__(self):
        # 训练数据集中所有可能的类别
        self.classes = None
        # 每个类别的先验概率
        self.priors = None
        # 每个类别的似然概率
        self.likelihoods = None

    def fit(self, X, y):
        # 计算y中每个类别的数量，并将其存储在counts变量中
        self.classes, counts = np.unique(y, return_counts=True)
        # 计算每个类别的先验概率，并将其存储在priors变量中
        self.priors = counts / len(y)
        # 计算每个类别的特征的均值和方差，并将它们存储在likelihoods变量中
        self.likelihoods = {}
        for c in self.classes:
            X_c = X[y == c]
            self.likelihoods[c] = {
                'mean': np.mean(X_c, axis=0),
                'var': np.var(X_c, axis=0)
            }

    def predict(self, X):
        predictions = []
        for x in X:
            # 创建一个空列表posteriors，用于存储后验概率
            posteriors = []
            for c in self.classes:
                # 计算先验概率
                prior = np.log(self.priors[np.where(self.classes == c)[0][0]])
                # 计算似然概率
                likelihood = -0.5 * np.sum(np.log(2 * np.pi * self.likelihoods[c]['var']))
                likelihood -= 0.5 * np.sum(((x - self.likelihoods[c]['mean']) ** 2) / self.likelihoods[c]['var'])
                # 计算后验概率并将其添加到posteriors列表中
                posteriors.append(prior + likelihood)
            # 将具有最大后验概率的类别添加到predictions列表中
            predictions.append(self.classes[np.argmax(posteriors)])
        return predictions

if __name__ == "__main__":
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score
    from utils import load_cora
    # 划分训练集和测试集
    X_train, y_train, X_test, y_test = load_cora()
    # 使用我们的朴素贝叶斯分类器
    our_nb = NaiveBayesClassifier()
    our_nb.fit(X_train, y_train)
    our_predictions = our_nb.predict(X_test)

    # 使用sklearn的朴素贝叶斯分类器
    sklearn_nb = GaussianNB()
    sklearn_nb.fit(X_train, y_train)
    sklearn_predictions = sklearn_nb.predict(X_test)

    # 比较准确率
    our_accuracy = accuracy_score(y_test, our_predictions)
    sklearn_accuracy = accuracy_score(y_test, sklearn_predictions)

    print(f"Our Naive Bayes Classifier accuracy: {our_accuracy}")
    print(f"Sklearn Naive Bayes Classifier accuracy: {sklearn_accuracy}")

    # Our Naive Bayes Classifier accuracy: 0.556
    # Sklearn Naive Bayes Classifier accuracy: 0.556


