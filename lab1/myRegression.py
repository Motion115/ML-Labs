import numpy as np
import matplotlib.pyplot as plt

class myRegression:
    def __init__(self):
        self.w = None   # 截距（即w）
        self.b = None        # 系数（即b）
        self.theta = None       # 将截距与系数合并为θ
        self.bgd_params = {}    # 梯度下降的参数
 
    def reset_params(self, theta, **bgd_params):
        """设置系数和参数"""
        theta = np.reshape(theta, (-1,))  # 将系数转为向量格式
        self.theta = theta
        self.w = self.theta[0]
        self.b = self.theta[1:]
        self.bgd_params = bgd_params
        return self
 
    def fit(self, train_x, train_y, method=2):
        """快速进入不同的训练方法"""
        if method in [1, 'normal_equation']:
            self.fitNormalEquation(train_x, train_y)
        elif method in [2, 'gradient_descent']:
            self.fitGradientDescent(train_x, train_y)
        elif method in [3, 'normal_equation_re']:
            self.fitNormalEquation(train_x, train_y, regularized=True)
        elif method in [4, 'gradient_descent_reg']:
            self.fitGradientDescent(train_x, train_y, regularized=True)
 
    def fitNormalEquation(self, train_x, train_y,
               regularized=False, lamda=1.5):
        """使用正规方程法训练模型（可选择是否正则化）"""
        # 若不选择正则项，则将其系数设为0
        lamda = 0 if not regularized else lamda
        # 在原有x上加一行1，用于与截距相乘，形成X
        X = np.hstack([np.ones((len(train_x), 1)), train_x])
 
        reg = lamda*np.eye(X.shape[1])
        reg[0, 0] = 0
        theta = np.linalg.pinv(X.T.dot(X) + reg)
        theta = theta.dot(X.T).dot(train_y)
        self.reset_params(theta)
 
    def fitGradientDescent(self, train_x, train_y,
                alpha=0.1, iters=20000, regularized=False, lamda=1.5,
                alpha_v_step=5000, alpha_v_rate=0.95, loss_show=False,
                theta=None):
        """使用梯度下降法训练模型（可选择是否正则化）"""
        # 初始化
        X = np.hstack([np.ones((len(train_x), 1)), train_x])
        num_data, len_theta = X.shape[0], X.shape[1]
        self.theta = np.ones((len_theta, 1)) if theta is None else theta
        lamda = 0 if not regularized else lamda     # 若不选择正则项，则将其系数设为0
        losses = []  # 记录迭代时的损失值变化
 
        # 梯度下降
        for i in range(iters):
            # 对MSE求导
            res = np.reshape(np.dot(X, self.theta), (-1,))
            error = res - train_y
            update = [np.reshape(error * X[:, j], (-1, 1)) for j in range(len_theta)]
            update = np.hstack(update)
            update = np.reshape(np.mean(update, axis=0), (-1, 1))
            # 更新学习率（每隔一定的迭代次数就按比缩小学习率）
            if i > 0 and i % alpha_v_step == 0:
                alpha = alpha * alpha_v_rate
            # 更新参数（若含正则项，则会在梯度下降前适当缩小原系数）
            self.theta = self.theta * (1-alpha*(lamda/num_data)) - alpha*update
            losses.append(self.loss(train_x, train_y))
 
        # 绘图展示迭代过程中损失值的变化
        self.reset_params(self.theta, alpha=alpha, iters=iters, lamda=lamda)
        if loss_show:
            plt.plot(range(len(losses)), losses)
            plt.title("MSE-Loss of BGD")  # 图形标题
            plt.xlabel("iters")  # x轴名称
            plt.show()
        return losses
 
    def predict(self, pred_x):
        pred_x = np.hstack([np.ones((len(pred_x), 1)), pred_x])
        pred_y = pred_x.dot(self.theta)
        pred_y = np.reshape(pred_y, (-1,))  #保证输出结果是向量
        return pred_y
 
    def loss(self, x, y):
        """MSE损失函数"""
        pred_y = self.predict(x)
        mse = np.sum(np.power((pred_y - y), 2)) / y.shape[0]
        return round(mse, 4)
 
    def score(self, x, y):
        """使用R2_score进行模型评价"""
        mse = self.loss(x, y)
        var = np.var(y)
        return round(1-mse/var, 4)