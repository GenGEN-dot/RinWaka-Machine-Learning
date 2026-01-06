"""
梯度下降练习
哈哈还在写梯度下降......试着把copilot关掉自己写吧
"""

# 完美线性关系：y = 2 + 3*x1 + 4*x2

X_train = [
    [1, 1, 2],
    [1, 2, 3],
    [1, 3, 1],
    [1, 4, 4],
    [1, 5, 2],
    [1, 6, 5],
    [1, 7, 3],
    [1, 8, 6],
    [1, 9, 4],
    [1, 10, 5]
]

y_label = [
    2 + 3*1 + 4*2,    # 13
    2 + 3*2 + 4*3,    # 20
    2 + 3*3 + 4*1,    # 15
    2 + 3*4 + 4*4,    # 30
    2 + 3*5 + 4*2,    # 25
    2 + 3*6 + 4*5,    # 40
    2 + 3*7 + 4*3,    # 35
    2 + 3*8 + 4*6,    # 50
    2 + 3*9 + 4*4,    # 45
    2 + 3*10 + 4*5    # 52
]

# 期望结果：应该能完美收敛到 k=[2, 3, 4]，MSE=0

import numpy

#============

class Gradient_Descent:
    """
    梯度下降
    """
    def __init__(self):
        self.k = [0.00, 0.00, 0.00]  # 初始化参数

    def compute_loss(self, X, y):
        """
        Step1: 计算损失
        """

        m = len(X)
        total_error = 0.00
        for i in range (m):
            prediction = sum( self.k[j] * X[i][j] for j in range(len(self.k)))
            error = prediction - y[i]
            total_error += error ** 2
            loss = total_error / (2 * m)
        return loss

    def compute_gradients(self, X, y):
        """
        Step2: 梯度计算
        """
        m = len(X)
        gradients = [0.0 for _ in range(len(self.k))]

        for i in range(m):
            prediction = sum( self.k[j] * X[i][j] for j in range(len(self.k)))
            error = prediction - y[i]
            for j in range(len(self.k)):
                gradients[j] += error * X[i][j]
