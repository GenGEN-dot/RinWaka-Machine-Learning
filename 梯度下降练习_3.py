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
        self.k = [0.00, 0.00, 0.00]  # 参数：k0, k1, k2
        self.predictions = []        # 存储当前参数的预测值
        self.X_current = None        # 当前使用的数据X（用于检查是否需要重新计算）
        self.k_current = None        # 当前用于计算预测值的参数（用于检查是否需要重新计算）

    def compute_predictions(self, X, force_recompute=False):
        """
        计算所有样本的预测值
        force_recompute: 强制重新计算（即使参数和数据没变）
        
        为什么要有这个函数？
        - 避免在compute_loss和compute_gradients中重复计算预测值
        - 提高效率，尤其是在数据量大时
        """
        # 检查是否需要重新计算：
        # 1. 预测值列表为空（第一次计算）
        # 2. 数据X变了
        # 3. 参数k变了
        # 4. 强制重新计算
        if (not self.predictions or X != self.X_current or self.k != self.k_current or force_recompute):
            
            self.predictions = []  # 清空旧的预测值
            for i in range(len(X)):
                pred = sum(self.k[j] * X[i][j] for j in range(len(self.k)))
                self.predictions.append(pred)
            
            # 记录当前状态，方便下次检查
            self.X_current = X[:] if isinstance(X, list) else X.copy()
            self.k_current = self.k.copy()
        
        return self.predictions
    
    def compute_loss(self, X, y):
        """
        Step1: 计算损失
        """

        m = len(X)
        total_error = 0.00

        for i in range (m):
            prediction = self.predictions
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
            prediction = self.predictions
            for j in range(len(self.k)):
                gradients[j] += error * X[i][j]

        for j in range(len(self.k)):
            gradients[j] /= m
        
        return gradients
    
    def updata_parameters (self, gradients, learning_rate = 0.01):
        """
        参数更新
        """
        for j in range(len(self.k)):
            self.k[j] -= learning_rate * gradients[j]
        return self.k

    def train(self, X, y, learning_rate, epochs):
        """
        完整训练流程
        """
        losses = []

        for epoch in range(epochs):
            predictions = self.predictions
            loss = self.compute_loss
            losses.append(loss)
            for j in range(len(self.k))
                gradients = self.compute_gradients


def main():
    pass


if __name__ == "__main__":
    main()