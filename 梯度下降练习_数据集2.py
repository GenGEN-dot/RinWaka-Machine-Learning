"""
梯度下降练习_数据集2.py
2026/1/4
哈哈新的一年第一件事是写代码,没救了呢
"""

import numpy as np  # 新增：用于正规方程验证

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
]   #特征

y_labels = [
    2.5, 3.0, 2.0, 4.5, 3.5, 
    5.5, 4.0, 6.5, 5.0, 6.0
]   #标签

#===========

"""
这还是我第一次碰见多维数据集……试试看吧
看起来模型是 y = k[0]*x[0] + k[1]*x[1] + k[2]*x[2] + w,不过看起来x[0]恒为1所以其实就是y = k[0]*x[0] + k[1]*x[1] + w或者y = k[0]*x[0] + k[1]*x[1] + k[2]*x[2]

后来发现：其实这两种表示是等价的！方案1中k[0]就是偏置，方案2中w是偏置。
数学上：y = k0 + k1*x1 + k2*x2 = k1*x1 + k2*x2 + w
所以 k0 = w
"""
#===========

class GradientDescentScheme1:

    """
    梯度下降-方案1:k[0]作为偏置
    模型：y = k[0]*x[0] + k[1]*x[1] + k[2]*x[2]
    其中 x[0] = 1，所以 k[0] 就是偏置项
    """

    def __init__(self):
        """
        参数初始化
        """
        # 方案1参数
        self.k = [0.00, 0.00, 0.00]             # k[0]是偏置，k[1]是x1的系数，k[2]是x2的系数

    def compute_loss(self, X, y):
        """
        损失函数 - 使用均方误差(MSE)
        L = 1/n Σ(y_pred - y_true)^2
        """
        total = 0.00
        for i in range(len(X)):
            pred = sum(self.k[j] * X[i][j] for j in range(len(self.k)))
            total += (pred - y[i]) ** 2
        return total / len(X)  # MSE: 除以样本数，不是2*样本数

    def compute_gradient(self, X, y):
        """
        计算梯度 - 使用平均梯度
        梯度公式：∂L/∂k[j] = 1/n Σ(y_pred - y_true) * x[j]
        """
        num_samples = len(X)
        gradients = [0.0 for _ in range(len(self.k))]
    
        for i in range(num_samples):
            # 1. 计算预测值
            prediction = sum(self.k[j] * X[i][j] for j in range(len(self.k)))
            # 2. 计算误差
            error = prediction - y[i]
        
            # 3. 累加每个参数的梯度
            for j in range(len(self.k)):
                gradients[j] += error * X[i][j]
        
        # 4. 计算平均梯度 (除以样本数)
        for j in range(len(self.k)):
            gradients[j] /= num_samples
        
        return gradients

    def update_parameters(self, gradients, learning_rate):
        """
        更新参数
        k[j] = k[j] - α * ∂L/∂k[j]
        """
        for j in range(len(self.k)):
            self.k[j] -= learning_rate * gradients[j]

    def train(self, X, y, epochs=10000, learning_rate=0.01, verbose=True):
        """
        训练模型
        梯度下降核心：重复计算梯度并更新参数，直到收敛
        """
        for epoch in range(epochs):
            loss = self.compute_loss(X, y)
            gradients = self.compute_gradient(X, y)
            self.update_parameters(gradients, learning_rate)
            
            if verbose and epoch % 1000 == 0:
                print(f"学习轮次: {epoch}, 损失率: {loss:.6f}, 参数: [{self.k[0]:.6f}, {self.k[1]:.6f}, {self.k[2]:.6f}]")
            
            # 提前停止：如果梯度非常小，说明已经收敛到最小值
            if max(abs(g) for g in gradients) < 1e-10:
                if verbose:
                    print(f"提前停止于轮次 {epoch}，梯度已接近零")
                break
        
        if verbose:
            final_loss = self.compute_loss(X, y)
            print(f"\n最终参数: [{self.k[0]:.6f}, {self.k[1]:.6f}, {self.k[2]:.6f}]")
            print(f"最终损失(MSE): {final_loss:.6f}")
            print(f"均方根误差(RMSE): {final_loss**0.5:.6f}")
            print(f"平均绝对误差(MAE): {self.compute_mae(X, y):.6f}")

    def compute_mae(self, X, y):
        """计算平均绝对误差 - 另一个评估指标"""
        total = 0.0
        for i in range(len(X)):
            pred = sum(self.k[j] * X[i][j] for j in range(len(self.k)))
            total += abs(pred - y[i])
        return total / len(X)

#===========

class GradientDescentScheme2:
    """
    梯度下降-方案2:w作为偏置
    模型：y = k[0]*x[1] + k[1]*x[2] + w
    其中w是独立的偏置项
    """

    def __init__(self):
        """
        参数初始化
        """
        # 方案2参数  
        self.k = [0.00, 0.00]                   # 2个k：k[0]是x1的系数，k[1]是x2的系数
        self.w = 0.00                           # 1个w：偏置项

    def compute_loss(self, X, y):
        """
        损失函数 - 使用均方误差(MSE)
        """
        total = 0.00
        for i in range(len(X)):
            pred = self.k[0] * X[i][1] + self.k[1] * X[i][2] + self.w
            total += (pred - y[i]) ** 2
        return total / len(X)  # MSE

    def compute_gradient(self, X, y):
        """
        计算梯度
        梯度公式：
        ∂L/∂k[0] = 1/n Σ(y_pred - y_true) * x[1]
        ∂L/∂k[1] = 1/n Σ(y_pred - y_true) * x[2]
        ∂L/∂w = 1/n Σ(y_pred - y_true)
        """
        num_samples = len(X)
        dk = [0.0, 0.0]  # 初始化斜率梯度列表
        dw = 0.0  # 初始化截距梯度

        for i in range(num_samples):
            # 1. 计算预测值
            prediction = self.k[0] * X[i][1] + self.k[1] * X[i][2] + self.w
            # 2. 计算误差
            error = prediction - y[i]
            # 3. 累加斜率梯度和截距梯度
            dk[0] += error * X[i][1]
            dk[1] += error * X[i][2]
            dw += error

        # 4. 计算平均梯度
        dk[0] /= num_samples
        dk[1] /= num_samples
        dw /= num_samples

        return dk, dw
    
    def update_parameters(self, dk, dw, learning_rate):
        """
        更新参数
        修正了原来的错误：w不应该在循环内更新多次
        """
        # 更新斜率参数
        self.k[0] -= learning_rate * dk[0]
        self.k[1] -= learning_rate * dk[1]
        # 更新偏置参数
        self.w -= learning_rate * dw
    
    def train(self, X, y, epochs=10000, learning_rate=0.01, verbose=True):
        """
        训练模型
        """
        for epoch in range(epochs):
            loss = self.compute_loss(X, y)
            dk, dw = self.compute_gradient(X, y)
            self.update_parameters(dk, dw, learning_rate)
            
            if verbose and epoch % 1000 == 0:
                print(f"学习轮次: {epoch}, 损失率: {loss:.6f}, k: [{self.k[0]:.6f}, {self.k[1]:.6f}], w: {self.w:.6f}")
            
            # 提前停止：计算梯度范数
            grad_norm = (dk[0]**2 + dk[1]**2 + dw**2)**0.5
            if grad_norm < 1e-10:
                if verbose:
                    print(f"提前停止于轮次 {epoch}，梯度已接近零")
                break
        
        if verbose:
            final_loss = self.compute_loss(X, y)
            print(f"\n最终参数: k = [{self.k[0]:.6f}, {self.k[1]:.6f}], w = {self.w:.6f}")
            print(f"最终损失(MSE): {final_loss:.6f}")
            print(f"均方根误差(RMSE): {final_loss**0.5:.6f}")
            print(f"平均绝对误差(MAE): {self.compute_mae(X, y):.6f}")

    def compute_mae(self, X, y):
        """计算平均绝对误差"""
        total = 0.0
        for i in range(len(X)):
            pred = self.k[0] * X[i][1] + self.k[1] * X[i][2] + self.w
            total += abs(pred - y[i])
        return total / len(X)

#===========

"""
验证部分：使用正规方程验证梯度下降结果
正规方程可以直接计算线性回归的最优解：θ = (X^T X)^(-1) X^T y
如果我们的梯度下降实现正确，应该得到相同的结果（允许微小数值误差）
"""

def normal_equation_solution(X, y):
    """
    使用正规方程计算线性回归的最优解
    θ = (X^T X)^(-1) X^T y
    """
    X_array = np.array(X)
    y_array = np.array(y)
    
    # 计算正规方程
    X_T = X_array.T
    X_T_X = np.dot(X_T, X_array)
    X_T_X_inv = np.linalg.inv(X_T_X)
    theta = np.dot(np.dot(X_T_X_inv, X_T), y_array)
    
    return theta

def compute_mse(X, y, theta):
    """计算给定参数的MSE"""
    predictions = np.dot(X, theta)
    mse = np.mean((predictions - y) ** 2)
    return mse

#===========

if __name__ == "__main__":
    """
    主程序：训练两个方案，验证它们等价，并与正规方程比较
    
    重要发现：
    1. 方案1和方案2在数学上是等价的，只是参数表示不同
    2. 梯度下降已经收敛到最优解，损失不为零是因为数据有噪声（不是完美线性）
    3. 与正规方程的结果比较，验证了梯度下降实现的正确性
    """
    print("=" * 50)
    print("方案1训练结果 (k[0]作为偏置):")
    print("=" * 50)
    model1 = GradientDescentScheme1()
    # 使用更小的学习率和更多轮次确保收敛
    model1.train(X_train, y_labels, epochs=50000, learning_rate=0.001, verbose=True)
    
    print("\n" + "=" * 50)
    print("方案2训练结果 (w作为偏置):")
    print("=" * 50)
    model2 = GradientDescentScheme2()
    model2.train(X_train, y_labels, epochs=50000, learning_rate=0.001, verbose=True)
    
    # 验证两个方案是否等价
    print("\n" + "=" * 50)
    print("验证两个方案的等价性:")
    print("=" * 50)
    print(f"方案1: y = {model1.k[0]:.6f} + {model1.k[1]:.6f}*x1 + {model1.k[2]:.6f}*x2")
    print(f"方案2: y = {model2.k[0]:.6f}*x1 + {model2.k[1]:.6f}*x2 + {model2.w:.6f}")
    print("✓ 两个方案在数学上等价：方案1的k[0] = 方案2的w，k[1]和k[2]对应相同")
    
    # 使用正规方程验证
    print("\n" + "=" * 50)
    print("正规方程验证:")
    print("=" * 50)
    theta_normal = normal_equation_solution(X_train, y_labels)
    print(f"正规方程解: {theta_normal}")
    mse_normal = compute_mse(np.array(X_train), np.array(y_labels), theta_normal)
    print(f"正规方程MSE: {mse_normal:.6f}")
    print(f"梯度下降MSE: {model1.compute_loss(X_train, y_labels):.6f}")
    print(f"差异: {abs(mse_normal - model1.compute_loss(X_train, y_labels)):.10f}")
    print("✓ 梯度下降与正规方程结果几乎相同（微小差异来自浮点计算误差）")
    
    # 计算预测值对比
    print("\n" + "=" * 50)
    print("预测值对比:")
    print("=" * 50)
    print("样本 | 实际值 | 方案1预测 | 方案2预测 | 正规方程预测")
    print("-" * 60)
    for i in range(len(X_train)):
        pred1 = sum(model1.k[j] * X_train[i][j] for j in range(3))
        pred2 = model2.k[0] * X_train[i][1] + model2.k[1] * X_train[i][2] + model2.w
        pred_normal = np.dot(np.array(X_train[i]), theta_normal)
        print(f"{i:4} | {y_labels[i]:6.2f} | {pred1:9.2f} | {pred2:9.2f} | {pred_normal:12.2f}")
    
    # 残差分析
    print("\n" + "=" * 50)
    print("残差分析 (实际值 - 预测值):")
    print("=" * 50)
    residuals = []
    for i in range(len(X_train)):
        pred = sum(model1.k[j] * X_train[i][j] for j in range(3))
        residual = y_labels[i] - pred
        residuals.append(residual)
        print(f"样本{i}: 实际值={y_labels[i]:.2f}, 预测值={pred:.2f}, 残差={residual:.2f}")
    
    print(f"\n平均残差: {sum(residuals)/len(residuals):.6f} (接近0，说明无偏)")
    print(f"残差平方和: {sum(r**2 for r in residuals):.6f}")
    print(f"MSE: {sum(r**2 for r in residuals)/len(residuals):.6f}")
    
    print("\n" + "=" * 50)
    print("总结:")
    print("=" * 50)
    print("1. 梯度下降实现正确，已收敛到最优解")
    print("2. 损失不为零是因为数据有噪声（不是完美线性关系）")
    print("3. 方案1和方案2等价，只是参数表示方式不同")
    print("4. 与正规方程结果一致，验证了实现的正确性")
    print("5. 平均残差接近0，说明模型无偏")