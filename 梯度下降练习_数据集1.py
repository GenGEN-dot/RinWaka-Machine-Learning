"""
梯度下降练习_数据集1.py
2025/12/29
哇年底了还在写代码，没救了呢(无感情)
"""

# ============

# 导入所需的库……好吧我其实根本没有用到numpy或者sklearn之类的，我写了几次梯度下降一个库都没用到(目移)

#============

X_train = [2.5, 1.5, 3.0, 2.0, 4.0, 3.5, 1.0, 5.0, 4.5, 6.0]    #特征

y_labels = [5.0, 3.5, 6.0, 4.5, 8.0, 7.0, 2.0, 10.0, 9.0, 12.0] #标签

# ============

k = 0.00    #斜率初始化
b = 0.00    #截距初始化

def normalize_features(X):
    pass

def compute_cost(X, y, k, b):
    """
    损失函数
    """
    num_samples = len(X)
    total_cost = 0.0
    for i in range(num_samples):
        prediction = k * X[i] + b
        error = prediction - y[i]
        total_cost += error ** 2
    return total_cost / (2 * num_samples)

def compute_gradient(X, y, k, b):
    """
    计算梯度
    """
    num_samples = len(X)
    dk = 0.00
    dw = 0.00
    for i in range(num_samples):
        prediction = k * X[i] + b
        error = prediction - y[i]
        dk += error * X[i]
        dw += error
        dk /= num_samples
        dw /= num_samples
    return dk, dw

def update_parameters(k, b, dk, db, learning_rate):
    """
    更新参数
    """

    
def main():
    pass


if __name__ == "__main__":
    main()