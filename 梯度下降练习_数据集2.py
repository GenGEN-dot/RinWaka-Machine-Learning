"""
梯度下降练习_数据集2.py
2026/1/4
哈哈新的一年第一件事是写代码,没救了呢
"""

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

#这还是我第一次碰见多维数据集……试试看吧
#看起来模型是 y = k[0]*x[0] + k[1]*x[1] + k[2]*x[2] + w，不过看起来x[0]恒为1所以其实就是y = k[0]*x[0] + k[1]*x[1] + w或者y = k[0]*x[0] + k[1]*x[1] + k[2]*x[2]

#===========

# 方案1参数
k1 = [0.00, 0.00, 0.00]             # k[0]是偏置
w1 = None                           # 不使用

# 方案2参数  
k2 = [0.00, 0.00]                   # 2个k
w2 = 0.00                           # 1个w


def compute_loss_scheme1(X, y):
    """
    损失函数
    方案1:k[0]作为偏置
    """
    total = 0.00                    #损失初始化
    for i in range(len(X)):
        pred = sum(                 #计算预测值
            k1[j] * X[i][j] 
                for j in range(len(k1)))
        total += (pred - y[i]) ** 2 #累计平方误差
    return total / (2 * len(X))

def compute_loss_scheme2(X, y):
    """
    损失函数
    方案2:w作为偏置
    """
    total = 0.00
    for i in range(len(X)):
        pred = k2[0] * X[i][1] + k2[1] * X[i][2] + w2
        total += (pred - y[i]) ** 2
    return total / (2 * len(X))

#===========

def compute_gradient_scheme1(X, y):

    """
    计算梯度
    方案1:k[0]作为偏置
    """
    num_samples = len(X)

    gradients = [0.0 for _ in range(len(k1))]  # 初始化梯度列表，等价于初始化dk，dw
    
    for i in range(num_samples):
        # 1. 计算预测值
        prediction = sum(k1[j] * X[i][j] for j in range(len(k1)))
        
        # 2. 计算误差
        error = prediction - y[i]
        
        # 3. 累加每个参数的梯度
        for j in range(len(k1)):
            gradients[j] += error * X[i][j]  # 第j个参数的梯度
        
    # 4. 计算平均梯度
    for j in range(len(k1)):
        gradients[j] /= num_samples
        
    return gradients

def compute_gradient_scheme2(X, y):
    """
    计算梯度
    方案2:w作为偏置
    """
    num_samples = len(X)

    dk = [0.0 for _ in range(len(k2))]  # 初始化斜率梯度列表
    dw = 0.0                            # 初始化截距梯度

    for i in range(num_samples):
        # 1. 计算预测值
        prediction = k2[0] * X[i][1] + k2[1] * X[i][2] + w2
        
        # 2. 计算误差
        error = prediction - y[i]
        
        # 3. 累加斜率梯度和截距梯度
        dk[0] += error * X[i][1]
        dk[1] += error * X[i][2]
        dw += error
        
    # 4. 计算平均梯度
    for j in range(len(k2)):
        dk[j] /= num_samples
    dw /= num_samples

    return dk, dw   

def update_parameters_scheme1(k, gradients, learning_rate):
    """
    更新参数
    方案1:k[0]作为偏置
    """
    for j in range(len(k)):
        k[j] -= learning_rate * gradients[j]
    return k