"""
梯度下降练习_数据集1.py
2025/12/29
哇年底了还在写代码，没救了呢(无感情)
"""

# ============

# 导入所需的库……好吧我其实根本没有用到numpy或者sklearn之类的，我写了几次梯度下降一个库都没用到(目移)

#============

X_train = [2.5, 1.5, 3.0, 2.0, 4.0, 3.5, 1.0, 5.0, 4.5, 6.0]    #特征

y_labels = [5.0, 3.0, 6.0, 4.0, 8.0, 7.0, 2.0, 10.0, 9.0, 12.0] #标签
y_labels = [5.0, 3.0, 6.0, 4.0, 8.0, 7.0, 2.0, 10.0, 9.0, 12.0] #标签

# ============

k = 0.00    #斜率初始化
w = 0.00    #截距初始化
w = 0.00    #截距初始化

def normalize_features(X):
    pass

def compute_cost(X, y, k, w):
def compute_cost(X, y, k, w):
    """
    损失函数
    """
    num_samples = len(X)                    #样本数量
    total_cost = 0.0                        #损失初始化
    for i in range(num_samples):            #计算损失
        prediction = k * X[i] + w           #预测值
        error = prediction - y[i]           #误差
        total_cost += error ** 2            #平方误差累加
    return total_cost / (2 * num_samples)   #返回平均损失
    num_samples = len(X)                    #样本数量
    total_cost = 0.0                        #损失初始化
    for i in range(num_samples):            #计算损失
        prediction = k * X[i] + w           #预测值
        error = prediction - y[i]           #误差
        total_cost += error ** 2            #平方误差累加
    return total_cost / (2 * num_samples)   #返回平均损失

def compute_gradient(X, y, k, w):
def compute_gradient(X, y, k, w):
    """
    计算梯度
    """
    num_samples = len(X)                    #样本数量
    dk = 0.00                               #斜率梯度初始化
    num_samples = len(X)                    #样本数量
    dk = 0.00                               #斜率梯度初始化
    dw = 0.00
    for i in range(num_samples):            #更新流程
        prediction = k * X[i] + w           #预测值
        error = prediction - y[i]           #误差
        dk += error * X[i]                  #斜率梯度
        dw += error                         #截距梯度
    dk /= num_samples                       #取平均
    dw /= num_samples
    return dk, dw

def update_parameters(k, w, dk, dw, learning_rate):
def update_parameters(k, w, dk, dw, learning_rate):
    """
    更新参数
    """
    k -= learning_rate * dk                 #更新斜率
    w -= learning_rate * dw                 #更新截距
    return k, w

def train(X, y, k, w, learning_rate=0.01, epochs=1000, 
          loss_threshold=1e-10, patience=20):
    """
    训练模型
    """
    prev_cost = float('inf')
    no_improve_count = 0
    
    for epoch in range(epochs):
        k_gradient, w_gradient = compute_gradient(X, y, k, w)
        k, w = update_parameters(k, w, k_gradient, w_gradient, learning_rate)
        
        cost = compute_cost(X, y, k, w)
        
        # 检查损失是否足够小
        if cost < loss_threshold:
            print(f"🎉 训练完成于第{epoch}次迭代！损失={cost}")
            print(f"最终参数: k={k}, w={w}")
            return k, w
        
        # 检查损失是否还在下降
        cost_change = prev_cost - cost
        if cost_change < 1e-8:  # 下降很小
            no_improve_count += 1
        else:
            no_improve_count = 0
        
        if no_improve_count >= patience:
            print(f"⚠️ 训练提前停止于第{epoch}次迭代，损失不再显著下降")
            print(f"最终损失: {cost}, k={k}, w={w}")
            return k, w
        
        # 每100次输出一次
        if epoch % 100 == 0:
            print(f"迭代 {epoch}: 损失={cost:.10f}, k={k:.6f}, w={w:.6f}")
        
        prev_cost = cost
    
    print(f"训练完成（达到最大迭代次数{epochs}）")
    print(f"最终参数: k={k}, w={w}")
    return k, w

    k -= learning_rate * dk                 #更新斜率
    w -= learning_rate * dw                 #更新截距
    return k, w

def train(X, y, k, w, learning_rate=0.01, epochs=1000, 
          loss_threshold=1e-10, patience=20):
    """
    训练模型
    """
    prev_cost = float('inf')
    no_improve_count = 0
    
    for epoch in range(epochs):
        k_gradient, w_gradient = compute_gradient(X, y, k, w)
        k, w = update_parameters(k, w, k_gradient, w_gradient, learning_rate)
        
        cost = compute_cost(X, y, k, w)
        
        # 检查损失是否足够小
        if cost < loss_threshold:
            print(f"🎉 训练完成于第{epoch}次迭代！损失={cost}")
            print(f"最终参数: k={k}, w={w}")
            return k, w
        
        # 检查损失是否还在下降
        cost_change = prev_cost - cost
        if cost_change < 1e-8:  # 下降很小
            no_improve_count += 1
        else:
            no_improve_count = 0
        
        if no_improve_count >= patience:
            print(f"⚠️ 训练提前停止于第{epoch}次迭代，损失不再显著下降")
            print(f"最终损失: {cost}, k={k}, w={w}")
            return k, w
        
        # 每100次输出一次
        if epoch % 100 == 0:
            print(f"迭代 {epoch}: 损失={cost:.10f}, k={k:.6f}, w={w:.6f}")
        
        prev_cost = cost
    
    print(f"训练完成（达到最大迭代次数{epochs}）")
    print(f"最终参数: k={k}, w={w}")
    return k, w

def main():
    print("当前特征:", X_train)
    print("初始参数: k=", k, " w=", w)
    
    # 训练模型
    final_k, final_w = train(
        X_train, y_labels, k, w, 
        learning_rate=0.075, 
        epochs=10000,
        loss_threshold=1e-10,
        patience=20
    )
    
    # 交互式预测
    while True:
        user_input = input("\n请输入一个特征值进行预测(输入'exit'退出): ")
        if user_input.lower() == 'exit':
            break
        try:
            feature_value = float(user_input)
            prediction = final_k * feature_value + final_w
            print(f"预测结果: {prediction:.2f}")
        except ValueError:
            print("请输入有效的数字或'exit'退出。")
    
    print("程序结束")
    print("当前特征:", X_train)
    print("初始参数: k=", k, " w=", w)
    
    # 训练模型
    final_k, final_w = train(
        X_train, y_labels, k, w, 
        learning_rate=0.075, 
        epochs=10000,
        loss_threshold=1e-10,
        patience=20
    )
    
    # 交互式预测
    while True:
        user_input = input("\n请输入一个特征值进行预测(输入'exit'退出): ")
        if user_input.lower() == 'exit':
            break
        try:
            feature_value = float(user_input)
            prediction = final_k * feature_value + final_w
            print(f"预测结果: {prediction:.2f}")
        except ValueError:
            print("请输入有效的数字或'exit'退出。")
    
    print("程序结束")
if __name__ == "__main__":
    main()