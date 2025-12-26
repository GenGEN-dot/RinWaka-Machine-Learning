'''
线性回归手感练习
额三百年没写python了最近都在写arduino导致python手感全无哈哈
好吧先练练手
2025/12/25……欸还是圣诞节欸
'''

#首先先确认一下实际模型吧，y=x^2……欸说起来我只写过一元一次方程的二次的能用线性回归吗算了写了就知道了

X_train = [1,2,3,4,5]
y_train = [1,4,9,16,25]  #实际模型y=x^2
a, b, c=1, 1, 1  #初始化参数

def F (X, a , b, c):
    '''
    看得出来是个二次函数对吧，我懒得想函数名了就直接一个大写F得了然后这是初始模型嗯
    '''
    return a*X**2 + b*X + c

def L (X_train, a, b, c):
    '''
    损失函数
    '''
    m = len(X_train)    #样本数量
    error = 0
    for i in range(m):
        error += (F(X_train[i], a, b, c) - y_train[i])**2   #套f(x)带到方差
    total_error = error / (2*m)     #方差算完算损失
    return total_error

def J (X_train, a, b, c):
    '''
    对损失函数求偏导
    '''
    m = len(X_train)
    da, db, dc = 0, 0, 0    #初始化偏导数
    for i in range(m):
        # 计算预测值和误差
        prediction = F(X_train[i], a, b, c)
        error = prediction - y_train[i]
        
        # 根据公式计算梯度：
        # ∂L/∂a = (1/m) Σ [(F(x_i) - y_i) * x_i²]
        # ∂L/∂b = (1/m) Σ [(F(x_i) - y_i) * x_i]
        # ∂L/∂c = (1/m) Σ [(F(x_i) - y_i)]
        da += error * (X_train[i]**2)   #对a求偏导
        db += error * X_train[i]        #对b求偏导
        dc += error                     #对c求偏导
        
    da /= m
    db /= m
    dc /= m
    return da, db, dc

def main():
    global a, b, c
    learning_rate = 0.005   #学习率
    epochs = 50000         #迭代次数

    for epoch in range(epochs):
        da, db, dc = J(X_train, a, b, c)   #计算偏导数
        a -= learning_rate * da              #更新参数a
        b -= learning_rate * db              #更新参数b
        c -= learning_rate * dc              #更新参数c

        if epoch % 1000 == 0:
            loss = L(X_train, a, b, c)
            print(f'学习轮次 {epoch}, 损失率: {loss}, a: {a}, b: {b}, c: {c}')

    print(f'最终参数: a: {a}, b: {b}, c: {c}')
    
    # 测试一下
    print("\n测试预测:")
    for x in X_train:
        print(f"F({x}) = {F(x, a, b, c):.2f}, 实际值: {x**2}")

if __name__ == "__main__":
    main()