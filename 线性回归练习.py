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

def Lim ():
    '''
    由于我觉得求极限太麻烦了因此直接写个函数来导，望周知
    '''
    LimDelta = #我去我还没写完但是不管了我先睡个觉先

def F (X, a , b, c):
    '''
    看得出来是个二次函数对吧，我懒得想函数名了就直接一个大写F得了然后这是初始模型嗯
    '''
    return a*X**2 + b*X + c

def L ():
    '''
    损失函数
    '''
    m = len(X_train)    #样本数量
    error = 0
    for i in range(m):
        error += (F(X_train[i], a, b, c) - y_train[i])**2   #套f(x)带到方差
    total_error = error / (2*m)     #方差算完算损失
    return total_error

def J ():
    '''
    对损失函数求偏导
    '''
    m = len(X_train)
    da, da, dc = 0, 0, 0    #初始化偏导数
    da = 

def main():
    pass

if __name__ == "__main__":
    main()