'''
求导数练习
'''

def F (X, k=1, b=0):
    '''
    简单的一元一次函数
    '''
    return k*X + b

def Lim (X, k=1, b=0):
    '''
    求导
    '''
    LimDelta = 1e-7 #1.0*10^-7
    dF = (F(X + LimDelta, k, b) - F(X, k, b)) / LimDelta
    return dF

def Lim_Central(X, k=1, b=0):
    '''
    求导，但是中心差分
    '''
    LimDelta = 1e-7 #依旧1.0*10^-7
    dF = (F(X + LimDelta, k, b) - F(X - LimDelta, k, b)) / (2 * LimDelta)
    return dF

def main():
    x = 5
    y = F(x) #点(5, f(5))
    dF = Lim_Central(x)
    print(f'函数f(x)=k*x+b, 其中k={1}, b={0}')
    print(f'在点x={x}处的导数为{dF}')

if __name__ == "__main__":
    main()