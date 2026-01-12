"""
2026/1/12
梯度下降练习_4.py
喵了个咪的最后练一个训练集我就去练下一个
"""
import numpy as np

#============

#生成数据集

def generate_linear_dataset(n_samples=1000, seed=42):
    """
    生成具有不同尺度特征的线性数据集
    
    参数:
    n_samples: 样本数量，默认为1000
    seed: 随机种子，默认为42
    
    返回:
    X: 特征矩阵 (n_samples, 10)
    y: 目标值 (n_samples,)
    true_coef: 真实系数 (包括截距)
    """
    # 设置随机种子以便复现结果
    np.random.seed(seed)
    
    # 生成不同尺度的特征
    n_features = 10
    
    # 为每个特征指定不同的范围
    feature_ranges = [
        (0, 1),          # x1: 0-1
        (0, 100),        # x2: 0-100
        (-10, 10),       # x3: -10到10
        (0, 1),          # x4: 0-1
        (0, 100),        # x5: 0-100
        (-10, 10),       # x6: -10到10
        (0, 1),          # x7: 0-1
        (0, 100),        # x8: 0-100
        (-10, 10),       # x9: -10到10
        (0, 1)           # x10: 0-1
    ]
    
    # 生成特征矩阵 X
    X = np.zeros((n_samples, n_features))
    for i, (low, high) in enumerate(feature_ranges):
        X[:, i] = np.random.uniform(low, high, n_samples)
    
    # 真实系数（包括截距）
    true_intercept = 2.0
    true_coefficients = np.array([1.5, -0.8, 2.2, -1.1, 0.5, 3.0, -2.0, 0.3, 1.8, -0.7])
    
    # 计算无噪声的y值
    y_true = true_intercept + X @ true_coefficients
    
    # 添加高斯噪声（标准差0.5）
    noise = np.random.normal(0, 0.5, n_samples)
    y = y_true + noise
    
    # 将截距也加入到系数数组中
    true_coef = np.concatenate([[true_intercept], true_coefficients])
    
    return X, y, true_coef

# 生成数据集
X_train, y_label, true_coef = generate_linear_dataset(n_samples=1000, seed=42)

# 打印数据集基本信息
print("数据集形状:")
print(f"X: {X_train.shape}")  # (1000, 10)
print(f"y: {y_label.shape}")  # (1000,)

print("\n前5个样本的特征值:")
print(X_train[:5])

print("\n前5个样本的目标值:")
print(y_label[:5])

print("\n特征统计信息:")
for i in range(10):
    print(f"x{i+1}: 均值={X_train[:, i].mean():.2f}, 标准差={X_train[:, i].std():.2f}, "
          f"范围=[{X_train[:, i].min():.2f}, {X_train[:, i].max():.2f}]")

print("\n真实系数（包括截距）:")
coef_names = ['截距'] + [f'x{i+1}' for i in range(10)]
for name, value in zip(coef_names, true_coef):
    print(f"{name}: {value}")

print(f"\ny的统计信息: 均值={y_label.mean():.2f}, 标准差={y_label.std():.2f}")

#============

class Gradient_Descent:
    """
    具体的训练流程
    """

    def __init__(self, k_init=None, gradient_threshold=1e-15, param_threshold=1e-6, loss_threshold=1e-15):
        """
        初始化梯度下降类
        
        参数:
        k_init: 初始参数传入, 11个
        gradient_threshold: 梯度阈值, 当梯度小于此值时停止迭代
        param_threshold: 参数变化阈值, 当参数变化小于此值时停止迭代
        loss_threshold: 损失变化阈值, 当损失变化小于此值时停止迭代
        """
        # 修正初始化：11个参数（截距+10个特征权重）
        if k_init is None:
            # 初始化11个参数，全部设为0.00
            self.k = [0.00 for _ in range(11)]  # 11个参数：k[0]是截距，k[1]-k[10]是特征权重
        else:
            self.k = k_init.copy()  # 使用传入的初始参数
        
        self.predictions = []  # 存储当前参数的预测值
        self.X_current = None  # 当前使用的数据X（用于检查是否需要重新计算）
        self.k_current = None  # 当前用于计算预测值的参数（用于检查是否需要重新计算）
        
        # 提前停止条件的阈值
        self.gradient_threshold = gradient_threshold
        self.param_threshold = param_threshold
        self.loss_threshold = loss_threshold
        
        # 添加一些训练过程中的记录
        self.loss_history = []  # 记录每次迭代的损失值
        self.gradient_history = []  # 记录每次迭代的梯度

    def compute_predictions(self, X, force_recompute=False):
        """
        计算所有样本的预测值
        force_recompute: 强制重新计算（即使参数和数据没变）
        
        为什么要有这个函数？
        - 避免在compute_loss和compute_gradients中重复计算预测值
        - 提高效率，尤其是在数据量大时

        pred: 预测值列表
        X_current: 当前使用的数据X
        k_current: 当前用于计算预测值的参数
        """
        # 检查是否需要重新计算：
        # 1. 预测值列表为空（第一次计算）
        # 2. 数据X变了
        # 3. 参数k变了
        # 4. 强制重新计算
        if (not self.predictions or X != self.X_current or self.k != self.k_current or force_recompute):
            
            self.predictions = []  # 清空旧的预测值
            for i in range(len(X)):
                pred = sum(
                    self.k[j] * X[i][j] 
                        for j in range(len(self.k))
                        )
                self.predictions.append(pred)
            
            # 记录当前状态，方便下次检查
            self.X_current = X[:] if isinstance(X, list) else X.copy()
            self.k_current = self.k.copy()
        
        return self.predictions
    
    def predict(self, X_sample):
        """
        对单个样本进行预测
        """
        return sum(self.k[j] * X_sample[j] for j in range(len(self.k)))
    
    def compute_loss(self, X, y):
        """
        损失计算

        """

        m = len(X)          #获取列表长
        total = 0.00        #损失初始化

        for i in range(m):
            prediction = self.compute_predictions - y