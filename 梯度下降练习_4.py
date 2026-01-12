"""
2026/1/12
梯度下降练习_4.py
喵了个咪的最后练一个训练集我就去练下一个
"""
import numpy as np

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
X, y, true_coef = generate_linear_dataset(n_samples=1000, seed=42)

# 打印数据集基本信息
print("数据集形状:")
print(f"X: {X.shape}")  # (1000, 10)
print(f"y: {y.shape}")  # (1000,)

print("\n前5个样本的特征值:")
print(X[:5])

print("\n前5个样本的目标值:")
print(y[:5])

print("\n特征统计信息:")
for i in range(10):
    print(f"x{i+1}: 均值={X[:, i].mean():.2f}, 标准差={X[:, i].std():.2f}, "
          f"范围=[{X[:, i].min():.2f}, {X[:, i].max():.2f}]")

print("\n真实系数（包括截距）:")
coef_names = ['截距'] + [f'x{i+1}' for i in range(10)]
for name, value in zip(coef_names, true_coef):
    print(f"{name}: {value}")

print(f"\ny的统计信息: 均值={y.mean():.2f}, 标准差={y.std():.2f}")