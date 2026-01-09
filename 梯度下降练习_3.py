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

#============

import random

#============

class Gradient_Descent:

    """
    梯度下降
    """

    def __init__(self, k_init=None, gradient_threshold=1e-15, param_threshold=1e-6, loss_threshold=1e-15):
        if k_init is None:
            self.k = [0.00, 0.00, 0.00]         # 默认参数：k0, k1, k2
        else:
            self.k = k_init.copy()              # 使用传入的初始参数
        
        self.predictions = []               # 存储当前参数的预测值
        self.X_current = None               # 当前使用的数据X（用于检查是否需要重新计算）
        self.k_current = None               # 当前用于计算预测值的参数（用于检查是否需要重新计算）
        self.k_target = [2.0, 3.0, 4.0]     # K值理论值，嘛也只有这种真的完美契合的能达到理论吧
        
        # 提前停止条件的阈值
        self.gradient_threshold = gradient_threshold  # 梯度接近0的阈值
        self.param_threshold = param_threshold        # 参数接近理论值的阈值
        self.loss_threshold = loss_threshold          # 损失足够小的阈值

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
    
    def predict(self, X_sample):
        """
        对单个样本进行预测
        """
        return sum(self.k[j] * X_sample[j] for j in range(len(self.k)))
    
    def compute_loss(self, X, y):
        """
        Step1: 计算损失
        """

        m = len(X)
        total_error = 0.00
        predictions = self.compute_predictions(X)

        for i in range (m):
            prediction = predictions[i]
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

        predictions = self.compute_predictions(X)

        for i in range(m):
            prediction = predictions[i]
            error = prediction - y[i]
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
    
    def is_close_to_target(self):
        """
        检查参数是否接近理论值
        使用self.param_threshold作为阈值
        """
        for i in range(len(self.k)):
            if abs(self.k[i] - self.k_target[i]) > self.param_threshold:
                return False
        return True

    def is_good_enough(self, X, y):
        """
        检查模型是否足够好（损失足够小）
        使用self.loss_threshold作为阈值
        """    
        # 检查是否"足够好"
        # 在实际中，我们通常看损失是否小于某个阈值
        return self.compute_loss(X, y) < self.loss_threshold

    def train(self, X, y, learning_rate = 0.01, epochs = 10000, verbose = True):
        """
        完整训练流程
        """
        losses = []

        for epoch in range(epochs):
            
            #计算损失
            loss = self.compute_loss(X, y)
            losses.append(loss)

            #计算梯度
            gradients = self.compute_gradients(X, y)

            #更新参数
            self.updata_parameters(gradients, learning_rate)

            if verbose and epoch % 1000 == 0:
                print(f"学习轮次: {epoch}, 损失率: {loss:.6f}, 参数: [{self.k[0]:.6f}, {self.k[1]:.6f}, {self.k[2]:.6f}]")
            
            # 提前停止条件1: 梯度接近于0
            if max(abs(g) for g in gradients) < self.gradient_threshold:
                if verbose:
                    print(f"提前停止于轮次 {epoch}，梯度已接近零（阈值: {self.gradient_threshold:.1e}）")
                break
            
            # 提前停止条件2：参数接近理论值
            elif self.is_close_to_target():
                if verbose:
                    print(f"提前停止于轮次 {epoch}，参数已训练至理论值（阈值: {self.param_threshold:.1e}）")
                    print(f"当前参数: {[round(v, 12) for v in self.k]}")
                    print(f"目标参数: {self.k_target}")
                break

            # 提前停止条件3：损失足够小
            elif self.compute_loss(X, y) < self.loss_threshold:
                if verbose:
                    print(f"提前停止于轮次 {epoch}，损失已足够小（阈值: {self.loss_threshold:.1e}）")
                break
        
        if verbose:
            final_loss = self.compute_loss(X, y)
            print(f"\n最终参数: [{self.k[0]:.12f}, {self.k[1]:.12f}, {self.k[2]:.12f}]")
            print(f"最终损失(MSE): {final_loss:.12f}")
            print(f"均方根误差(RMSE): {final_loss**0.5:.12f}")


def get_float_input(prompt, default_value):
    """获取浮点数输入，支持默认值"""
    user_input = input(prompt)
    if not user_input.strip():
        return default_value
    try:
        return float(user_input)
    except ValueError:
        print(f"无效输入，使用默认值: {default_value}")
        return default_value

def get_int_input(prompt, default_value):
    """获取整数输入，支持默认值"""
    user_input = input(prompt)
    if not user_input.strip():
        return default_value
    try:
        return int(user_input)
    except ValueError:
        print(f"无效输入，使用默认值: {default_value}")
        return default_value

def main():
    print("=" * 50)
    print("梯度下降练习 - 自定义参数初始化")
    print("=" * 50)
    
    # 1. 选择是否启用随机K初始化
    use_random_k = input("是否启用随机K初始化？(y/n, 默认n): ").strip().lower()
    
    if use_random_k == 'y':
        print("使用随机初始化K值")
        k_init = [random.uniform(-0.1, 0.1) for _ in range(3)]
    else:
        k_input = input("请输入初始K值（三个数，用逗号分隔，默认[0,0,0]）: ").strip()
        if k_input:
            try:
                parts = k_input.split(',')
                if len(parts) == 3:
                    k_init = [float(part.strip()) for part in parts]
                    print(f"使用自定义K值: {k_init}")
                else:
                    print("输入格式错误，使用默认值[0,0,0]")
                    k_init = [0.0, 0.0, 0.0]
            except ValueError:
                print("输入包含非数字，使用默认值[0,0,0]")
                k_init = [0.0, 0.0, 0.0]
        else:
            k_init = [0.0, 0.0, 0.0]
            print("使用默认K值[0,0,0]")
    
    # 2. 设置提前停止条件的阈值
    print("\n" + "=" * 50)
    print("设置提前停止条件阈值")
    print("=" * 50)
    print("当前默认值:")
    print(f"1. 梯度接近0的阈值: 1e-15")
    print(f"2. 参数接近理论值的阈值: 1e-6")
    print(f"3. 损失足够小的阈值: 1e-15")
    print("\n是否修改这些阈值？(y/n, 默认n): ", end="")
    
    modify_thresholds = input().strip().lower()
    
    if modify_thresholds == 'y':
        print("\n请输入新的阈值（按回车使用默认值）:")
        gradient_threshold = get_float_input("1. 梯度接近0的阈值 (默认1e-15): ", 1e-15)
        param_threshold = get_float_input("2. 参数接近理论值的阈值 (默认1e-6): ", 1e-6)
        loss_threshold = get_float_input("3. 损失足够小的阈值 (默认1e-15): ", 1e-15)
    else:
        gradient_threshold = 1e-15
        param_threshold = 1e-6
        loss_threshold = 1e-15
        print("使用默认阈值")
    
    # 创建模型实例
    model = Gradient_Descent(
        k_init=k_init,
        gradient_threshold=gradient_threshold,
        param_threshold=param_threshold,
        loss_threshold=loss_threshold
    )
    
    # 3. 获取训练参数
    print("\n" + "=" * 50)
    print("设置训练参数")
    print("=" * 50)
    
    epochs = get_int_input("请输入学习轮次 (默认10000): ", 10000)
    learning_rate = get_float_input("请输入学习率 (默认0.01): ", 0.01)
    
    # 使用用户指定的参数进行训练
    print(f"\n开始训练:")
    print(f"初始K值: {model.k}")
    print(f"学习轮次: {epochs}")
    print(f"学习率: {learning_rate}")
    print(f"梯度阈值: {gradient_threshold:.1e}")
    print(f"参数阈值: {param_threshold:.1e}")
    print(f"损失阈值: {loss_threshold:.1e}")
    
    model.train(X_train, y_label, learning_rate=learning_rate, epochs=epochs, verbose=True)

    # 4. 预测循环
    while True:
        print("\n" + "=" * 50)
        User_Input = input("请输入新的数据以测试当前模型，或输入\"exit\"以退出。\n"
                   f"输入格式: x1, x2 （两个特征值）\n"
                   f"当前参数: [{model.k[0]:.12f}, {model.k[1]:.12f}, {model.k[2]:.12f}]\n"
                   f"请输入: ")
        if User_Input.lower() == "exit":
            print("退出程序。")
            break
        try:
            parts = User_Input.split(',')
            if len(parts) != 2:
                print("错误：请输入两个数字，用逗号分隔。")
                continue
            x0 = float(parts[0].strip())
            x1 = float(parts[1].strip())
            features = [1, x0, x1]  # 偏置项1

            # 使用模型进行预测
            prediction = model.predict(features)

            # 计算理论值（用于比较）
            theoretical = 2 + 3*x0 + 4*x1

            print(f"\n预测结果:")
            print(f"特征值: x0=1, x1={x0}, x2={x1}")
            print(f"模型预测值: {prediction:.12f}")
            print(f"理论值: {theoretical:.12f}")
            print(f"预测误差: {abs(prediction - theoretical):.12f}")
            print(f"相对误差: {abs(prediction - theoretical)/abs(theoretical)*100:.6f}%" if theoretical != 0 else "相对误差: 无穷大 (理论值为0)")
            
        except ValueError:
            print("输入无效，请确保输入的是两个有效的数字。")

if __name__ == "__main__":
    main()