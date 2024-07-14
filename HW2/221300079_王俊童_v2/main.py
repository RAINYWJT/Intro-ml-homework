from utils import verify_sha256, load_coil20
from utils import LDAMeta, RidgeMeta
from utils import lda_sanity_check, classifier_2_sanity_check, classifier_n_sanity_check

import multiprocessing as mp

import tqdm
import numpy as np
import matplotlib.pyplot as plt

COIL20_ZIP_PATH = "./coil-20-proc.zip"
COIL20_HASH_STR = "517c5594820eb40066ba0ff6842e7f09392bf7fde849bf9cb9c28445b0f29e88"

# NOTE 你可能会用到这两个函数
from scipy.linalg import pinv, eig

class LDA:
    def __init__(self, n_dimension: int) -> None:
        """
        初始化超参数

        参数:
            n_dimension: int
                需要降维到的维度
        """
        assert n_dimension > 0
        self.n_dimension = n_dimension
        self.W = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        训练模型

        参数:
            x: np.ndarray
                形状为 (train_size, dimension) 的图像数据，类型为 np.float32
            y: np.ndarray
                形状为 (train_size,) 的标签数据，顺序对应于 x，类型为 np.int32

        返回:
            无返回值
        """
        x = x.astype(np.float32)
        y = y.astype(np.int32)

        n = y.max() + 1
        m, d = x.shape

        mean_total = np.mean(x, axis=0)

        Sw = np.zeros((d, d))
        Sb = np.zeros((d, d))

        # cal eigenvalue
        for i in range(n):
            xi = x[y == i]
            mean_class = np.mean(xi, axis=0)
            Sw += (xi - mean_class).T @ (xi - mean_class)
            Sb += m * (mean_class - mean_total).reshape(-1, 1) @ (mean_class - mean_total).reshape(1, -1)

        # eigenvalue get
        eigenvalues, eigenvectors = eig(pinv(Sw) @ Sb)
        idx = np.argsort(eigenvalues)[::-1]
        W = eigenvectors[:, idx][:, :self.n_dimension]
        self.W = W

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        测试模型

        参数:
            x: np.ndarray
                形状为 (test_size, dimension) 的图像数据，类型为 np.float32

        返回:
            y: np.ndarray
                形状为 (test_size, n_dimension) 的降维后数据，类型为 np.float32
        """
        x = x.astype(np.float32)
        result = x @ self.W
        return result.astype(np.float32)


class Ridge2(RidgeMeta):
    """Ridge2 类, 提供 fit 和 predict 两个方法"""

    def __init__(self, Lambda: float) -> None:
        """
        初始化超参数

        参数
            Lambda: float
                正则化系数 λ
        """

        assert Lambda > 0.0
        self.Lambda = Lambda
        self.w = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        训练模型

        参数
            x: np.ndarray
                形状为 (train_size, dimension) 图像数据, 类型 np.float32
            y: np.ndarray
                形状为 (train_size,) 标签数据, 顺序对应于 x, 类型为 np.int32

        返回
            无返回值
        """
        x = x.astype(np.float32)
        y = y.astype(np.int32)
        # n 是类别个数; m 是训练样本数量; d 是样例维度.
        n = y.max() + 1
        m, d = x.shape

        # 这里约定只有两类
        assert n == 2
        y_binary = np.where(y == 1, 1, -1)

        x_bias = np.c_[x, np.ones(m)]     
        # cal w
        self.w = np.linalg.inv(x_bias.T @ x_bias + self.Lambda * np.eye(d + 1)) @ x_bias.T @ y_binary

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        测试模型

        参数
            x: np.ndarray
                形状为 (test_size, dimension) 图像数据, 类型 np.float32

        返回
            y: np.ndarray
                形状为 (test_size,) 标签数据, 顺序对应于 x, 类型为 np.int32
        """

        # n 是类别个数; m 是训练样本数量; d 是样例维度.
        x = x.astype(np.float32)
        m, d = x.shape
        x_bias = np.c_[x, np.ones(m)]
        y_pred = np.sign(x_bias @ self.w)
        y_pred_binary = np.where(y_pred == 1, 1, 0)
        return y_pred_binary.astype(np.int32)


class RidgeN(RidgeMeta):
    """RidgeN 类, 提供 fit 和 predict 两个方法"""

    def __init__(self, Lambda: float) -> None:
        """
        初始化超参数

        参数
            Lambda: float
                正则化系数 λ
        """

        assert Lambda > 0.0
        self.Lambda = Lambda

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        训练模型

        参数
            x: np.ndarray
                形状为 (train_size, dimension) 图像数据, 类型 np.float32
            y: np.ndarray
                形状为 (train_size,) 标签数据, 顺序对应于 x, 类型为 np.int32

        返回
            无返回值
        """
        x = x.astype(np.float32)
        y = y.astype(np.int32)
        # n 是类别个数; m 是训练样本数量; d 是样例维度.
        n = y.max() + 1
        m, d = x.shape

        self.n = n
        self.ridge2_pool = [Ridge2(self.Lambda) for k in range(n)]

        for k in range(n):
            y_binary = np.where(y == k, 1, -1)
            self.ridge2_pool[k].fit(x, y_binary)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        测试模型

        参数
            x: np.ndarray
                形状为 (test_size, dimension) 图像数据, 类型 np.float32

        返回
            y: np.ndarray
                形状为 (test_size,) 标签数据, 顺序对应于 x, 类型为 np.int32
        """
        x = x.astype(np.float32)
        # n 是类别个数; m 是训练样本数量; d 是样例维度.
        m, d = x.shape
        scores = np.zeros((m, self.n))
        for k in range(self.n):
            scores[:, k] = self.ridge2_pool[k].predict(x)
        y_pred = np.argmax(scores, axis=1)
        return y_pred.astype(np.int32)


def main():
    verify_sha256(COIL20_ZIP_PATH, COIL20_HASH_STR)
    x_train, y_train, x_test, y_test = load_coil20(COIL20_ZIP_PATH)

    # 检查输入输出格式

    lda_sanity_check(LDA)
    print("LDA finish!!!")
    classifier_2_sanity_check(Ridge2)
    print("Ridge2 finish!!!")
    classifier_n_sanity_check(RidgeN)
    print("RidgeN finish!!!")

    # 把训练数据降至两维并绘制散点图
    # from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
    # lda = LinearDiscriminantAnalysis (n_components=2)
    lda = LDA(n_dimension = 2)
    lda.fit(x_train, y_train)
    x_train_2d = lda.transform(x_train)
    # print(x_train_2d.shape)

    plt.figure(figsize=(8, 6))
    plt.scatter(x_train_2d[:, 0], x_train_2d[:, 1], c=y_train)
    plt.title('Train LDA scatter graph')
    plt.xlabel('Principle 1')
    plt.ylabel('Principle 2')
    plt.savefig('Train LDA scatter graph')
    plt.show()
    
    # 把测试数据降至两维并绘制散点图

    # lda = LinearDiscriminantAnalysis (n_components=2)
    lda = LDA(n_dimension = 2)
    lda.fit(x_train, y_train)
    x_test_2d = lda.transform(x_test)
    # print(x_test_2d.shape)


    plt.figure(figsize=(8, 6))
    plt.scatter(x_test_2d[:, 0], x_test_2d[:, 1], c=y_test)
    plt.title('Test LDA scatter graph')
    plt.xlabel('Principle 1')
    plt.ylabel('Principle 2')
    plt.savefig('Test LDA scatter graph')
    plt.show()

    # 训练错误率和测试错误率随 λ 变化的折线图

    Lambda_seq = np.logspace(start=-4, stop=4, num=9).tolist()
    # print(Lambda_seq)
    train_error_rates = []
    test_error_rates = []

    for lambda_value in Lambda_seq:
        ridgen = RidgeN(lambda_value)
        ridgen.fit(x_train, y_train)

        train_pred = ridgen.predict(x_train)
        train_errors = np.sum(train_pred != y_train)  
        train_error_rate = train_errors / len(y_train)  
        train_error_rates.append(train_error_rate)

        test_pred = ridgen.predict(x_test)
        test_errors = np.sum(test_pred != y_test)  
        test_error_rate = test_errors / len(y_test) 
        test_error_rates.append(test_error_rate)


    print("训练错误率列表:", train_error_rates)
    print("测试错误率列表:", test_error_rates)
    plt.figure()
    plt.plot(Lambda_seq, train_error_rates, marker='o', color='b', label='Train Error Rate')
    plt.plot(Lambda_seq, test_error_rates, marker='s', color='r', label='Test Error Rate')

    plt.xscale('log') 
    plt.xlabel('Lambda')
    plt.ylabel('Error Rate')
    plt.title('Training and Test Error Rates vs. Lambda')
    plt.legend()
    plt.grid(True)
    plt.savefig('HW2')  
    plt.show()


if __name__ == "__main__":
    main()
