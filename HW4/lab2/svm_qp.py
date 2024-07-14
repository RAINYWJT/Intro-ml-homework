import cvxpy as cp # 建议使用cvxpy 1.4版本, 其他版本也可以
from sklearn.datasets import make_classification
import numpy as np
import matplotlib.pyplot as plt
import time

def solve_primal(X, y, C):
    '''
    :参数 X: ndarray, 形状为(m, d), 样例矩阵
    :参数 y: ndarray, 形状为(m), 样例标签向量
    :参数 C: 标量, 含义与教材式(6.35)中C相同
    :返回: w, b, SVM的权重与偏置
    '''
    m, d = X.shape
    y_ = y.reshape(-1, 1)

    w = cp.Variable((d, 1))
    b = cp.Variable()
    xi = cp.Variable((m, 1))

    loss = cp.sum(xi)
    reg = cp.sum_squares(w)

    # 定义CVXPY优化问题
    prob = cp.Problem(
    	cp.Minimize(0.5 * reg + C * loss), # 目标函数
        [cp.multiply(y_, X @ w + b) >= 1 - xi, xi >= 0] # 约束 
        )

    prob.solve()
    return w, b

def solve_dual(X, y, C):
    '''
    :参数 X: ndarray, 形状为(m, d), 样例矩阵
    :参数 y: ndarray, 形状为(m), 样例标签向量
    :参数 C: 标量, 含义与教材式(6.35)中C相同
    :返回: alpha，SVM的对偶变量
    '''
    m, d = X.shape  
    y = y.reshape(-1, 1) * 1.0  

    alpha = cp.Variable((m, 1),pos=True)
    Q = np.matmul(y * X, (y * X).T)
    prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(alpha, Q) - cp.sum(alpha))
                      ,[alpha <= C, 
                        alpha >= 0, 
                        cp.sum(cp.multiply(alpha,y)) == 0])
    prob.solve()
    return alpha.value


r_values = np.arange(1, 10, 1)
primal_times = []
dual_times = []

for r in r_values:
    m = 100  
    d = int(m * r)  
    X, y = make_classification(m, n_features=d)
    y = (y * 2 - 1)
    C = 1

    time_primal = time.time()
    solve_primal(X, y, C)
    time_primal_end = time.time()
    primal_times.append(time_primal_end - time_primal)

    time_dual = time.time()
    solve_dual(X, y, C)
    time_dual_end = time.time()
    dual_times.append(time_dual_end - time_dual)

plt.figure(figsize=(10, 6))
plt.plot(r_values, primal_times, label='Primal')
plt.plot(r_values, dual_times, label='Dual')
plt.xlabel('r = d/m')
plt.ylabel('Time (s)')
plt.title('Solving Time vs. r')
plt.legend()
plt.grid(True)
plt.savefig('Primal-dual Problem solving time from 1 to 10')
plt.show()