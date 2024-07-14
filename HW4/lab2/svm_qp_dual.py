import cvxpy as cp
import numpy as np

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

