from sklearn import svm
import numpy as np

def svm_kernel_linear(X, Y):
    '''
    :参数 X: ndarray, 形状(m, d), 样例矩阵
    :参数 Y: ndarray, 形状(m), 样例标签向量
    :返回: clf_linear, 训练好的分类器
    '''
    clf_linear = svm.SVC(kernel='linear', C=1.0)
    clf_linear.fit(X, Y)
    return clf_linear

def svm_kernel_rbf(X, Y):
    '''
    :参数 X: ndarray, 形状(m, d), 样例矩阵
    :参数 Y: ndarray, 形状(m), 样例标签向量
    :返回: clf_rbf, 训练好的分类器
    '''
    clf_rbf = svm.SVC(kernel='rbf', C=1.0)
    clf_rbf.fit(X, Y)
    return clf_rbf

def custom_kernel(X1, X2):
    '''
    :参数 X1: ndarray, 形状(m, d)
    :参数 X2: ndarray, 形状(n, d)
    :返回: 形状为(m, n)的Gram矩阵, 第(i,j)个元素为X1[i]和X2[j]之间的核函数值
    '''
    dist_squared = np.sum((X1[:, np.newaxis] - X2)**2, axis=2)
    K = 1 / (1 + dist_squared)
    return K

def svm_kernel_custom(X, Y):
    '''
    :参数 X: ndarray, 形状(m, d), 样例矩阵
    :参数 Y: ndarray, 形状(m), 样例标签向量
    :返回: clf_custom, 训练好的分类器
    '''
    clf_custom = svm.SVC(kernel=lambda X1, X2: custom_kernel(X1, X2), C=1.0)
    clf_custom.fit(X, Y)
    return clf_custom