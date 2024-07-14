import numpy as np
from sklearn.decomposition import PCA
import math
# 样本数据
X = np.array([[2, 3, 3, 4, 5, 7],
              [2, 4, 5, 5, 6, 8]])

# 计算各维的均值和标准差
mean = np.mean(X, axis=1)
std_dev = np.std(X, axis=1)
print("标准差: ", std_dev)

# 标准化样本矩阵
X_std = (X - mean.reshape(-1, 1)) / std_dev.reshape(-1, 1)
print("标准化后的样本矩阵: \n", X_std)

# 计算协方差矩阵
cov_matrix = np.cov(X_std)
print("协方差矩阵: \n", cov_matrix)

# 计算特征值和特征向量
eig_vals, eig_vecs = np.linalg.eig(cov_matrix)
print("特征值: ", eig_vals)
print("特征向量: \n", eig_vecs)

# 投影矩阵
W = eig_vecs[:, :1]
print("投影矩阵: \n", W)

# 重构阈值
t = 0.95

# PCA
pca = PCA(n_components=t)
print(X_std.T)
X_pca = pca.fit_transform(X_std.T)
print("PCA后样本在新空间的坐标矩阵: \n", X_pca)
