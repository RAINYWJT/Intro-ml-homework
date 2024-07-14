import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# 固定随机数种子
GLOBAL_SEED = 0
np.random.seed(GLOBAL_SEED)

# 加载数据
df_train_x = pd.read_csv('kaggle-data/train_x.csv', index_col='id')
df_train_y = pd.read_csv('kaggle-data/train_y.csv', index_col='id')
df_test_x = pd.read_csv('kaggle-data/test_x.csv', index_col='id')
df_test_y_demo = pd.read_csv('kaggle-data/test_y_demo.csv', index_col='id')

train_x = df_train_x.values
train_y = df_train_y.values.reshape((-1,))
test_x = df_test_x.values

# 标准化数据
def standardize(x, mu=None, sigma=None):
    if mu is None:
        mu = np.mean(x, axis=0)
    if sigma is None:
        sigma = np.std(x, axis=0) + 1e-3
    return (x - mu) / sigma, mu, sigma

train_x, mu, sigma = standardize(train_x)
test_x, _, _ = standardize(test_x, mu, sigma)

# 分割特征
num_features = 200
num_groups = 14

# 定义超参数网格
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto', 0.1, 1, 10],
    'degree': [2, 3, 4]
}

# 训练子分类器并进行网格搜索
classifiers = []
for i in range(num_groups):
    start_idx = i * num_features
    end_idx = (i + 1) * num_features
    sub_train_x = train_x[:, start_idx:end_idx]
    sub_test_x = test_x[:, start_idx:end_idx]
    print(start_idx, end_idx)
    svc = SVC(random_state=GLOBAL_SEED)
    grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(sub_train_x, train_y)
    best_svc = grid_search.best_estimator_
    classifiers.append(best_svc)

    # 打印最佳超参数
    print(f"Group {i+1} Best Parameters: {grid_search.best_params_}")

# 集成预测
test_preds = np.zeros((test_x.shape[0], num_groups))
for i, clf in enumerate(classifiers):
    start_idx = i * num_features
    end_idx = (i + 1) * num_features
    sub_test_x = test_x[:, start_idx:end_idx]
    test_preds[:, i] = clf.predict(sub_test_x)

# 多数投票集成
final_preds = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=test_preds)

# 保存预测结果
df = pd.read_csv('kaggle-data/test_y_demo.csv', index_col='id')
df['y'] = final_preds
df.to_csv('test_y.csv')