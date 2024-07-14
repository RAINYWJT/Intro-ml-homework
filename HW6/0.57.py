import os
import re
import glob
import time
import random
import datetime
import multiprocessing as mp
import numpy as np
import pandas as pd
import sklearn
import joblib
import torch
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

matplotlib.rcParams['figure.dpi'] = 128
matplotlib.rcParams['figure.figsize'] = (8, 6)

# 固定随机数种子
GLOBAL_SEED = 0

def fix_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

fix_seed(GLOBAL_SEED)

# 加载数据
df_train_x = pd.read_csv('kaggle-data/train_x.csv', index_col='id')
df_train_y = pd.read_csv('kaggle-data/train_y.csv', index_col='id')
df_test_x = pd.read_csv('kaggle-data/test_x.csv', index_col='id')
df_test_y_demo = pd.read_csv('kaggle-data/test_y_demo.csv', index_col='id')

train_x = df_train_x.values
train_y = df_train_y.values.reshape((-1,))
test_x = df_test_x.values

def standardize(x, mu=None, sigma=None):
    if mu is None:
        mu = np.mean(x, axis=0)
    if sigma is None:
        sigma = np.std(x, axis=0) + 1e-3
    return (x - mu) / sigma, mu, sigma

train_x, mu, sigma = standardize(train_x)
test_x, _, _ = standardize(test_x, mu, sigma)

param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto', 0.1, 1, 10],
    'degree': [2, 3, 4] 
}

svc = SVC(random_state=GLOBAL_SEED)
grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(train_x, train_y)
print("Best parameters found: ", grid_search.best_params_)
best_svc = grid_search.best_estimator_
test_y = best_svc.predict(test_x)
df = pd.read_csv('kaggle-data/test_y_demo.csv', index_col='id')
df['y'] = test_y  
df.to_csv('test_y.csv')
