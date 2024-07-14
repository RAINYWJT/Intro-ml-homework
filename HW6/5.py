import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import random
import multiprocessing as mp
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import optuna
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
# 定义LSTM模型
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# 训练LSTM模型
def train_lstm(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

# 训练基模型
def train_base_models(train_x, train_y):
    svm_model = SVC()
    rf_model = RandomForestClassifier()
    svm_model.fit(train_x, train_y)
    rf_model.fit(train_x, train_y)
    return svm_model, rf_model

# 定义元模型
class MetaModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MetaModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 训练元模型
def train_meta_model(meta_model, meta_train_loader, criterion, optimizer, num_epochs):
    meta_model.train()
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(meta_train_loader):
            optimizer.zero_grad()
            outputs = meta_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

# 主函数
def main():
    # 数据准备
    train_x_tensor = torch.tensor(train_x, dtype=torch.float32)
    train_y_tensor = torch.tensor(train_y, dtype=torch.long)
    test_x_tensor = torch.tensor(test_x, dtype=torch.float32)

    train_dataset = TensorDataset(train_x_tensor, train_y_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 训练LSTM模型
    lstm_model = LSTMClassifier(input_size=train_x.shape[1], hidden_size=128, num_layers=2, num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
    train_lstm(lstm_model, train_loader, criterion, optimizer, num_epochs=10)

    # 训练基模型
    svm_model, rf_model = train_base_models(train_x, train_y)

    # 生成元数据
    lstm_preds = lstm_model(train_x_tensor).argmax(dim=1).numpy()
    svm_preds = svm_model.predict(train_x)
    rf_preds = rf_model.predict(train_x)

    meta_train_x = np.column_stack((lstm_preds, svm_preds, rf_preds))
    meta_train_y = train_y

    meta_train_x_tensor = torch.tensor(meta_train_x, dtype=torch.float32)
    meta_train_y_tensor = torch.tensor(meta_train_y, dtype=torch.long)

    meta_train_dataset = TensorDataset(meta_train_x_tensor, meta_train_y_tensor)
    meta_train_loader = DataLoader(meta_train_dataset, batch_size=64, shuffle=True)

    # 训练元模型
    meta_model = MetaModel(input_size=meta_train_x.shape[1], hidden_size=64, num_classes=2)
    meta_optimizer = optim.Adam(meta_model.parameters(), lr=0.001)
    train_meta_model(meta_model, meta_train_loader, criterion, meta_optimizer, num_epochs=10)

    # 预测和评估
    lstm_test_preds = lstm_model(test_x_tensor).argmax(dim=1).numpy()
    svm_test_preds = svm_model.predict(test_x)
    rf_test_preds = rf_model.predict(test_x)

    meta_test_x = np.column_stack((lstm_test_preds, svm_test_preds, rf_test_preds))
    meta_test_x_tensor = torch.tensor(meta_test_x, dtype=torch.float32)

    meta_model.eval()
    with torch.no_grad():
        meta_test_preds = meta_model(meta_test_x_tensor).argmax(dim=1).numpy()

    # 将预测结果存储到 CSV 文件中
    df_test_y = pd.DataFrame(meta_test_preds, columns=['y'], index=df_test_x.index)
    df_test_y.index.name = 'id'
    df_test_y.to_csv('test_y.csv')

if __name__ == '__main__':
    main()