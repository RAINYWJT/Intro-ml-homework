import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 固定随机数种子
GLOBAL_SEED = 0

def fix_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)

fix_seed(GLOBAL_SEED)

df_train_x = pd.read_csv('kaggle-data/train_x.csv', index_col='id')
df_train_y = pd.read_csv('kaggle-data/train_y.csv', index_col='id')
df_test_x = pd.read_csv('kaggle-data/test_x.csv', index_col='id')
df_test_y_demo = pd.read_csv('kaggle-data/test_y_demo.csv', index_col='id')

train_x = df_train_x.values
train_y = df_train_y.values.reshape((-1,))
test_x = df_test_x.values

scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)

class AudioDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return torch.tensor(sample, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

class SimpleNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

    model.eval()
    val_preds = []
    val_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            val_preds.extend(predicted.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_accuracy = accuracy_score(val_labels, val_preds)
    print(f"Validation Accuracy: {val_accuracy}")
    return val_accuracy

kf = KFold(n_splits=5, shuffle=True, random_state=GLOBAL_SEED)
input_size = train_x.shape[1]
num_classes = len(np.unique(train_y))
criterion = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

best_model = None
best_accuracy = 0.0

for fold, (train_idx, val_idx) in enumerate(kf.split(train_x)):
    print(f"Fold {fold + 1}")
    train_data = train_x[train_idx], train_y[train_idx]
    val_data = train_x[val_idx], train_y[val_idx]

    train_dataset = AudioDataset(*train_data)
    val_dataset = AudioDataset(*val_data)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = SimpleNN(input_size, num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    val_accuracy = train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, num_epochs=100, device=device)

    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_model = model

test_dataset = AudioDataset(test_x, np.zeros(test_x.shape[0]))
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

best_model.eval()
test_y = []
with torch.no_grad():
    for inputs, _ in test_loader:
        inputs = inputs.to(device)
        outputs = best_model(inputs)
        _, predicted = torch.max(outputs, 1)
        test_y.extend(predicted.cpu().numpy())

df = pd.read_csv('kaggle-data/test_y_demo.csv', index_col='id')
df['y'] = test_y
df.to_csv('test_y.csv')