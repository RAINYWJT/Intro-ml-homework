from torch import nn
import torch.nn.functional as F
# You can add more imports here, if needed
import torch.optim as optim

class FashionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 512)  # Input layer
        self.fc2 = nn.Linear(512, 256)  # Hidden layer 1
        self.fc3 = nn.Linear(256, 128)  # Hidden layer 2
        self.fc4 = nn.Linear(128, 10)   # Output layer
        """
        tricks to solve overfitting
        """
        # trick 1: add dropout to every layer to solve the situation of overfitting
        # self.dropout = nn.Dropout(p=0.4) 

        # trick 2: add normolization to model to solve the situation
        self.optimizer = optim.SGD(self.parameters(), lr=0.01, weight_decay=0.05)


    def forward(self, x):
        x = self.flatten(x)
        
        x = F.relu(self.fc1(x))
        # x = self.dropout(x) # trick 1
        x = F.relu(self.fc2(x))
        # x = self.dropout(x) # trick 1
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class BetterFashionClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout_conv = nn.Dropout2d(p=0.25)
        self.dropout_fc = nn.Dropout(p=0.5)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * 3 * 3, 512) 
        self.fc2 = nn.Linear(512, num_classes)

        self.res1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        x = x.view(x.size(0), -1)
        x = self.dropout_fc(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
    
        return x
