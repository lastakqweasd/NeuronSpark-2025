import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

class FlowerDataset(Dataset):
    def __init__(self, image_paths, labels=None, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        if self.labels is not None:
            label = self.labels[idx]
            return image, label
        else:
            return image

# 加载标签数据
labels_df = pd.read_csv('NS-2025-00-data/train_labels.csv')

# 创建图像路径和标签的映射
train_image_dir = 'NS-2025-00-data/train_images/'
image_paths = [os.path.join(train_image_dir, fname) for fname in labels_df['file_name']]
labels = labels_df['label'].values

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(
    image_paths, labels, test_size=0.2, random_state=42, stratify=labels)

# 定义数据增强和预处理
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 创建数据集和数据加载器
train_dataset = FlowerDataset(X_train, y_train, train_transform)
val_dataset = FlowerDataset(X_val, y_val, val_transform)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 获取类别数量
num_classes = len(np.unique(labels))
print(f"Number of classes: {num_classes}")

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

class FlowerClassifier(nn.Module):
    def __init__(self, num_classes):
        super(FlowerClassifier, self).__init__()
        # 使用预训练的ResNet50
        self.base_model = models.resnet50(pretrained=True)
        
        # 冻结所有卷积层参数
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # 替换最后的全连接层
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        return self.base_model(x)

# 初始化模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = FlowerClassifier(num_classes).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 学习率调度器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.2)

# 早停机制
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

early_stopping = EarlyStopping(patience=5)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, early_stopping, num_epochs=30):
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        # 训练阶段
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        # 验证阶段
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data)
                
        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_epoch_acc = val_running_corrects.double() / len(val_loader.dataset)
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        print(f'Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}')
        
        # 更新学习率
        scheduler.step(val_epoch_loss)
        
        # 早停检查
        early_stopping(val_epoch_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
            
        # 保存最佳模型
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            torch.save(model.state_dict(), 'best_model.pth')

# 开始训练
train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, early_stopping)

# 加载最佳模型
model.load_state_dict(torch.load('best_model.pth'))

# 解冻部分层进行微调
for name, param in model.base_model.named_parameters():
    if 'layer4' in name or 'fc' in name:
        param.requires_grad = True

# 使用更小的学习率
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)

# 重新训练
early_stopping = EarlyStopping(patience=5)
train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, early_stopping, num_epochs=10)


# 加载测试图像
test_image_dir = 'NS-2025-00-data/test_images/'
test_images = [os.path.join(test_image_dir, fname) for fname in os.listdir(test_image_dir)]

# 创建测试数据集
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_dataset = FlowerDataset(test_images, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 进行预测
model.eval()
all_preds = []
image_names = os.listdir(test_image_dir)

with torch.no_grad():
    for inputs in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())

# 生成提交文件
submission = pd.DataFrame({
    'image': image_names,
    'label': all_preds
})
submission.to_csv('submission.csv', index=False)