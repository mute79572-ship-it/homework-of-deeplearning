# mnist_advanced.py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

# 1. 数据准备
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))
])

train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 分割验证集
train_size = int(0.8 * len(train_set))
val_size = len(train_set) - train_size
train_set, val_set = torch.utils.data.random_split(train_set, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=1000)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1000)

# 2. 定义三个模型用于对比
class SimpleCNN(nn.Module):
    """简单CNN模型"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(16*7*7, 64)
        self.fc2 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 16*7*7)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

class MLP(nn.Module):
    """多层感知机"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

class DeeperCNN(nn.Module):
    """更深层的CNN"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64*3*3, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(-1, 64*3*3)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# 3. 训练与评估函数
def train_model(model, model_name, optimizer, criterion, num_epochs=10):
    train_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        epoch_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # 验证阶段
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = correct / total
        val_accuracies.append(val_acc)
        print(f"{model_name} - Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    return train_losses, val_accuracies

def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())
    
    accuracy = correct / total
    return accuracy, all_preds, all_labels

# 4. 主实验：对比三个模型
models = {
    'MLP': MLP(),
    'SimpleCNN': SimpleCNN(),
    'DeeperCNN': DeeperCNN()
}

results = {}
for name, model in models.items():
    print(f"\n=== 训练 {name} 模型 ===")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    train_losses, val_accuracies = train_model(model, name, optimizer, criterion, num_epochs=10)
    test_acc, preds, labels = evaluate_model(model, test_loader)
    
    results[name] = {
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'test_accuracy': test_acc,
        'predictions': preds,
        'labels': labels
    }
    print(f"{name} 测试准确率: {test_acc:.4f}")

# 5. 可视化结果
# 5.1 训练损失对比
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
for name, result in results.items():
    plt.plot(result['train_losses'], label=name)
plt.title('训练损失对比')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# 5.2 验证准确率对比
plt.subplot(1, 3, 2)
for name, result in results.items():
    plt.plot(result['val_accuracies'], label=name)
plt.title('验证准确率对比')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

# 5.3 测试准确率对比
plt.subplot(1, 3, 3)
accuracies = [results[name]['test_accuracy'] for name in models.keys()]
bars = plt.bar(models.keys(), accuracies, color=['skyblue', 'lightgreen', 'salmon'])
plt.title('测试准确率对比')
plt.ylabel('Accuracy')
plt.ylim(0.9, 1.0)
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.005, 
             f'{acc:.3f}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')

# 5.4 混淆矩阵
best_model_name = max(results, key=lambda x: results[x]['test_accuracy'])
cm = confusion_matrix(results[best_model_name]['labels'], 
                      results[best_model_name]['predictions'])

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=range(10), yticklabels=range(10))
plt.title(f'混淆矩阵 - {best_model_name} (准确率: {results[best_model_name]["test_accuracy"]:.3f})')
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')

# 5.5 样本预测展示
best_model = models[best_model_name]
images, labels = next(iter(test_loader))
with torch.no_grad():
    outputs = best_model(images[:10])
_, preds = torch.max(outputs, 1)

plt.figure(figsize=(15, 3))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(images[i][0], cmap='gray')
    plt.title(f"Pred: {preds[i].item()}\nTrue: {labels[i].item()}", fontsize=8)
    plt.axis('off')

plt.suptitle(f'样本预测结果 ({best_model_name})', y=1.05)
plt.tight_layout()
plt.savefig('sample_predictions.png', dpi=150, bbox_inches='tight')

# 6. 打印结果表格
print("\n" + "="*50)
print("模型对比结果总结：")
print("="*50)
print(f"{'模型':<15} {'参数量':<15} {'测试准确率':<15}")
print("-"*50)

for name in models.keys():
    total_params = sum(p.numel() for p in models[name].parameters())
    print(f"{name:<15} {total_params:<15,} {results[name]['test_accuracy']:<15.3%}")

print("="*50)
print(f"最佳模型: {best_model_name} (准确率: {results[best_model_name]['test_accuracy']:.3%})")
