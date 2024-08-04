#准确率0.5560
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset

# 数据预处理
transform = transforms.Compose([
    transforms.Resize(128),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])  # 标准化
])


train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# 限制数据集大小
def reduce_dataset(dataset, fraction=0.1):
    num_samples = len(dataset)
    indices = list(range(num_samples))
    split = int(num_samples * fraction)
    subset_indices = indices[:split]
    return Subset(dataset, subset_indices)

train_dataset = reduce_dataset(train_dataset, fraction=0.1)
test_dataset = reduce_dataset(test_dataset, fraction=0.1)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=16,
    shuffle=True
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=16,
    shuffle=False
)

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 10)

    def forward(self, x):
        return self.resnet(x)

# 初始化分类器
classifier = Classifier()

# 设置损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=0.001)

def train_classifier(data_loader):
    classifier.train()
    for i, (images, labels) in enumerate(data_loader):
        optimizer.zero_grad()
        outputs = classifier(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

def evaluate(data_loader):
    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = classifier(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy

# 训练分类器
train_classifier(train_loader)

# 评估准确率
accuracy = evaluate(test_loader)
print(f'Accuracy: {accuracy:.4f}')

