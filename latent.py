#准确率0.80
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset

# 数据预处理
transform = transforms.Compose([
    transforms.Resize(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),  # 标准化处理
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

train_dataset = reduce_dataset(train_dataset)
test_dataset = reduce_dataset(test_dataset)

train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False)

# 特征提取器定义
class ResNetFeatureExtractor(nn.Module):
    def __init__(self, resnet_model):
        super(ResNetFeatureExtractor, self).__init__()
        self.resnet = nn.Sequential(*list(resnet_model.children())[:-2])  # 去掉 ResNet 的全连接层
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.resnet(x)
        x = self.flatten(x)
        return x

# 分类器定义
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(512 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 实例化模型和优化器
resnet = models.resnet18(pretrained=True)  # 使用预训练的 ResNet18
resnet_feature_extractor = ResNetFeatureExtractor(resnet)
classifier = Classifier()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=0.001)

# 训练分类器
def train_classifier(data_loader, feature_extractor, model, criterion, optimizer):
    feature_extractor.eval()
    model.train()

    running_loss = 0.0
    for images, labels in data_loader:
        features = feature_extractor(images)  # 提取特征
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * labels.size(0)
    epoch_loss = running_loss / len(data_loader.dataset)
    print(f'分类器： {epoch_loss}')

# 评估分类器
def evaluate_classifier(data_loader, feature_extractor, model):
    feature_extractor.eval()
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            features = feature_extractor(images)
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy

# 训练分类器
train_classifier(train_loader, resnet_feature_extractor, classifier, criterion, optimizer)

# 评估分类器
accuracy = evaluate_classifier(test_loader, resnet_feature_extractor, classifier)
print(f'Accuracy: {accuracy:.4f}')
