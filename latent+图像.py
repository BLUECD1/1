import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# 数据预处理（不包含标准化）
transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False)

# 特征提取器定义
class ResNetFeatureExtractor(nn.Module):
    def __init__(self, resnet_model):
        super(ResNetFeatureExtractor, self).__init__()
        # 修改 ResNet 的输入层以适应 32x32 的图像
        resnet_model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        # 去掉 ResNet 的全连接层
        self.resnet = nn.Sequential(*list(resnet_model.children())[:-2])

    def forward(self, x):
        x = self.resnet(x)
        return x

# 分类器定义
class CombinedClassifier(nn.Module):
    def __init__(self):
        super(CombinedClassifier, self).__init__()
        self.feature_size = 512*2*2   # 根据 ResNet 的输出特征图计算

        # 对于展平后的图像特征
        self.fc1 = nn.Linear(3 * 32 * 32 + self.feature_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, 10)  # CIFAR-10 有 10 个类别

    def forward(self, image, feature):
        image = image.view(image.size(0), -1)  # 展平图像
        feature = feature.view(feature.size(0), -1)  # 展平特征
        combined = torch.cat((image, feature), dim=1)  # 连接图像和特征
        x = torch.relu(self.fc1(combined))
        x = self.bn1(x)
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# 实例化模型和优化器

resnet = models.resnet18(pretrained=True)  # 使用预训练的 ResNet18
resnet_feature_extractor = ResNetFeatureExtractor(resnet)
combined_classifier = CombinedClassifier()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(combined_classifier.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

# 训练分类器
def train_classifier(data_loader, feature_extractor, model, criterion, optimizer, epochs=10):
    # 设置模型为训练模式
    feature_extractor.eval()  # 特征提取器通常不需要训练
    model.train()  # 分类器需要训练

    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in data_loader:
            features = feature_extractor(images)
            optimizer.zero_grad()
            outputs = model(images, features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # 累加损失
            running_loss += loss.item() * labels.size(0)
        # 计算每个epoch的平均损失
        epoch_loss = running_loss / len(data_loader.dataset)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}')
# 评估分类器
def evaluate_classifier(data_loader, feature_extractor, model):
    feature_extractor.eval()
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            features = feature_extractor(images)
            outputs = model(images, features)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy

# 训练分类器
train_classifier(train_loader, resnet_feature_extractor, combined_classifier, criterion, optimizer)

# 评估分类器
accuracy = evaluate_classifier(test_loader, resnet_feature_extractor, combined_classifier)
print(f'Accuracy: {accuracy:.4f}')

#%%
