import random
from collections import defaultdict
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import Omniglot
from torchvision import transforms
import matplotlib.pyplot as plt

# =========================
# 1. 参数设置
# =========================

N_WAY = 5
N_SHOT = 5
N_QUERY = 15

TRAIN_EPISODES = 300
TEST_EPISODES = 100
PRINT_EVERY = 50

LEARNING_RATE = 0.001

os.makedirs("outputs", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("当前使用设备：", device)

# =========================
# 2. 数据预处理
# =========================

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

# =========================
# 3. 加载数据集
# =========================

train_dataset = Omniglot(
    root="./data",
    background=True,
    transform=transform,
    download=True
)

test_dataset = Omniglot(
    root="./data",
    background=False,
    transform=transform,
    download=True
)

print("训练集大小：", len(train_dataset))
print("测试集大小：", len(test_dataset))

# =========================
# 4. 按类别整理图片
# =========================

def build_class_to_indices(dataset):
    class_to_indices = defaultdict(list)

    for index, item in enumerate(dataset._flat_character_images):
        image_path, label = item
        class_to_indices[label].append(index)

    return class_to_indices

train_class_to_indices = build_class_to_indices(train_dataset)
test_class_to_indices = build_class_to_indices(test_dataset)

print("训练类别数量：", len(train_class_to_indices))
print("测试类别数量：", len(test_class_to_indices))

# =========================
# 5. 采样一个 5-way 5-shot episode
# =========================

def sample_episode(dataset, class_to_indices):
    selected_classes = random.sample(list(class_to_indices.keys()), N_WAY)

    support_images = []
    support_labels = []
    query_images = []
    query_labels = []

    for new_label, old_label in enumerate(selected_classes):
        indices = class_to_indices[old_label]
        selected_indices = random.sample(indices, N_SHOT + N_QUERY)

        support_indices = selected_indices[:N_SHOT]
        query_indices = selected_indices[N_SHOT:]

        for idx in support_indices:
            img, _ = dataset[idx]
            support_images.append(img)
            support_labels.append(new_label)

        for idx in query_indices:
            img, _ = dataset[idx]
            query_images.append(img)
            query_labels.append(new_label)

    support_images = torch.stack(support_images).to(device)
    support_labels = torch.tensor(support_labels).to(device)
    query_images = torch.stack(query_images).to(device)
    query_labels = torch.tensor(query_labels).to(device)

    return support_images, support_labels, query_images, query_labels

# =========================
# 6. CNN 特征提取器
# =========================

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.block(x)

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            ConvBlock(1, 64),
            ConvBlock(64, 64),
            ConvBlock(64, 64),
            ConvBlock(64, 64)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return x

# =========================
# 7. 原型网络损失函数
# =========================

def prototypical_loss(model, support_images, query_images, query_labels):
    support_features = model(support_images)
    query_features = model(query_images)

    support_features = support_features.view(N_WAY, N_SHOT, -1)

    prototypes = support_features.mean(dim=1)

    distances = torch.cdist(query_features, prototypes)

    scores = -distances

    loss = F.cross_entropy(scores, query_labels)

    predictions = scores.argmax(dim=1)
    accuracy = (predictions == query_labels).float().mean().item()

    return loss, accuracy

# =========================
# 8. 模型训练
# =========================

model = FeatureExtractor().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_losses = []
train_accuracies = []

print("\n开始训练...")

for episode in range(1, TRAIN_EPISODES + 1):
    model.train()

    support_images, support_labels, query_images, query_labels = sample_episode(
        train_dataset,
        train_class_to_indices
    )

    loss, accuracy = prototypical_loss(
        model,
        support_images,
        query_images,
        query_labels
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())
    train_accuracies.append(accuracy)

    if episode % PRINT_EVERY == 0:
        print(
            f"Episode {episode}/{TRAIN_EPISODES} | "
            f"Loss: {loss.item():.4f} | "
            f"Accuracy: {accuracy * 100:.2f}%"
        )

# =========================
# 9. 模型测试
# =========================

print("\n开始测试...")

model.eval()
test_accuracies = []

with torch.no_grad():
    for episode in range(TEST_EPISODES):
        support_images, support_labels, query_images, query_labels = sample_episode(
            test_dataset,
            test_class_to_indices
        )

        loss, accuracy = prototypical_loss(
            model,
            support_images,
            query_images,
            query_labels
        )

        test_accuracies.append(accuracy)

average_accuracy = sum(test_accuracies) / len(test_accuracies)

print("\n==============================")
print("5-way 5-shot 测试结果")
print(f"平均准确率：{average_accuracy * 100:.2f}%")
print("==============================")

with open("outputs/08_final_accuracy_result.txt", "w", encoding="utf-8") as f:
    f.write("5-way 5-shot 测试结果\n")
    f.write(f"平均准确率：{average_accuracy * 100:.2f}%\n")

# =========================
# 10. 保存训练曲线
# =========================

plt.figure()
plt.plot(train_losses)
plt.xlabel("Episode")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.savefig("outputs/06_training_loss_curve.png", dpi=200, bbox_inches="tight")
plt.show()

plt.figure()
plt.plot([acc * 100 for acc in train_accuracies])
plt.xlabel("Episode")
plt.ylabel("Accuracy (%)")
plt.title("Training Accuracy")
plt.savefig("outputs/07_training_accuracy_curve.png", dpi=200, bbox_inches="tight")
plt.show()

print("Loss 曲线已保存：outputs/06_training_loss_curve.png")
print("Accuracy 曲线已保存：outputs/07_training_accuracy_curve.png")
print("最终准确率文本已保存：outputs/08_final_accuracy_result.txt")