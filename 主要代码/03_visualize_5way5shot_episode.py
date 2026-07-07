import random
import torch
from torchvision.datasets import Omniglot
from torchvision import transforms
import matplotlib.pyplot as plt
from collections import defaultdict
import os

N_WAY = 5
N_SHOT = 5
N_QUERY = 15

os.makedirs("outputs", exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

dataset = Omniglot(
    root="./data",
    background=True,
    transform=transform,
    download=True
)

print("数据集大小：", len(dataset))

class_to_indices = defaultdict(list)

for index, item in enumerate(dataset._flat_character_images):
    image_path, label = item
    class_to_indices[label].append(index)

print("类别数量：", len(class_to_indices))

def sample_episode():
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

    support_images = torch.stack(support_images)
    support_labels = torch.tensor(support_labels)
    query_images = torch.stack(query_images)
    query_labels = torch.tensor(query_labels)

    return support_images, support_labels, query_images, query_labels

support_images, support_labels, query_images, query_labels = sample_episode()

print("support_images 形状：", support_images.shape)
print("support_labels 形状：", support_labels.shape)
print("query_images 形状：", query_images.shape)
print("query_labels 形状：", query_labels.shape)

plt.figure(figsize=(8, 5))

for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(support_images[i].squeeze(), cmap="gray")
    plt.title(f"Class {support_labels[i].item()}")
    plt.axis("off")

plt.suptitle("5-way 5-shot Support Set")
plt.savefig("outputs/03_episode_support_set_5way5shot.png", dpi=200, bbox_inches="tight")
plt.show()

plt.figure(figsize=(10, 6))

for i in range(30):
    plt.subplot(5, 6, i + 1)
    plt.imshow(query_images[i].squeeze(), cmap="gray")
    plt.title(f"Class {query_labels[i].item()}")
    plt.axis("off")

plt.suptitle("Query Set Samples")
plt.savefig("outputs/04_episode_query_set_samples.png", dpi=200, bbox_inches="tight")
plt.show()

print("Support Set 图片已保存：outputs/03_episode_support_set_5way5shot.png")
print("Query Set 图片已保存：outputs/04_episode_query_set_samples.png")