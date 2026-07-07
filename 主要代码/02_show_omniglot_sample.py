from torchvision.datasets import Omniglot
from torchvision import transforms
import matplotlib.pyplot as plt
import os

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

img, label = dataset[0]

print("图片形状：", img.shape)
print("图片标签：", label)

plt.figure(figsize=(4, 4))
plt.imshow(img.squeeze(), cmap="gray")
plt.title(f"Omniglot Sample, Label: {label}")
plt.axis("off")
plt.savefig("outputs/02_omniglot_single_sample.png", dpi=200, bbox_inches="tight")
plt.show()

print("图片已保存：outputs/02_omniglot_single_sample.png")