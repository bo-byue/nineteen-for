import sys
import torch
import torchvision

print("Python version:")
print(sys.version)

print("\nPyTorch version:")
print(torch.__version__)

print("\nTorchvision version:")
print(torchvision.__version__)

print("\nCUDA available:")
print(torch.cuda.is_available())