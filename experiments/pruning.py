import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import torch_pruning as tp



import torch 
from models.vgg import VGG
from benchmark import benchmark

checkpoint = torch.load("models/vgg.cifar.pretrained.pth", map_location="cpu")
model = VGG()
model.load_state_dict(checkpoint['state_dict'])

sample = torch.randn(1, 3, 32, 32)

benchmark(model, sample, plot=True)