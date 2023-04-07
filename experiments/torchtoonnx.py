import torch
from torch.onnx import TrainingMode


def torch_to_onnx(model, sample, path):
    torch.onnx.export(model, sample, path, training=TrainingMode.TRAINING, verbose=True)

# run netron model.onnx to see the model graph