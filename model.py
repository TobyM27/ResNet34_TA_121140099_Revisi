import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet34_Weights

def get_model(num_classes=5):
    """
    Load a pre-trained ResNet34 model and modify the final layer for the number of classes.
    """
    model = models.resnet34(weights=ResNet34_Weights.DEFAULT)

    # Membekukan semua layer kecuali layer terakhir
    for param in model.parameters():
        param.requires_grad = False

    # Mengubah layer terakhir
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    for param in model.fc.parameters():
        param.required_grad = True
    return model