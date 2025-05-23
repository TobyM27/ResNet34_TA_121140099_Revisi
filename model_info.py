import torch
import os
import time
from torchvision import models
from torchinfo import summary
import argparse
from collections import OrderedDict
import torch.nn as nn
import thop  # Requires installation: pip install thop

def load_model(model_path):
    """Load a pre-trained ResNet-34 model from a .pth file."""
    # Initialize the model architecture
    model = models.resnet34()

    # Freeze all layers except the last one
    for param in model.parameters():
        param.requires_grad = False

    # Modify the final layer
    model.fc = nn.Linear(model.fc.in_features, 5)
    for param in model.fc.parameters():
        param.requires_grad = True  # Fixed typo here

    # Load the saved weights
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Handle DataParallel wrapping
    if 'module.' in list(state_dict.keys())[0]:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        state_dict = new_state_dict
    
    model.load_state_dict(state_dict)
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser(description='Model analysis for ResNet-34')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to the .pth model file')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for analysis')
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.model_path)
    print(f"Model loaded from {args.model_path}")

    # Model summary
    summary(model, 
           input_size=(args.batch_size, 3, 224, 224),
           col_names=["input_size", "output_size", "num_params", "trainable"],
           col_width=20,
           row_settings=["var_names"])

    # Additional metrics
    input_tensor = torch.randn(args.batch_size, 3, 224, 224)
    
    # MACs calculation
    macs, params = thop.profile(model, inputs=(input_tensor,), verbose=False)
    print(f"\n{'='*40}\nAdditional Metrics:")
    print(f"MACs: {macs/1e9:.2f} GMac")
    
    # Model size
    model_size = os.path.getsize(args.model_path) / (1024**2)  # MB
    print(f"Model Size: {model_size:.2f} MB")
    
    # Inference time measurement
    dummy_input = torch.randn(args.batch_size, 3, 224, 224)
    
    # Warm-up
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
    
    # Timing
    iterations = 100
    start_time = time.time()
    for _ in range(iterations):
        with torch.no_grad():
            _ = model(dummy_input)
    elapsed = time.time() - start_time
    
    print(f"Inference Time: {elapsed/iterations*1000:.2f} ms/batch")
    print(f"FPS: {iterations/elapsed:.2f} frames/second")
    print(f"{'='*40}")

if __name__ == "__main__":
    main()