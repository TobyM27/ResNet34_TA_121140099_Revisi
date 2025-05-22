import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from sklearn.metrics import confusion_matrix

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate_model(model, val_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return all_preds, all_labels

def save_metrics(fold_dir, all_labels, all_preds, class_names, train_losses, val_losses, train_accs, val_accs):
    # Save loss and accuracy plots
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.legend()
    plt.title('Accuracy Curves')
    
    plt.savefig(os.path.join(fold_dir, 'training_metrics.png'))
    plt.close()

    # Save classification report
    report = classification_report(all_labels, all_preds, target_names=class_names)
    with open(os.path.join(fold_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(fold_dir, 'confusion_matrix.png'))
    plt.close()

class_names = ['bercak_daun', 'daun_berkerut', 'daun_berputar', 'daun_menggulung', 'daun_menguning']

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, fold, save_dir):
    best_val_acc = 0.0
    best_model_wts = None
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(num_epochs):
        # Training phase
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validation phase
        val_loss, val_acc = validate_epoch(
            model, val_loader, criterion, device
        )

        # Update training statistics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = model.state_dict().copy()
            torch.save(best_model_wts, os.path.join(save_dir, f'best_fold{fold}.pth'))

        print(f'Fold {fold} Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}\n')

    # Final evaluation with best model
    model.load_state_dict(best_model_wts)
    all_preds, all_labels = evaluate_model(model, val_loader, device)
    
    print("\nDEBUGGING DATASET STRUCTURE:")
    print("Dataset structure:", type(train_loader.dataset))
    print("Available attributes:", dir(train_loader.dataset))
    print("Classes:", getattr(train_loader.dataset, 'classes', 'NOT FOUND'))

    # Get class names from dataset
    class_names = getattr(train_loader.dataset, 'classes', ['bercak_daun', 'daun_berkerut', 'daun_berputar', 'daun_menggulung', 'daun_menguning'])
    
    # Save all metrics and visualizations
    save_metrics(
        save_dir, all_labels, all_preds, class_names,
        train_losses, val_losses, train_accs, val_accs
    )

    return best_val_acc