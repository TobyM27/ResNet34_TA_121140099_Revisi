import argparse
import numpy as np
import torch
import random
import os
import gc
from datareader import load_data, get_kfold_loaders
from model import get_model
from train import train_model

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='results')
    args = parser.parse_args()

    set_seed(42)  # Seed for reproducibility
    
    # Device setup for Apple Silicon
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    full_dataset = load_data(args.data_dir)

    # Manual hyperparameter combinations (edit this list between runs)
    manual_combinations = [
        # Start with small batch sizes first
        {'lr': 0.001, 'epochs': 15, 'batch_size': 16, 'optimizer': 'adam'},
        {'lr': 0.001, 'epochs': 15, 'batch_size': 16, 'optimizer': 'sgd'},
        # Add more combinations below this line
    ]

    # Resume tracking setup
    resume_file = os.path.join(args.output_dir, 'completed_combinations.txt')
    completed = set()
    if os.path.exists(resume_file):
        with open(resume_file, 'r') as f:
            completed = set(f.read().splitlines())

    for hp in manual_combinations:
        hp_str = f"lr_{hp['lr']}_epochs_{hp['epochs']}_bs_{hp['batch_size']}_optim_{hp['optimizer']}"
        
        if hp_str in completed:
            print(f"⏩ Skipping completed: {hp_str}")
            continue
            
        hp_dir = os.path.join(args.output_dir, hp_str)
        os.makedirs(hp_dir, exist_ok=True)
        
        try:
            print(f"\n{'='*40}")
            print(f"🚀 Starting: {hp_str}")
            print(f"{'='*40}")
            
            # Get data loaders with MPS-friendly settings
            fold_loaders = get_kfold_loaders(
                full_dataset, 
                batch_size=hp['batch_size'],
                num_workers=0  
            )
            fold_accs = []

            for fold, (train_loader, val_loader) in enumerate(fold_loaders):
                fold_dir = os.path.join(hp_dir, f'fold_{fold}')
                os.makedirs(fold_dir, exist_ok=True)

                # Create fresh model
                model = get_model().to(device)
                criterion = torch.nn.CrossEntropyLoss()
                
                # Optimizer setup
                if hp['optimizer'] == 'adam':
                    optimizer = torch.optim.Adam(model.parameters(), lr=hp['lr'])
                else:
                    optimizer = torch.optim.SGD(model.parameters(), 
                                              lr=hp['lr'], 
                                              momentum=0.9)

                # Train with MPS optimizations
                best_acc = train_model(
                    model, train_loader, val_loader,
                    criterion, optimizer, hp['epochs'],
                    device, fold, fold_dir
                )
                fold_accs.append(best_acc)

                # MPS memory cleanup
                del model, optimizer, criterion
                gc.collect()  # Force Python GC
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()  # MPS-specific cache clearing

            # Save results
            with open(os.path.join(hp_dir, 'summary.txt'), 'w') as f:
                f.write(f'Mean Validation Acc: {np.mean(fold_accs):.4f}\n')
                f.write(f'Std Validation Acc: {np.std(fold_accs):.4f}\n')

            # Mark as completed
            with open(resume_file, 'a') as f:
                f.write(f"{hp_str}\n")

            print(f"✅ Completed: {hp_str}")

        except Exception as e:
            print(f"❌ Error in {hp_str}: {str(e)}")
            with open(os.path.join(args.output_dir, 'errors.log'), 'a') as f:
                f.write(f"Failed: {hp_str} - {str(e)}\n")

if __name__ == '__main__':
    main()