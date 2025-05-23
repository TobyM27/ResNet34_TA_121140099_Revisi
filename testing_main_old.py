import argparse
import itertools
import numpy as np
import torch
import random
from datareader import load_data, get_kfold_loaders
from model import get_model
from train import train_model
import os

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='results')
    args = parser.parse_args()

    set_seed(42)  # Seed for reproducibility

    # Hyperparameters
    lrs = [0.01, 0.001, 0.0001]
    epochs_list = [10, 15]
    batch_sizes = [16, 32]
    optimizers = ['adam', 'sgd']
    k_folds = 5

    full_dataset = load_data(args.data_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for lr, epochs, bs, optim_name in itertools.product(lrs, epochs_list, batch_sizes, optimizers):
        hp_str = f'lr_{lr}_epochs_{epochs}_bs_{bs}_optim_{optim_name}'
        hp_dir = os.path.join(args.output_dir, hp_str)
        os.makedirs(hp_dir, exist_ok=True)

        fold_loaders = get_kfold_loaders(full_dataset, bs, k=k_folds)
        fold_accs = []

        for fold, (train_loader, val_loader) in enumerate(fold_loaders):
            fold_dir = os.path.join(hp_dir, f'fold_{fold}')
            os.makedirs(fold_dir, exist_ok=True)

            model = get_model().to(device)
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr) if optim_name == 'adam' else torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

            best_acc = train_model(
                model, train_loader, val_loader, criterion, optimizer, 
                epochs, device, fold, fold_dir
            )
            fold_accs.append(best_acc)

        with open(os.path.join(hp_dir, 'summary.txt'), 'w') as f:
            f.write(f'Mean Validation Acc: {np.mean(fold_accs):.4f}\n')
            f.write(f'Std Validation Acc: {np.std(fold_accs):.4f}\n')

if __name__ == '__main__':
    main()