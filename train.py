"""Training entrypoint
"""
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from models.classification import VGG11Classifier
from data.pets_dataset import *
from models.localization import VGG11Localizer
from losses.iou_loss import *
import numpy as np
from tqdm import tqdm


def train(model, criterion, train_loader, val_loader, optimizer, num_epochs=50, device='cuda'):
    model = model.to(device)
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_bar = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{num_epochs}] Train')
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_total += labels.size(0)
            # _, predicted = torch.max(outputs, 1)
            # train_correct += (predicted == labels).sum().item()

            train_bar.set_postfix({
                'loss': f'{train_loss/train_total:.4f}',
                # 'acc': f'{train_correct/train_total:.4f}'
            })

        train_loss = train_loss / train_total
        # train_acc = train_correct / train_total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        val_bar = tqdm(val_loader, desc=f'Epoch [{epoch+1}/{num_epochs}] Val  ')
        with torch.no_grad():
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_total += labels.size(0)
                # _, predicted = torch.max(outputs, 1)
                # val_correct += (predicted == labels).sum().item()

                val_bar.set_postfix({
                    'loss': f'{val_loss/val_total:.4f}',
                    # 'acc': f'{val_correct/val_total:.4f}'
                })

        val_loss = val_loss / val_total
        # val_acc = val_correct / val_total

        # wandb.log({
        #     "epoch": epoch + 1,
        #     "train_loss": train_loss,
        #     "train_acc": train_acc,
        #     "val_loss": val_loss,
        #     "val_acc": val_acc
        # })

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        # history['train_acc'].append(train_acc)
        # history['val_acc'].append(val_acc)

    return history

def main():
    # wandb.init(project="oxford-pets-vgg11")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGG11Localizer(dropout_p=0.5)
    criterion = Localize_loss()

    with open('data/annotations/trainval.txt') as f:
        contents = [line for line in f.read().split('\n') if line.strip()]

    np.random.shuffle(contents)
    split_idx = int(len(contents)*0.8)
    train_data = contents[:split_idx]
    val_data = contents[split_idx:]
    
    train_dataset = OxfordIIITPetDataset_localize(train_data, Image_transform)
    val_dataset = OxfordIIITPetDataset_localize(val_data, None)

    train_dataloader = DataLoader(train_dataset, 16, True)
    val_dataloader = DataLoader(val_dataset, 16, False)

    # FIX: Lower learning rate for from-scratch training
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    history = train(model, criterion, train_dataloader, val_dataloader, optimizer, device=device)
    
    # wandb.finish()

if __name__=='__main__':
    main()