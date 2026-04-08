"""Training entrypoint
"""
import os
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from losses.iou_loss import Localize_loss, IoULoss
from models.classification import VGG11Classifier
from models.localization   import VGG11Localizer
from models.segmentation   import VGG11UNet

from data.pets_dataset import OxfordIIITPetDataset_classify, OxfordIIITPetDataset_localize, OxfordIIITPetDataset_Segmentation, Image_transform



def dice_score(pred_logits: torch.Tensor, target: torch.Tensor,
               num_classes: int = 3, eps: float = 1e-6) -> float:
    pred = pred_logits.argmax(dim=1)
    dice = 0.0
    for c in range(num_classes):
        pred_c   = (pred == c).float()
        target_c = (target == c).float()
        intersection = (pred_c * target_c).sum()
        dice += (2 * intersection + eps) / (pred_c.sum() + target_c.sum() + eps)
    return (dice / num_classes).item()


def pixel_accuracy(pred_logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = pred_logits.argmax(dim=1)
    return (pred == target).sum().item() / target.numel()


def macro_f1(preds: torch.Tensor, targets: torch.Tensor, num_classes: int = 37) -> float:
    f1 = 0.0
    for c in range(num_classes):
        tp = ((preds == c) & (targets == c)).sum().item()
        fp = ((preds == c) & (targets != c)).sum().item()
        fn = ((preds != c) & (targets == c)).sum().item()
        precision = tp / (tp + fp + 1e-6)
        recall    = tp / (tp + fn + 1e-6)
        f1 += 2 * precision * recall / (precision + recall + 1e-6)
    return f1 / num_classes


def read_annotations(ann_file: str) -> list:
    with open(ann_file) as f:
        lines = [l.strip() for l in f if not l.startswith('#') and l.strip()]
    return lines


def get_dataloaders(args):
    lines = read_annotations(args.ann_file)
    random.shuffle(lines)
    n_val   = int(len(lines) * args.val_split)
    train_l = lines[n_val:]
    val_l   = lines[:n_val]

    if args.task == 'classification':
        train_ds = OxfordIIITPetDataset_classify(train_l, transform=Image_transform)
        val_ds   = OxfordIIITPetDataset_classify(val_l)
    elif args.task == 'localization':
        train_ds = OxfordIIITPetDataset_localize(train_l, transform=Image_transform)
        val_ds   = OxfordIIITPetDataset_localize(val_l)
    elif args.task == 'segmentation':
        train_ds = OxfordIIITPetDataset_Segmentation(train_l, transform=Image_transform)
        val_ds   = OxfordIIITPetDataset_Segmentation(val_l)
    else:
        raise ValueError(f"Unknown task: {args.task}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers, pin_memory=True)
    return train_loader, val_loader


def load_encoder_weights(model, encoder_init, device):
    ckpt = torch.load(encoder_init, map_location=device)
    encoder_weights = {
        k.replace('encoder.', ''): v
        for k, v in ckpt.items() if k.startswith('encoder.')
    }
    model.encoder.load_state_dict(encoder_weights, strict=False)
    print(f"Loaded encoder weights from {encoder_init}")


def get_model(args, device):
    if args.task == 'classification':
        model = VGG11Classifier(num_classes=37, dropout_p=args.dropout_p)

    elif args.task == 'localization':
        model = VGG11Localizer(dropout_p=args.dropout_p)
        if args.encoder_init and os.path.exists(args.encoder_init):
            load_encoder_weights(model, args.encoder_init, device)

    elif args.task == 'segmentation':
        model = VGG11UNet(num_classes=3, dropout_p=args.dropout_p)
        if args.encoder_init and os.path.exists(args.encoder_init):
            load_encoder_weights(model, args.encoder_init, device)

        if args.freeze_strategy == 'full_freeze':
            for p in model.encoder.parameters():
                p.requires_grad = False
            print("Encoder fully frozen.")

        elif args.freeze_strategy == 'partial':
            for name, p in model.encoder.named_parameters():
                p.requires_grad = any(b in name for b in ['block4', 'block5'])
            print("Encoder partially frozen (block4, block5 trainable).")

        else:
            print("Full fine-tuning.")

    return model.to(device)


def train_epoch_classification(model, loader, optimizer, criterion, device, epoch, global_step):
    model.train()
    total_loss, all_preds, all_targets = 0.0, [], []

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        total_loss += batch_loss * imgs.size(0)
        all_preds.append(logits.argmax(1).cpu())
        all_targets.append(labels.cpu())

        batch_acc = (logits.argmax(1) == labels).float().mean().item()
        print(f"  [Epoch {epoch}][Batch {batch_idx+1}/{len(loader)}] loss: {batch_loss:.4f} | acc: {batch_acc:.4f}", end="\r")
        wandb.log({'batch/train_loss': batch_loss, 'batch/train_acc': batch_acc, 'global_step': global_step})
        global_step += 1

    all_preds   = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, (all_preds == all_targets).float().mean().item(), macro_f1(all_preds, all_targets), global_step


@torch.no_grad()
def val_epoch_classification(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_targets = 0.0, [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * imgs.size(0)
        all_preds.append(logits.argmax(1).cpu())
        all_targets.append(labels.cpu())

    all_preds   = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, (all_preds == all_targets).float().mean().item(), macro_f1(all_preds, all_targets)


def train_epoch_localization(model, loader, optimizer, criterion, device, epoch, global_step, iou_fn=IoULoss()):
    model.train()
    total_loss, total_iou = 0.0, 0.0

    for batch_idx, (imgs, bboxes) in enumerate(loader):
        imgs, bboxes = imgs.to(device), bboxes.to(device).float()
        optimizer.zero_grad()
        preds = model(imgs)
        loss  = criterion(preds,bboxes)
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        total_loss += batch_loss * imgs.size(0)
        with torch.no_grad():
            batch_iou = (1.0 - iou_fn(preds.detach(), bboxes)).item()
            total_iou += batch_iou * imgs.size(0)

        print(f"  [Epoch {epoch}][Batch {batch_idx+1}/{len(loader)}] loss: {batch_loss:.4f} | iou: {batch_iou:.4f}", end="\r")
        wandb.log({'batch/train_loss': batch_loss, 'batch/train_iou': batch_iou, 'global_step': global_step})
        global_step += 1

    n = len(loader.dataset)
    return total_loss / n, total_iou / n, global_step


@torch.no_grad()
def val_epoch_localization(model, loader, criterion, device, iou_fn = IoULoss()):
    model.eval()
    total_loss, total_iou = 0.0, 0.0

    for imgs, bboxes in loader:
        imgs, bboxes = imgs.to(device), bboxes.to(device).float()
        preds    = model(imgs)
        loss     = criterion(preds, bboxes)

        total_loss += loss.item() * imgs.size(0)
        total_iou  += (1.0 - iou_fn(preds, bboxes)).item() * imgs.size(0)

    n = len(loader.dataset)
    return total_loss / n, total_iou / n


def train_epoch_segmentation(model, loader, optimizer, criterion, device, epoch, global_step):
    model.train()
    total_loss, total_dice, total_acc = 0.0, 0.0, 0.0

    for batch_idx, (imgs, masks) in enumerate(loader):
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        total_loss += batch_loss * imgs.size(0)
        with torch.no_grad():
            batch_dice = dice_score(logits, masks)
            batch_acc  = pixel_accuracy(logits, masks)
            total_dice += batch_dice * imgs.size(0)
            total_acc  += batch_acc  * imgs.size(0)

        print(f"  [Epoch {epoch}][Batch {batch_idx+1}/{len(loader)}] loss: {batch_loss:.4f} | dice: {batch_dice:.4f} | acc: {batch_acc:.4f}", end="\r")
        wandb.log({'batch/train_loss': batch_loss, 'batch/train_dice': batch_dice, 'batch/train_acc': batch_acc, 'global_step': global_step})
        global_step += 1

    n = len(loader.dataset)
    return total_loss / n, total_dice / n, total_acc / n, global_step


@torch.no_grad()
def val_epoch_segmentation(model, loader, criterion, device):
    model.eval()
    total_loss, total_dice, total_acc = 0.0, 0.0, 0.0

    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        loss   = criterion(logits, masks)

        total_loss += loss.item() * imgs.size(0)
        total_dice += dice_score(logits, masks) * imgs.size(0)
        total_acc  += pixel_accuracy(logits, masks) * imgs.size(0)

    n = len(loader.dataset)
    return total_loss / n, total_dice / n, total_acc / n


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | Task: {args.task}")
    iou_fn = IoULoss()
    wandb.init(
        project = args.wandb_project,
        name    = args.run_name or f"{args.task}_lr{args.lr}_bs{args.batch_size}",
        config  = vars(args),
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    train_loader, val_loader = get_dataloaders(args)
    model = get_model(args, device)
    wandb.watch(model, log='gradients', log_freq=100)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5
    )

    classify_loss  = nn.CrossEntropyLoss()
    localize_loss = Localize_loss()
    segment_loss = nn.CrossEntropyLoss(ignore_index=255)

    best_val_loss = float('inf')
    global_step   = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        if args.task == 'classification':
            tr_loss, tr_acc, tr_f1, global_step = train_epoch_classification(model, train_loader, optimizer, classify_loss, device, epoch, global_step)
            vl_loss, vl_acc, vl_f1 = val_epoch_classification(model, val_loader, classify_loss, device)

            print(f"  Train Loss: {tr_loss:.4f} | Acc: {tr_acc:.4f} | F1: {tr_f1:.4f}")
            print(f"  Val   Loss: {vl_loss:.4f} | Acc: {vl_acc:.4f} | F1: {vl_f1:.4f}")

            wandb.log({
                'epoch': epoch, 'lr': optimizer.param_groups[0]['lr'],
                'train/loss': tr_loss, 'train/accuracy': tr_acc, 'train/macro_f1': tr_f1,
                'val/loss':   vl_loss, 'val/accuracy':   vl_acc, 'val/macro_f1':   vl_f1,
            })

        elif args.task == 'localization':
            tr_loss, tr_iou, global_step = train_epoch_localization(model, train_loader, optimizer, localize_loss, device, epoch, global_step, iou_fn)
            vl_loss, vl_iou = val_epoch_localization(model, val_loader, localize_loss, device, iou_fn)

            print(f"  Train Loss: {tr_loss:.4f} | IoU: {tr_iou:.4f}")
            print(f"  Val   Loss: {vl_loss:.4f} | IoU: {vl_iou:.4f}")

            wandb.log({
                'epoch': epoch, 'lr': optimizer.param_groups[0]['lr'],
                'train/loss': tr_loss, 'train/iou': tr_iou,
                'val/loss':   vl_loss, 'val/iou':   vl_iou,
            })

        elif args.task == 'segmentation':
            tr_loss, tr_dice, tr_acc, global_step = train_epoch_segmentation(model, train_loader, optimizer, segment_loss, device, epoch, global_step)
            vl_loss, vl_dice, vl_acc = val_epoch_segmentation(model, val_loader, segment_loss, device)

            print(f"  Train Loss: {tr_loss:.4f} | Dice: {tr_dice:.4f} | Acc: {tr_acc:.4f}")
            print(f"  Val   Loss: {vl_loss:.4f} | Dice: {vl_dice:.4f} | Acc: {vl_acc:.4f}")

            wandb.log({
                'epoch': epoch, 'lr': optimizer.param_groups[0]['lr'],
                'train/loss': tr_loss, 'train/dice': tr_dice, 'train/pixel_acc': tr_acc,
                'val/loss':   vl_loss, 'val/dice':   vl_dice, 'val/pixel_acc':   vl_acc,
            })

        scheduler.step(vl_loss)

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            ckpt_path = os.path.join(args.checkpoint_dir, f"best_{args.task}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Saved best checkpoint -> {ckpt_path}")
            wandb.save(ckpt_path)

    wandb.finish()
    print("\nTraining complete.")


def parse_args():
    p = argparse.ArgumentParser(description="DA6401 Assignment 2 Training Script")

    p.add_argument('--task', type=str, required=True,
                   choices=['classification', 'localization', 'segmentation'])
    p.add_argument('--ann_file',        type=str,   default='data/annotations/trainval.txt')
    p.add_argument('--val_split',       type=float, default=0.2)
    p.add_argument('--epochs',          type=int,   default=50)
    p.add_argument('--batch_size',      type=int,   default=24)
    p.add_argument('--lr',              type=float, default=1e-4)
    p.add_argument('--weight_decay',    type=float, default=1e-4)
    p.add_argument('--dropout_p',       type=float, default=0.5)
    p.add_argument('--num_workers',     type=int,   default=4)
    p.add_argument('--encoder_init',    type=str,   default='checkpoints/best_classification.pth')
    p.add_argument('--freeze_strategy', type=str,   default='full_finetune',
                   choices=['full_freeze', 'partial', 'full_finetune'])
    p.add_argument('--checkpoint_dir',  type=str,   default='checkpoints')
    p.add_argument('--wandb_project',   type=str,   default='da6401_assignment2')
    p.add_argument('--run_name',        type=str,   default='')

    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)