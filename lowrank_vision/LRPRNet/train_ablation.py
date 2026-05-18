"""Train LRPR variants (full/baseline/no_l2/no_res) and report Top-1 + Top-5."""
import sys
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, '/gpool/home/wanghongyang/WangHY/LRPRNet/LRPR-MobileNetV2')

def get_model(variant, num_classes=10, width_mult=1.0, rank_ratio=0.125):
    if variant == 'no_l2':
        from models.lrprnet_no_l2 import LRPRNet_NoL2
        return LRPRNet_NoL2(num_classes=num_classes, width_mult=width_mult, rank_ratio=rank_ratio)
    elif variant == 'no_res':
        from models.lrprnet_no_res import LRPRNet_NoRes
        return LRPRNet_NoRes(num_classes=num_classes, width_mult=width_mult, rank_ratio=rank_ratio)
    elif variant == 'full':
        from models.lrprnet import LRPRNet
        return LRPRNet(num_classes=num_classes, width_mult=width_mult, rank_ratio=rank_ratio)
    elif variant == 'baseline':
        from models.mobilenetv2 import MobileNetV2
        return MobileNetV2(num_classes=num_classes, width_mult=width_mult)
    else:
        raise ValueError(f"Unknown variant: {variant}")

def accuracy(output, target, topk=(1, 5)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct1 = 0
    correct5 = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        correct1 += acc1.item() * inputs.size(0) / 100
        correct5 += acc5.item() * inputs.size(0) / 100
        total += inputs.size(0)
    return total_loss / total, 100 * correct1 / total, 100 * correct5 / total

def evaluate(model, loader, device):
    model.eval()
    correct1 = 0
    correct5 = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            correct1 += acc1.item() * inputs.size(0) / 100
            correct5 += acc5.item() * inputs.size(0) / 100
            total += inputs.size(0)
    return 100 * correct1 / total, 100 * correct5 / total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', type=str, required=True, choices=['no_l2', 'no_res', 'full', 'baseline'])
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training LRPR variant: {args.variant} on {device}")

    # Data
    data_root = '/gpool/home/wanghongyang/WangHY/LRPRNet/LRPR-MobileNetV2/data/imagewoof2-160'
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(160),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize(182),
        transforms.CenterCrop(160),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_dataset = datasets.ImageFolder(os.path.join(data_root, 'train'), train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_root, 'val'), val_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Model
    model = get_model(args.variant).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,} = {total_params/1e6:.2f}M")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=4e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_top1 = 0
    best_top5 = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_top1, train_top5 = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        val_top1, val_top5 = evaluate(model, val_loader, device)

        if val_top1 > best_top1:
            best_top1 = val_top1
            best_top5 = val_top5
            ckpt_path = f'/gpool/home/wanghongyang/WangHY/LRPRNet/LRPR-MobileNetV2/checkpoints/lrpr/lrpr_{args.variant}_best.pth'
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'best_top1': best_top1, 'best_top5': best_top5}, ckpt_path)

        if epoch % 50 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.4f} Top1: {train_top1:.2f}% Top5: {train_top5:.2f}% | Val Top1: {val_top1:.2f}% Top5: {val_top5:.2f}% | Best: {best_top1:.2f}%/{best_top5:.2f}%")

    print(f"\n{'='*50}")
    print(f"FINAL RESULT [{args.variant}]: Best Val Top-1: {best_top1:.2f}%, Best Val Top-5: {best_top5:.2f}%")
    print(f"Params: {total_params:,} = {total_params/1e6:.2f}M")
    print(f"{'='*50}")

if __name__ == '__main__':
    main()
