"""
Training script for FIS-enhanced model
Compatible with Channel + utils.py
Support CIFAR10 and ImageNet (Vanilla dataset)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import argparse
import os
from datetime import datetime

# from model import JSCC_FIS  # (old)
from model_baseline import DeepJSCC_FIS
from channel import Channel
from dataset import Vanilla
from utils import get_psnr, set_seed


# =========================================
# Train 1 epoch
# =========================================
def train_one_epoch(model, train_loader, channel, optimizer, criterion, epoch, args, writer):
    model.train()

    total_loss = 0
    total_psnr = 0
    total_alloc = 0
    total_samples = 0

    for batch_idx, (images, _) in enumerate(train_loader):
        images = images.cuda()

        optimizer.zero_grad()

        # Forward
        encoded, decoded, info = model(
            images,
            target_rate=args.target_rate,
            return_info=True
        )

        # Channel
        encoded_noisy = channel(encoded)

        # Decode
        decoded_noisy = model.decoder(encoded_noisy)

        # Loss
        loss = criterion(decoded_noisy, images)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Metrics
        batch_size = images.size(0)
        psnr = get_psnr(decoded_noisy * 255.0, images * 255.0)
        avg_A = info["A_stats"]["A_mean"]

        total_loss += loss.item() * batch_size
        total_psnr += psnr.item() * batch_size
        total_alloc += avg_A * batch_size
        total_samples += batch_size

        if batch_idx % args.log_interval == 0:
            print(
                f'Epoch [{epoch}][{batch_idx}/{len(train_loader)}] '
                f'Loss: {total_loss/total_samples:.4f} '
                f'PSNR: {total_psnr/total_samples:.2f} '
                f'AvgA: {total_alloc/total_samples:.3f}'
            )

    avg_loss = total_loss / total_samples
    avg_psnr = total_psnr / total_samples
    avg_A = total_alloc / total_samples

    writer.add_scalar('Train/Loss', avg_loss, epoch)
    writer.add_scalar('Train/PSNR', avg_psnr, epoch)
    writer.add_scalar('Train/AvgA', avg_A, epoch)

    return avg_loss, avg_psnr


# =========================================
# Validation
# =========================================
def validate(model, val_loader, channel, criterion, epoch, args, writer):
    model.eval()

    total_loss = 0
    total_psnr = 0
    total_alloc = 0
    total_samples = 0

    with torch.no_grad():
        for images, _ in val_loader:
            images = images.cuda()

            encoded, decoded, info = model(
                images,
                target_rate=args.target_rate,
                return_info=True
            )

            encoded_noisy = channel(encoded)
            decoded_noisy = model.decoder(encoded_noisy)

            loss = criterion(decoded_noisy, images)

            batch_size = images.size(0)
            psnr = get_psnr(decoded_noisy * 255.0, images * 255.0)
            avg_A = info["A_stats"]["A_mean"]

            total_loss += loss.item() * batch_size
            total_psnr += psnr.item() * batch_size
            total_alloc += avg_A * batch_size
            total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_psnr = total_psnr / total_samples
    avg_A = total_alloc / total_samples

    print(
        f'Validation - Loss: {avg_loss:.4f} '
        f'PSNR: {avg_psnr:.2f} '
        f'AvgA: {avg_A:.3f}'
    )

    writer.add_scalar('Val/Loss', avg_loss, epoch)
    writer.add_scalar('Val/PSNR', avg_psnr, epoch)
    writer.add_scalar('Val/AvgA', avg_A, epoch)

    return avg_loss, avg_psnr


# =========================================
# Dataset Loader
# =========================================
def get_dataset(name, batch_size):

    if name.lower() == 'cifar10':

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        train_dataset = datasets.CIFAR10(
            root='./dataset',
            train=True,
            download=True,
            transform=transform
        )

        val_dataset = datasets.CIFAR10(
            root='./dataset',
            train=False,
            download=True,
            transform=transform
        )

    elif name.lower() == 'imagenet':

        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

        train_dataset = Vanilla('./dataset/ImageNet/train', transform)
        val_dataset = Vanilla('./dataset/ImageNet/val', transform)

    else:
        raise ValueError("Unsupported dataset")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, val_loader


# =========================================
# Main
# =========================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=int, default=16)
    parser.add_argument('--channel_num', type=int, default=16)

    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)

    parser.add_argument('--snr', type=float, default=10.0)
    parser.add_argument('--channel_type', type=str, default='AWGN',
                        choices=['AWGN', 'Rayleigh'])

    parser.add_argument('--target_rate', type=float, default=0.5)

    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default='./out/checkpoint')
    parser.add_argument('--log_dir', type=str, default='./out/logs')

    args = parser.parse_args()

    set_seed(42)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f'FIS_{args.dataset}_{args.channel_type}_SNR{args.snr}_{timestamp}'

    save_path = os.path.join(args.save_dir, exp_name)
    log_path = os.path.join(args.log_dir, exp_name)

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    writer = SummaryWriter(log_path)

    model = DeepJSCC_FIS(c=args.C, snr_db=args.snr, rate_budget=args.target_rate).cuda()
    channel = Channel(channel_type=args.channel_type, snr=args.snr).cuda()

    train_loader, val_loader = get_dataset(args.dataset, args.batch_size)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    best_psnr = 0

    for epoch in range(1, args.epochs + 1):

        print(f'\n=== Epoch {epoch}/{args.epochs} ===')

        train_loss, train_psnr = train_one_epoch(
            model, train_loader, channel, optimizer, criterion, epoch, args, writer
        )

        val_loss, val_psnr = validate(
            model, val_loader, channel, criterion, epoch, args, writer
        )

        scheduler.step()

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(model.state_dict(), os.path.join(save_path, 'best.pth'))
            print(f'Best model saved (PSNR: {val_psnr:.2f} dB)')

    writer.close()
    print(f'\nTraining completed. Best PSNR: {best_psnr:.2f} dB')


if __name__ == '__main__':
    main()
