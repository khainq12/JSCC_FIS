"""
Training script for FIS-enhanced model
Compatible with:
- model.py (JSCC_FIS)
- channel.py (Channel class only)
- dataset.py (Vanilla)
- utils.py (get_psnr)
Now supports:
--dataset cifar10
--dataset folder
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
from datetime import datetime

from torchvision import datasets, transforms

from model import JSCC_FIS
from channel import Channel
from dataset import Vanilla
from utils import get_psnr


# =========================
# AverageMeter
# =========================
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count if self.count != 0 else 0


# =========================
# Train One Epoch
# =========================
def train_one_epoch(model, train_loader, channel, optimizer, criterion, epoch, args, writer):
    model.train()
    device = next(model.parameters()).device

    loss_meter = AverageMeter()
    psnr_meter = AverageMeter()

    for batch_idx, (images, _) in enumerate(train_loader):
        images = images.to(device)

        optimizer.zero_grad()

        encoded, decoded, info = model(
            images,
            snr=args.snr,
            target_rate=args.target_rate,
            return_info=True
        )

        encoded_noisy = channel(encoded)
        decoded_noisy = model.decoder(encoded_noisy)

        loss = criterion(decoded_noisy, images)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        psnr = get_psnr(decoded_noisy, images).item()

        loss_meter.update(loss.item(), images.size(0))
        psnr_meter.update(psnr, images.size(0))

        if batch_idx % args.log_interval == 0:
            print(f'Epoch [{epoch}] [{batch_idx}/{len(train_loader)}] '
                  f'Loss: {loss_meter.avg:.4f} '
                  f'PSNR: {psnr_meter.avg:.2f}')

    writer.add_scalar('Train/Loss', loss_meter.avg, epoch)
    writer.add_scalar('Train/PSNR', psnr_meter.avg, epoch)

    return loss_meter.avg, psnr_meter.avg


# =========================
# Validation
# =========================
def validate(model, val_loader, channel, criterion, epoch, args, writer):
    model.eval()
    device = next(model.parameters()).device

    loss_meter = AverageMeter()
    psnr_meter = AverageMeter()

    with torch.no_grad():
        for images, _ in val_loader:
            images = images.to(device)

            encoded, decoded, info = model(
                images,
                snr=args.snr,
                target_rate=args.target_rate,
                return_info=True
            )

            encoded_noisy = channel(encoded)
            decoded_noisy = model.decoder(encoded_noisy)

            loss = criterion(decoded_noisy, images)
            psnr = get_psnr(decoded_noisy, images).item()

            loss_meter.update(loss.item(), images.size(0))
            psnr_meter.update(psnr, images.size(0))

    print(f'Validation - Loss: {loss_meter.avg:.4f} '
          f'PSNR: {psnr_meter.avg:.2f}')

    writer.add_scalar('Val/Loss', loss_meter.avg, epoch)
    writer.add_scalar('Val/PSNR', psnr_meter.avg, epoch)

    return loss_meter.avg, psnr_meter.avg


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()

    # Dataset choice
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'folder'],
                        help='dataset type')

    parser.add_argument('--data_root', type=str, default='./dataset/train')
    parser.add_argument('--val_root', type=str, default='./dataset/val')

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)

    parser.add_argument('--C', type=int, default=16)
    parser.add_argument('--channel_num', type=int, default=16)

    parser.add_argument('--snr', type=float, default=10.0)
    parser.add_argument('--channel', type=str, default='AWGN',
                        choices=['AWGN', 'Rayleigh'])

    parser.add_argument('--target_rate', type=float, default=0.5)
    parser.add_argument('--log_interval', type=int, default=10)

    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = f'./out/FIS_{timestamp}'
    os.makedirs(save_dir, exist_ok=True)

    writer = SummaryWriter(save_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = JSCC_FIS(C=args.C, channel_num=args.channel_num).to(device)

    # =========================
    # Dataset loading
    # =========================
    if args.dataset == 'cifar10':
        transform = transforms.ToTensor()

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

    elif args.dataset == 'folder':
        train_dataset = Vanilla(args.data_root)
        val_dataset = Vanilla(args.val_root)

    else:
        raise ValueError("Unknown dataset type")

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=(device.type == "cuda"))

    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=(device.type == "cuda"))

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    channel = Channel(channel_type=args.channel,
                      snr=args.snr).to(device)

    best_psnr = 0

    for epoch in range(1, args.epochs + 1):
        print(f'\n===== Epoch {epoch}/{args.epochs} =====')

        train_loss, train_psnr = train_one_epoch(
            model, train_loader, channel,
            optimizer, criterion, epoch, args, writer
        )

        val_loss, val_psnr = validate(
            model, val_loader, channel,
            criterion, epoch, args, writer
        )

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(model.state_dict(),
                       os.path.join(save_dir, 'best.pth'))
            print(f'Best model saved (PSNR: {best_psnr:.2f} dB)')

    writer.close()
    print(f'\nTraining completed. Best PSNR: {best_psnr:.2f} dB')


if __name__ == '__main__':
    main()
