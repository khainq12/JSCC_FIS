"""
Training script for FIS-enhanced Deep JSCC
Fully compatible with your current utils.py
"""

import os
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from model import JSCC_FIS
from channel import Channel
from dataset import Vanilla
from utils import get_psnr


# =====================================================
# Average Meter (self-contained)
# =====================================================

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# =====================================================
# Training
# =====================================================

def train_one_epoch(model, train_loader, channel, optimizer, criterion, epoch, args, writer):
    model.train()

    loss_meter = AverageMeter()
    psnr_meter = AverageMeter()
    bits_meter = AverageMeter()

    for batch_idx, (images, _) in enumerate(train_loader):

        images = images.cuda(non_blocking=True)

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

        psnr = get_psnr(decoded_noisy * 255.0, images * 255.0).item()
        avg_bits = info['avg_bits']

        loss_meter.update(loss.item(), images.size(0))
        psnr_meter.update(psnr, images.size(0))
        bits_meter.update(avg_bits, images.size(0))

        if batch_idx % args.log_interval == 0:
            print(f'Epoch [{epoch}] [{batch_idx}/{len(train_loader)}] '
                  f'Loss: {loss_meter.avg:.4f} '
                  f'PSNR: {psnr_meter.avg:.2f} '
                  f'Bits: {bits_meter.avg:.2f}')

    writer.add_scalar('Train/Loss', loss_meter.avg, epoch)
    writer.add_scalar('Train/PSNR', psnr_meter.avg, epoch)
    writer.add_scalar('Train/AvgBits', bits_meter.avg, epoch)

    return loss_meter.avg, psnr_meter.avg, bits_meter.avg


# =====================================================
# Validation
# =====================================================

def validate(model, val_loader, channel, criterion, epoch, args, writer):
    model.eval()

    loss_meter = AverageMeter()
    psnr_meter = AverageMeter()
    bits_meter = AverageMeter()

    with torch.no_grad():
        for images, _ in val_loader:

            images = images.cuda(non_blocking=True)

            encoded, decoded, info = model(
                images,
                snr=args.snr,
                target_rate=args.target_rate,
                return_info=True
            )

            encoded_noisy = channel(encoded)
            decoded_noisy = model.decoder(encoded_noisy)

            loss = criterion(decoded_noisy, images)
            psnr = get_psnr(decoded_noisy * 255.0, images * 255.0).item()
            avg_bits = info['avg_bits']

            loss_meter.update(loss.item(), images.size(0))
            psnr_meter.update(psnr, images.size(0))
            bits_meter.update(avg_bits, images.size(0))

    print(f'Validation - Loss: {loss_meter.avg:.4f} '
          f'PSNR: {psnr_meter.avg:.2f} '
          f'Bits: {bits_meter.avg:.2f}')

    writer.add_scalar('Val/Loss', loss_meter.avg, epoch)
    writer.add_scalar('Val/PSNR', psnr_meter.avg, epoch)
    writer.add_scalar('Val/AvgBits', bits_meter.avg, epoch)

    return loss_meter.avg, psnr_meter.avg, bits_meter.avg


# =====================================================
# Main
# =====================================================

def main():

    parser = argparse.ArgumentParser(description='Train FIS-Enhanced Deep JSCC')

    parser.add_argument('--C', type=int, default=16)
    parser.add_argument('--channel_num', type=int, default=16)

    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'imagenet'])
    parser.add_argument('--batch_size', type=int, default=64)

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    parser.add_argument('--snr', type=float, default=10.0)
    parser.add_argument('--channel', type=str, default='AWGN',
                        choices=['AWGN', 'Rayleigh'])

    parser.add_argument('--target_rate', type=float, default=0.5)

    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default='./out/checkpoint')
    parser.add_argument('--log_dir', type=str, default='./out/logs')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f'FIS_{args.dataset}_C{args.C}_SNR{args.snr}_{timestamp}'

    save_path = os.path.join(args.save_dir, exp_name)
    log_path = os.path.join(args.log_dir, exp_name)

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    writer = SummaryWriter(log_path)

    # Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    if args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(
            root='./data',
            train=True,
            download=True,
            transform=transform
        )

        val_dataset = datasets.CIFAR10(
            root='./data',
            train=False,
            download=True,
            transform=transform
        )
    else:
        train_dataset = Vanilla('./dataset/ImageNet/train', transform)
        val_dataset = Vanilla('./dataset/ImageNet/val', transform)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=2,
                              pin_memory=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=2,
                            pin_memory=True)

    model = JSCC_FIS(C=args.C, channel_num=args.channel_num).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=640,
                                          gamma=0.1)

    channel = Channel(channel_type=args.channel, snr=args.snr)

    best_psnr = 0

    for epoch in range(1, args.epochs + 1):

        print(f'\n=== Epoch {epoch}/{args.epochs} ===')

        train_loss, train_psnr, train_bits = train_one_epoch(
            model, train_loader, channel,
            optimizer, criterion, epoch, args, writer
        )

        val_loss, val_psnr, val_bits = validate(
            model, val_loader, channel,
            criterion, epoch, args, writer
        )

        scheduler.step()

        if val_psnr > best_psnr:
            best_psnr = val_psnr

            torch.save(model.state_dict(),
                       os.path.join(save_path, 'best.pth'))

            print(f'Best model saved (PSNR: {val_psnr:.2f} dB)')

    writer.close()
    print(f'\nTraining completed. Best PSNR: {best_psnr:.2f} dB')


if __name__ == '__main__':
    main()
