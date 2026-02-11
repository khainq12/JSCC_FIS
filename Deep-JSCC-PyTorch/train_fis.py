"""
Training script for FIS-Enhanced Deep JSCC
File: train_fis.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
from datetime import datetime

from model import JSCC_FIS
from channel import AWGN, Rayleigh
from dataset import get_dataloader
from utils import AverageMeter, calculate_psnr


# ============================================================
# Train One Epoch
# ============================================================

def train_one_epoch(model, train_loader, channel, optimizer, criterion,
                    epoch, args, writer, device):

    model.train()

    loss_meter = AverageMeter()
    psnr_meter = AverageMeter()
    bits_meter = AverageMeter()

    for batch_idx, (images, _) in enumerate(train_loader):

        images = images.to(device)

        optimizer.zero_grad()

        # Forward (includes encoder + FIS quantization)
        encoded, _, info = model(
            images,
            snr=args.snr,
            target_rate=args.target_rate,
            return_info=True
        )

        # Channel
        encoded_noisy = channel(encoded)

        # Decode AFTER channel
        decoded = model.decoder(encoded_noisy)

        # Loss
        loss = criterion(decoded, images)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Metrics
        psnr = calculate_psnr(images, decoded)
        avg_bits = info["avg_bits"]

        loss_meter.update(loss.item(), images.size(0))
        psnr_meter.update(psnr, images.size(0))
        bits_meter.update(avg_bits, images.size(0))

        if batch_idx % args.log_interval == 0:
            print(f'Epoch [{epoch}] [{batch_idx}/{len(train_loader)}] '
                  f'Loss: {loss_meter.avg:.4f} '
                  f'PSNR: {psnr_meter.avg:.2f} '
                  f'Bits: {bits_meter.avg:.2f}')

    # TensorBoard
    writer.add_scalar('Train/Loss', loss_meter.avg, epoch)
    writer.add_scalar('Train/PSNR', psnr_meter.avg, epoch)
    writer.add_scalar('Train/AvgBits', bits_meter.avg, epoch)

    return loss_meter.avg, psnr_meter.avg, bits_meter.avg


# ============================================================
# Validation
# ============================================================

def validate(model, val_loader, channel, criterion,
             epoch, args, writer, device):

    model.eval()

    loss_meter = AverageMeter()
    psnr_meter = AverageMeter()
    bits_meter = AverageMeter()

    with torch.no_grad():

        for images, _ in val_loader:

            images = images.to(device)

            encoded, _, info = model(
                images,
                snr=args.snr,
                target_rate=args.target_rate,
                return_info=True
            )

            encoded_noisy = channel(encoded)
            decoded = model.decoder(encoded_noisy)

            loss = criterion(decoded, images)
            psnr = calculate_psnr(images, decoded)
            avg_bits = info["avg_bits"]

            loss_meter.update(loss.item(), images.size(0))
            psnr_meter.update(psnr, images.size(0))
            bits_meter.update(avg_bits, images.size(0))

    print(f'Validation - '
          f'Loss: {loss_meter.avg:.4f} '
          f'PSNR: {psnr_meter.avg:.2f} '
          f'Bits: {bits_meter.avg:.2f}')

    writer.add_scalar('Val/Loss', loss_meter.avg, epoch)
    writer.add_scalar('Val/PSNR', psnr_meter.avg, epoch)
    writer.add_scalar('Val/AvgBits', bits_meter.avg, epoch)

    return loss_meter.avg, psnr_meter.avg, bits_meter.avg


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument('--C', type=int, default=16)
    parser.add_argument('--channel_num', type=int, default=16)

    # Training
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    # Channel
    parser.add_argument('--snr', type=float, default=10.0)
    parser.add_argument('--channel', type=str, default='AWGN',
                        choices=['AWGN', 'Rayleigh'])

    # FIS
    parser.add_argument('--target_rate', type=float, default=0.5)

    # Misc
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--save_dir', type=str, default='./out/checkpoint')
    parser.add_argument('--log_dir', type=str, default='./out/logs')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ========================================================
    # Experiment folder
    # ========================================================

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f'FIS_{args.dataset}_C{args.C}_SNR{args.snr}_{timestamp}'

    save_path = os.path.join(args.save_dir, exp_name)
    log_path = os.path.join(args.log_dir, exp_name)

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    writer = SummaryWriter(log_path)

    # ========================================================
    # Model
    # ========================================================

    model = JSCC_FIS(
        C=args.C,
        channel_num=args.channel_num
    ).to(device)

    # ========================================================
    # Dataset
    # ========================================================

    train_loader = get_dataloader(args.dataset, 'train', args.batch_size)
    val_loader = get_dataloader(args.dataset, 'val', args.batch_size)

    # ========================================================
    # Loss & Optimizer
    # ========================================================

    criterion = nn.MSELoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=30,
        gamma=0.5
    )

    # ========================================================
    # Channel
    # ========================================================

    if args.channel == 'AWGN':
        channel = AWGN(args.snr)
    else:
        channel = Rayleigh(args.snr)

    channel = channel.to(device)

    # ========================================================
    # Training Loop
    # ========================================================

    best_psnr = 0.0

    for epoch in range(1, args.epochs + 1):

        print(f'\n=== Epoch {epoch}/{args.epochs} ===')

        train_loss, train_psnr, train_bits = train_one_epoch(
            model, train_loader, channel, optimizer,
            criterion, epoch, args, writer, device
        )

        val_loss, val_psnr, val_bits = validate(
            model, val_loader, channel,
            criterion, epoch, args, writer, device
        )

        scheduler.step()

        # Save best model
        if val_psnr > best_psnr:
            best_psnr = val_psnr

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'psnr': val_psnr,
                'args': vars(args)
            }, os.path.join(save_path, 'best.pth'))

            print(f'Best model saved (PSNR: {val_psnr:.2f} dB)')

        # Periodic checkpoint
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(save_path, f'epoch_{epoch}.pth'))

    writer.close()

    print(f'\nTraining completed. Best PSNR: {best_psnr:.2f} dB')


if __name__ == '__main__':
    main()
