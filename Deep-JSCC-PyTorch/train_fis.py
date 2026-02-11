"""
Training script for FIS-enhanced model
File: train_fis.py (NEW FILE)
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


def train_one_epoch(model, train_loader, channel, optimizer, criterion, epoch, args, writer):
    """Train one epoch"""
    model.train()
    
    loss_meter = AverageMeter()
    psnr_meter = AverageMeter()
    bits_meter = AverageMeter()
    
    for batch_idx, (images, _) in enumerate(train_loader):
        images = images.cuda()
        
        optimizer.zero_grad()
        
        # Forward with FIS
        encoded, decoded, info = model(
            images, 
            snr=args.snr, 
            target_rate=args.target_rate,
            return_info=True
        )
        
        # Apply channel
        encoded_noisy = channel(encoded)
        
        # Decode noisy signal
        decoded_noisy = model.decoder(encoded_noisy)
        
        # Loss
        loss = criterion(decoded_noisy, images)
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Metrics
        psnr = calculate_psnr(images, decoded_noisy)
        avg_bits = info['avg_bits']
        
        loss_meter.update(loss.item(), images.size(0))
        psnr_meter.update(psnr, images.size(0))
        bits_meter.update(avg_bits, images.size(0))
        
        if batch_idx % args.log_interval == 0:
            print(f'Epoch [{epoch}][{batch_idx}/{len(train_loader)}] '
                  f'Loss: {loss_meter.avg:.4f} '
                  f'PSNR: {psnr_meter.avg:.2f} '
                  f'Bits: {bits_meter.avg:.2f}')
    
    # Log to tensorboard
    global_step = epoch * len(train_loader)
    writer.add_scalar('Train/Loss', loss_meter.avg, global_step)
    writer.add_scalar('Train/PSNR', psnr_meter.avg, global_step)
    writer.add_scalar('Train/AvgBits', bits_meter.avg, global_step)
    
    return loss_meter.avg, psnr_meter.avg, bits_meter.avg


def validate(model, val_loader, channel, criterion, epoch, args, writer):
    """Validation"""
    model.eval()
    
    loss_meter = AverageMeter()
    psnr_meter = AverageMeter()
    bits_meter = AverageMeter()
    
    with torch.no_grad():
        for images, _ in val_loader:
            images = images.cuda()
            
            # Forward
            encoded, decoded, info = model(
                images,
                snr=args.snr,
                target_rate=args.target_rate,
                return_info=True
            )
            
            # Apply channel
            encoded_noisy = channel(encoded)
            decoded_noisy = model.decoder(encoded_noisy)
            
            # Metrics
            loss = criterion(decoded_noisy, images)
            psnr = calculate_psnr(images, decoded_noisy)
            avg_bits = info['avg_bits']
            
            loss_meter.update(loss.item(), images.size(0))
            psnr_meter.update(psnr, images.size(0))
            bits_meter.update(avg_bits, images.size(0))
    
    print(f'Validation - Loss: {loss_meter.avg:.4f} '
          f'PSNR: {psnr_meter.avg:.2f} '
          f'Bits: {bits_meter.avg:.2f}')
    
    # Log to tensorboard
    writer.add_scalar('Val/Loss', loss_meter.avg, epoch)
    writer.add_scalar('Val/PSNR', psnr_meter.avg, epoch)
    writer.add_scalar('Val/AvgBits', bits_meter.avg, epoch)
    
    return loss_meter.avg, psnr_meter.avg, bits_meter.avg


def main():
    parser = argparse.ArgumentParser(description='Train FIS-Enhanced Deep JSCC')
    
    # Model
    parser.add_argument('--C', type=int, default=16, help='Number of channels')
    parser.add_argument('--channel_num', type=int, default=16, help='Latent channels')
    
    # Training
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'imagenet'])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    
    # Channel
    parser.add_argument('--snr', type=float, default=10.0, help='SNR in dB')
    parser.add_argument('--channel', type=str, default='AWGN', choices=['AWGN', 'Rayleigh'])
    
    # FIS
    parser.add_argument('--target_rate', type=float, default=0.5, help='Target rate [0, 1]')
    
    # Misc
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default='./out/checkpoint')
    parser.add_argument('--log_dir', type=str, default='./out/logs')
    
    args = parser.parse_args()
    
    # Create directories
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f'FIS_{args.dataset}_C{args.C}_SNR{args.snr}_{timestamp}'
    
    save_path = os.path.join(args.save_dir, exp_name)
    log_path = os.path.join(args.log_dir, exp_name)
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    
    # Tensorboard
    writer = SummaryWriter(log_path)
    
    # Model
    model = JSCC_FIS(C=args.C, channel_num=args.channel_num).cuda()
    print(f"Model created: {exp_name}")
    
    # Dataset
    train_loader = get_dataloader(args.dataset, 'train', args.batch_size)
    val_loader = get_dataloader(args.dataset, 'val', args.batch_size)
    
    # Loss & Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=640, gamma=0.1)
    
    # Channel
    if args.channel == 'AWGN':
        channel = AWGN(snr=args.snr)
    else:
        channel = Rayleigh(snr=args.snr)
    
    # Training loop
    best_psnr = 0
    
    for epoch in range(1, args.epochs + 1):
        print(f'\n=== Epoch {epoch}/{args.epochs} ===')
        
        # Train
        train_loss, train_psnr, train_bits = train_one_epoch(
            model, train_loader, channel, optimizer, criterion, epoch, args, writer
        )
        
        # Validate
        val_loss, val_psnr, val_bits = validate(
            model, val_loader, channel, criterion, epoch, args, writer
        )
        
        # Scheduler
        scheduler.step()
        
        # Save best model
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'psnr': val_psnr,
                'args': args
            }, os.path.join(save_path, 'best.pth'))
            print(f'Best model saved (PSNR: {val_psnr:.2f} dB)')
        
        # Save checkpoint
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
