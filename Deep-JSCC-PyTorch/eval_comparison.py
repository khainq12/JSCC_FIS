"""
Evaluation script to compare baseline vs FIS
File: eval_comparison.py (NEW FILE)
"""

import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from model import JSCC, JSCC_FIS
from channel import AWGN, Rayleigh
from dataset import get_dataloader
from utils import calculate_psnr, calculate_ssim


def evaluate_model(model, test_loader, snr_list, channel_type, args):
    """Evaluate model at different SNRs"""
    model.eval()
    
    results = {'SNR': snr_list, 'PSNR': [], 'SSIM': []}
    
    for snr in snr_list:
        print(f'\nEvaluating at SNR = {snr} dB')
        
        # Channel
        if channel_type == 'AWGN':
            channel = AWGN(snr=snr)
        else:
            channel = Rayleigh(snr=snr)
        
        psnr_list = []
        ssim_list = []
        
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.cuda()
                
                # Forward
                if isinstance(model, JSCC_FIS):
                    encoded, decoded, info = model(images, snr=snr, target_rate=args.target_rate, return_info=True)
                else:
                    encoded, decoded = model(images)
                
                # Apply channel
                encoded_noisy = channel(encoded)
                decoded_noisy = model.decoder(encoded_noisy)
                
                # Metrics
                psnr = calculate_psnr(images, decoded_noisy)
                ssim = calculate_ssim(images, decoded_noisy)
                
                psnr_list.append(psnr)
                ssim_list.append(ssim)
        
        avg_psnr = np.mean(psnr_list)
        avg_ssim = np.mean(ssim_list)
        
        results['PSNR'].append(avg_psnr)
        results['SSIM'].append(avg_ssim)
        
        print(f'PSNR: {avg_psnr:.2f} dB, SSIM: {avg_ssim:.4f}')
    
    return results


def plot_comparison(baseline_results, fis_results, save_path):
    """Plot comparison"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # PSNR
    axes[0].plot(baseline_results['SNR'], baseline_results['PSNR'], 'o-', label='Baseline', linewidth=2)
    axes[0].plot(fis_results['SNR'], fis_results['PSNR'], 's-', label='FIS-Enhanced', linewidth=2)
    axes[0].set_xlabel('SNR (dB)', fontsize=12)
    axes[0].set_ylabel('PSNR (dB)', fontsize=12)
    axes[0].set_title('PSNR vs SNR', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # SSIM
    axes[1].plot(baseline_results['SNR'], baseline_results['SSIM'], 'o-', label='Baseline', linewidth=2)
    axes[1].plot(fis_results['SNR'], fis_results['SSIM'], 's-', label='FIS-Enhanced', linewidth=2)
    axes[1].set_xlabel('SNR (dB)', fontsize=12)
    axes[1].set_ylabel('SSIM', fontsize=12)
    axes[1].set_title('SSIM vs SNR', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f'Plot saved to {save_path}')


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--baseline_checkpoint', type=str, required=True)
    parser.add_argument('--fis_checkpoint', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--snr_list', nargs='+', type=float, default=[1, 4, 7, 10, 13])
    parser.add_argument('--channel', type=str, default='AWGN')
    parser.add_argument('--target_rate', type=float, default=0.5)
    parser.add_argument('--save_plot', type=str, default='comparison.png')
    
    args = parser.parse_args()
    
    # Load models
    print('Loading baseline model...')
    baseline = JSCC(C=16, channel_num=16).cuda()
    checkpoint = torch.load(args.baseline_checkpoint)
    baseline.load_state_dict(checkpoint['model_state_dict'])
    
    print('Loading FIS model...')
    fis_model = JSCC_FIS(C=16, channel_num=16).cuda()
    checkpoint = torch.load(args.fis_checkpoint)
    fis_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Dataset
    test_loader = get_dataloader(args.dataset, 'test', batch_size=32)
    
    # Evaluate
    print('\n=== Evaluating Baseline ===')
    baseline_results = evaluate_model(baseline, test_loader, args.snr_list, args.channel, args)
    
    print('\n=== Evaluating FIS ===')
    fis_results = evaluate_model(fis_model, test_loader, args.snr_list, args.channel, args)
    
    # Print table
    print('\n=== Comparison Table ===')
    print(f'{"SNR":<8} {"Baseline PSNR":<15} {"FIS PSNR":<15} {"Gain":<10}')
    print('-' * 50)
    for i, snr in enumerate(args.snr_list):
        gain = fis_results['PSNR'][i] - baseline_results['PSNR'][i]
        print(f'{snr:<8.1f} {baseline_results["PSNR"][i]:<15.2f} {fis_results["PSNR"][i]:<15.2f} {gain:<10.2f}')
    
    # Plot
    plot_comparison(baseline_results, fis_results, args.save_plot)


if __name__ == '__main__':
    main()
