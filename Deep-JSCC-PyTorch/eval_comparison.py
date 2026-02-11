"""
Evaluation script
Compatible with updated JSCC_FIS model
Pipeline:
encode → channel → decode
"""

import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from model import JSCC, JSCC_FIS
from channel import Channel
from dataset import Vanilla
from utils import get_psnr

import torch.nn.functional as F


# ============================================================
# Simple SSIM
# ============================================================

def calculate_ssim(img1, img2):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu1 = img1.mean()
    mu2 = img2.mean()

    sigma1 = img1.var()
    sigma2 = img2.var()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()

    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2))

    return ssim.item()


# ============================================================
# Evaluation
# ============================================================

def evaluate_model(model, test_loader,
                   snr_list, channel_type,
                   args, device):

    model.eval()

    results = {
        'SNR': snr_list,
        'PSNR': [],
        'SSIM': []
    }

    for snr in snr_list:

        print(f'\nEvaluating at SNR = {snr} dB')

        channel = Channel(channel_type=channel_type,
                          snr=snr).to(device)

        psnr_list = []
        ssim_list = []

        with torch.no_grad():
            for images, _ in test_loader:

                images = images.to(device)

                # =========================
                # Encode
                # =========================
                if isinstance(model, JSCC_FIS):

                    encoded, _, info = model(
                        images,
                        snr=snr,
                        target_rate=args.target_rate,
                        return_info=True
                    )

                else:
                    encoded, _ = model(images)

                # =========================
                # Channel
                # =========================
                encoded_noisy = channel(encoded)

                # =========================
                # Decode
                # =========================
                decoded = model.decoder(encoded_noisy)

                psnr = get_psnr(decoded, images).item()
                ssim = calculate_ssim(decoded, images)

                psnr_list.append(psnr)
                ssim_list.append(ssim)

        avg_psnr = np.mean(psnr_list)
        avg_ssim = np.mean(ssim_list)

        results['PSNR'].append(avg_psnr)
        results['SSIM'].append(avg_ssim)

        print(f'PSNR: {avg_psnr:.2f} dB, SSIM: {avg_ssim:.4f}')

    return results


# ============================================================
# Plot
# ============================================================

def plot_comparison(baseline_results,
                    fis_results,
                    save_path):

    plt.figure(figsize=(12, 5))

    # PSNR
    plt.subplot(1, 2, 1)
    plt.plot(baseline_results['SNR'],
             baseline_results['PSNR'],
             'o-', label='Baseline')

    plt.plot(fis_results['SNR'],
             fis_results['PSNR'],
             's-', label='FIS')

    plt.xlabel('SNR (dB)')
    plt.ylabel('PSNR (dB)')
    plt.title('PSNR vs SNR')
    plt.legend()
    plt.grid(True)

    # SSIM
    plt.subplot(1, 2, 2)
    plt.plot(baseline_results['SNR'],
             baseline_results['SSIM'],
             'o-', label='Baseline')

    plt.plot(fis_results['SNR'],
             fis_results['SSIM'],
             's-', label='FIS')

    plt.xlabel('SNR (dB)')
    plt.ylabel('SSIM')
    plt.title('SSIM vs SNR')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f'Plot saved to {save_path}')


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--baseline_checkpoint',
                        type=str, required=True)

    parser.add_argument('--fis_checkpoint',
                        type=str, required=True)

    parser.add_argument('--test_root',
                        type=str,
                        default='./dataset/test')

    parser.add_argument('--batch_size',
                        type=int,
                        default=32)

    parser.add_argument('--snr_list',
                        nargs='+',
                        type=float,
                        default=[1, 4, 7, 10, 13])

    parser.add_argument('--channel',
                        type=str,
                        default='AWGN',
                        choices=['AWGN', 'Rayleigh'])

    # ⚠ IMPORTANT
    parser.add_argument('--target_rate',
                        type=float,
                        default=8.0)

    parser.add_argument('--save_plot',
                        type=str,
                        default='comparison.png')

    args = parser.parse_args()

    device = torch.device("cuda"
                          if torch.cuda.is_available()
                          else "cpu")

    print('Loading baseline model...')
    baseline = JSCC(C=16,
                    channel_num=16).to(device)
    baseline.load_state_dict(
        torch.load(args.baseline_checkpoint,
                   map_location=device)
    )

    print('Loading FIS model...')
    fis_model = JSCC_FIS(C=16,
                         channel_num=16).to(device)
    fis_model.load_state_dict(
        torch.load(args.fis_checkpoint,
                   map_location=device)
    )

    test_dataset = Vanilla(args.test_root)
    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False)

    print('\n=== Evaluating Baseline ===')
    baseline_results = evaluate_model(
        baseline, test_loader,
        args.snr_list, args.channel,
        args, device
    )

    print('\n=== Evaluating FIS ===')
    fis_results = evaluate_model(
        fis_model, test_loader,
        args.snr_list, args.channel,
        args, device
    )

    print('\n=== Comparison Table ===')
    print(f'{"SNR":<8} {"Baseline":<12} '
          f'{"FIS":<12} {"Gain":<10}')
    print('-' * 45)

    for i, snr in enumerate(args.snr_list):
        gain = fis_results['PSNR'][i] - \
               baseline_results['PSNR'][i]

        print(f'{snr:<8.1f} '
              f'{baseline_results["PSNR"][i]:<12.2f} '
              f'{fis_results["PSNR"][i]:<12.2f} '
              f'{gain:<10.2f}')

    plot_comparison(baseline_results,
                    fis_results,
                    args.save_plot)


if __name__ == '__main__':
    main()
