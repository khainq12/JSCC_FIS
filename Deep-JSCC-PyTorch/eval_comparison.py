"""
Evaluation script to compare baseline vs FIS
File: eval_comparison.py
"""

import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os

from model import JSCC, JSCC_FIS
from channel import AWGN, Rayleigh
from dataset import get_dataloader
from utils import calculate_psnr, calculate_ssim


# ============================================================
# Evaluation
# ============================================================

def evaluate_model(model, test_loader, snr_list,
                   channel_type, args, device):

    model.eval()

    results = {
        "SNR": snr_list,
        "PSNR": [],
        "SSIM": []
    }

    for snr in snr_list:

        print(f'\nEvaluating at SNR = {snr} dB')

        # Channel
        if channel_type == "AWGN":
            channel = AWGN(snr)
        else:
            channel = Rayleigh(snr)

        channel = channel.to(device)

        psnr_values = []
        ssim_values = []

        with torch.no_grad():

            for images, _ in test_loader:

                images = images.to(device)

                # Forward
                if isinstance(model, JSCC_FIS):
                    encoded, _, _ = model(
                        images,
                        snr=snr,
                        target_rate=args.target_rate,
                        return_info=True
                    )
                else:
                    encoded, _ = model(images)

                # Channel
                encoded_noisy = channel(encoded)

                # Decode AFTER channel
                decoded = model.decoder(encoded_noisy)

                # Metrics
                psnr = calculate_psnr(images, decoded)
                ssim = calculate_ssim(images, decoded)

                psnr_values.append(float(psnr))
                ssim_values.append(float(ssim))

        avg_psnr = np.mean(psnr_values)
        avg_ssim = np.mean(ssim_values)

        results["PSNR"].append(avg_psnr)
        results["SSIM"].append(avg_ssim)

        print(f'PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f}')

    return results


# ============================================================
# Plot
# ============================================================

def plot_comparison(baseline_results, fis_results, save_path):

    plt.figure(figsize=(12, 5))

    # PSNR
    plt.subplot(1, 2, 1)
    plt.plot(baseline_results["SNR"], baseline_results["PSNR"], 'o-', label="Baseline")
    plt.plot(fis_results["SNR"], fis_results["PSNR"], 's-', label="FIS")
    plt.xlabel("SNR (dB)")
    plt.ylabel("PSNR (dB)")
    plt.title("PSNR vs SNR")
    plt.legend()
    plt.grid(True)

    # SSIM
    plt.subplot(1, 2, 2)
    plt.plot(baseline_results["SNR"], baseline_results["SSIM"], 'o-', label="Baseline")
    plt.plot(fis_results["SNR"], fis_results["SSIM"], 's-', label="FIS")
    plt.xlabel("SNR (dB)")
    plt.ylabel("SSIM")
    plt.title("SSIM vs SNR")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\nPlot saved to {save_path}")


# ============================================================
# Main
# ============================================================

def load_checkpoint(model, path, device):

    checkpoint = torch.load(path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    return model


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--baseline_checkpoint", type=str, required=True)
    parser.add_argument("--fis_checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--snr_list", nargs="+", type=float,
                        default=[1, 4, 7, 10, 13])
    parser.add_argument("--channel", type=str, default="AWGN",
                        choices=["AWGN", "Rayleigh"])
    parser.add_argument("--target_rate", type=float, default=0.5)
    parser.add_argument("--save_plot", type=str, default="comparison.png")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ========================================================
    # Load models
    # ========================================================

    print("Loading baseline model...")
    baseline = JSCC(C=16, channel_num=16).to(device)
    baseline = load_checkpoint(baseline, args.baseline_checkpoint, device)

    print("Loading FIS model...")
    fis_model = JSCC_FIS(C=16, channel_num=16).to(device)
    fis_model = load_checkpoint(fis_model, args.fis_checkpoint, device)

    # ========================================================
    # Dataset
    # ========================================================

    test_loader = get_dataloader(args.dataset, "test", batch_size=32)

    # ========================================================
    # Evaluate
    # ========================================================

    print("\n=== Evaluating Baseline ===")
    baseline_results = evaluate_model(
        baseline, test_loader, args.snr_list,
        args.channel, args, device
    )

    print("\n=== Evaluating FIS ===")
    fis_results = evaluate_model(
        fis_model, test_loader, args.snr_list,
        args.channel, args, device
    )

    # ========================================================
    # Print Table
    # ========================================================

    print("\n=== Comparison Table ===")
    print(f'{"SNR":<8} {"Baseline":<15} {"FIS":<15} {"Gain":<10}')
    print("-" * 50)

    for i, snr in enumerate(args.snr_list):
        gain = fis_results["PSNR"][i] - baseline_results["PSNR"][i]
        print(f'{snr:<8.1f} '
              f'{baseline_results["PSNR"][i]:<15.2f} '
              f'{fis_results["PSNR"][i]:<15.2f} '
              f'{gain:<10.2f}')

    # ========================================================
    # Plot
    # ========================================================

    plot_comparison(baseline_results, fis_results, args.save_plot)


if __name__ == "__main__":
    main()
