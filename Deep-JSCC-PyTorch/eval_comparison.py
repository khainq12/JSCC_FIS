"""
Evaluation script to compare Baseline DeepJSCC vs FIS-JSCC
Compatible with:
- model_baseline.py (DeepJSCC original)
- model.py (JSCC_FIS)
"""

import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import JSCC_FIS
from model_baseline import DeepJSCC
from channel import Channel
from utils import get_psnr


# =========================================
# Simple SSIM
# =========================================
def simple_ssim(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = x.mean()
    mu_y = y.mean()

    sigma_x = x.var()
    sigma_y = y.var()
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean()

    ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
        (mu_x ** 2 + mu_y ** 2 + C1) *
        (sigma_x + sigma_y + C2)
    )

    return ssim.item()


# =========================================
# Dataset Loader (CIFAR10 only)
# =========================================
def get_test_loader(dataset_name):

    if dataset_name.lower() == "cifar10":

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        test_dataset = datasets.CIFAR10(
            root="./dataset",
            train=False,
            download=True,
            transform=transform
        )

    else:
        raise ValueError("Currently only CIFAR10 supported.")

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return test_loader


# =========================================
# Evaluate model
# =========================================
def evaluate_model(model, test_loader, snr_list, channel_type, args):

    model.eval()
    results = {"SNR": snr_list, "PSNR": [], "SSIM": []}

    for snr in snr_list:

        print(f"\nEvaluating at SNR = {snr} dB")

        channel = Channel(channel_type=channel_type, snr=snr).cuda()

        psnr_list = []
        ssim_list = []

        with torch.no_grad():
            for images, _ in test_loader:

                images = images.cuda()

                # =========================
                # BASELINE DeepJSCC
                # =========================
                if isinstance(model, DeepJSCC):

                    # encode
                    encoded = model.encoder(images)

                    # channel
                    encoded_noisy = channel(encoded)

                    # decode
                    decoded = model.decoder(encoded_noisy)

                # =========================
                # FIS-JSCC
                # =========================
                else:
                    encoded, decoded, info = model(
                        images,
                        snr=snr,
                        target_rate=args.target_rate,
                        return_info=True
                    )

                    # channel after quantization
                    encoded_noisy = channel(encoded)
                    decoded = model.decoder(encoded_noisy)

                # Metrics
                psnr = get_psnr(decoded * 255.0, images * 255.0)
                ssim = simple_ssim(images, decoded)

                psnr_list.append(psnr.item())
                ssim_list.append(ssim)

        avg_psnr = np.mean(psnr_list)
        avg_ssim = np.mean(ssim_list)

        results["PSNR"].append(avg_psnr)
        results["SSIM"].append(avg_ssim)

        print(f"PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f}")

    return results


# =========================================
# Plot comparison
# =========================================
def plot_comparison(baseline_results, fis_results, save_path):

    plt.figure(figsize=(12, 5))

    # PSNR
    plt.subplot(1, 2, 1)
    plt.plot(baseline_results["SNR"], baseline_results["PSNR"], "o-", label="Baseline")
    plt.plot(fis_results["SNR"], fis_results["PSNR"], "s-", label="FIS-Enhanced")
    plt.xlabel("SNR (dB)")
    plt.ylabel("PSNR (dB)")
    plt.title("PSNR vs SNR")
    plt.grid(True)
    plt.legend()

    # SSIM
    plt.subplot(1, 2, 2)
    plt.plot(baseline_results["SNR"], baseline_results["SSIM"], "o-", label="Baseline")
    plt.plot(fis_results["SNR"], fis_results["SSIM"], "s-", label="FIS-Enhanced")
    plt.xlabel("SNR (dB)")
    plt.ylabel("SSIM")
    plt.title("SSIM vs SNR")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\nPlot saved to {save_path}")


# =========================================
# Main
# =========================================
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

    # =========================
    # Load Baseline
    # =========================
    print("Loading baseline DeepJSCC...")
    baseline = DeepJSCC(c=16).cuda()
    baseline.load_state_dict(torch.load(args.baseline_checkpoint))
    baseline.eval()

    # =========================
    # Load FIS
    # =========================
    print("Loading FIS model...")
    fis_model = JSCC_FIS(C=16, channel_num=16).cuda()
    fis_model.load_state_dict(torch.load(args.fis_checkpoint))
    fis_model.eval()

    test_loader = get_test_loader(args.dataset)

    print("\n=== Evaluating Baseline ===")
    baseline_results = evaluate_model(
        baseline, test_loader, args.snr_list, args.channel, args
    )

    print("\n=== Evaluating FIS ===")
    fis_results = evaluate_model(
        fis_model, test_loader, args.snr_list, args.channel, args
    )

    # =========================
    # Comparison Table
    # =========================
    print("\n=== Comparison Table ===")
    print(f"{'SNR':<8}{'Baseline':<12}{'FIS':<12}{'Gain':<10}")
    print("-" * 40)

    for i, snr in enumerate(args.snr_list):
        gain = fis_results["PSNR"][i] - baseline_results["PSNR"][i]
        print(f"{snr:<8.1f}"
              f"{baseline_results['PSNR'][i]:<12.2f}"
              f"{fis_results['PSNR'][i]:<12.2f}"
              f"{gain:<10.2f}")

    plot_comparison(baseline_results, fis_results, args.save_plot)


if __name__ == "__main__":
    main()
