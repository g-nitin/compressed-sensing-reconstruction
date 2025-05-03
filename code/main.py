import os

import matplotlib.pyplot as plt
import numpy as np
from algorithms import fista, ista
from cs_utils import (
    calculate_psnr,
    load_image,
    plot_convergence,
    plot_results,
    save_image,
)
from operators import (
    GaussianMeasurementOperator,
    PartialFourierMeasurementOperator,
    WaveletOperator,
)

# Configuration
ROOT = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(ROOT, "..", "data", "victor-crespo-FeuL_GAoSWQ-unsplash.jpg")
IMAGE_SIZE = 1024  # Resize image to IMAGE_SIZE x IMAGE_SIZE; Or None
SAMPLING_RATES = [0.10, 0.25, 0.50, 0.75]  # 10%, 25%, ...
MEASUREMENT_TYPE = "fourier"  # 'gaussian' or 'fourier'
WAVELET_NAME = "db4"  # Wavelet basis for sparsifying transform
WAVELET_LEVEL = None  # Let PyWavelets choose level, or set integer
WAVELET_MODE = "periodization"  # Or "symmetric", etc.

# Regularization parameter
# Usually start with a small value, increase if reconstruction is noisy, decrease if too smooth/features lost
LAMBDA_REG = 0.01

# Algorithm parameters
MAX_ITER = 200
TOL = 1e-4
USE_BACKTRACKING = True
VERBOSE = True  # Show progress bars and iteration info

# Output directory
RESULTS_DIR = os.path.join(ROOT, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "plots"), exist_ok=True)

# Load Image
print(f"Loading image: {IMAGE_PATH}")
img_true = load_image(IMAGE_PATH, size=IMAGE_SIZE)
img_shape = img_true.shape
n_pixels = np.prod(img_shape)
print(f"Image size: {img_shape}, Total pixels: {n_pixels}")

# Setup Sparsifying Transform Operator
print(f"Using Wavelet Transform: {WAVELET_NAME}, Mode: {WAVELET_MODE}")
# Pass the mode explicitly
Psi = WaveletOperator(
    img_shape, wavelet=WAVELET_NAME, level=WAVELET_LEVEL, mode=WAVELET_MODE
)
print(f"Wavelet coefficient vector length: {Psi.coeffs_vec_len}")

# Run Experiments
results_summary = []
reconstructions_dict = {}
results_dict = {}

for rate in SAMPLING_RATES:
    print(f"\n--- Running Experiment: Sampling Rate = {rate * 100:.0f}%")
    num_measurements = int(rate * n_pixels)
    print(f"Number of measurements (m): {num_measurements}")

    # Setup Measurement Operator
    if MEASUREMENT_TYPE.lower() == "gaussian":
        print("Using Gaussian measurements")
        A = GaussianMeasurementOperator(num_measurements, n_pixels)
    elif MEASUREMENT_TYPE.lower() == "fourier":
        print("Using Partial Fourier measurements")
        A = PartialFourierMeasurementOperator(num_measurements, img_shape)
    else:
        raise ValueError(f"Unknown measurement type: {MEASUREMENT_TYPE}")

    # Generate Measurements
    print("Generating measurements y = A @ x_true...")
    x_true_vec = img_true.ravel()
    y = A.forward(x_true_vec)
    # Add noise
    # noise_level = 0.01
    # noise = np.random.randn(*y.shape) * noise_level * np.linalg.norm(y) / np.sqrt(len(y))
    # y += noise
    print(f"Measurement vector shape: {y.shape}")

    # Run ISTA
    img_recon_ista, s_ista, obj_hist_ista, time_hist_ista, time_ista = ista(
        A,
        Psi,
        y,
        lambda_reg=LAMBDA_REG,
        max_iter=MAX_ITER,
        tol=TOL,
        use_backtracking=USE_BACKTRACKING,
        verbose=VERBOSE,
    )
    psnr_ista = calculate_psnr(img_true, img_recon_ista)
    print(f"ISTA Reconstruction PSNR: {psnr_ista:.2f} dB")
    results_summary.append(
        {
            "rate": rate,
            "algo": "ISTA",
            "psnr": psnr_ista,
            "time": time_ista,
            "lambda": LAMBDA_REG,
            "measurement": MEASUREMENT_TYPE,
        }
    )
    reconstructions_dict[(rate, "ISTA")] = img_recon_ista
    results_dict[(rate, "ISTA")] = {"psnr": psnr_ista, "time": time_ista}

    # Save ISTA results
    save_image(
        img_recon_ista,
        os.path.join(
            RESULTS_DIR, "images", f"recon_ista_{MEASUREMENT_TYPE}_{rate * 100:.0f}.png"
        ),
    )
    fig_ista_recon = plot_results(
        img_true, img_recon_ista, MEASUREMENT_TYPE, rate, "ISTA", psnr_ista, time_ista
    )
    fig_ista_recon.savefig(
        os.path.join(
            RESULTS_DIR, "plots", f"recon_ista_{MEASUREMENT_TYPE}_{rate * 100:.0f}.png"
        )
    )
    plt.close(fig_ista_recon)
    fig_ista_conv = plot_convergence(obj_hist_ista, "ISTA")
    fig_ista_conv.savefig(
        os.path.join(
            RESULTS_DIR, "plots", f"conv_ista_{MEASUREMENT_TYPE}_{rate * 100:.0f}.png"
        )
    )
    plt.close(fig_ista_conv)

    # Run FISTA
    img_recon_fista, s_fista, obj_hist_fista, time_hist_fista, time_fista = fista(
        A,
        Psi,
        y,
        lambda_reg=LAMBDA_REG,
        max_iter=MAX_ITER,
        tol=TOL,
        use_backtracking=USE_BACKTRACKING,
        verbose=VERBOSE,
    )
    psnr_fista = calculate_psnr(img_true, img_recon_fista)
    print(f"FISTA Reconstruction PSNR: {psnr_fista:.2f} dB")
    results_summary.append(
        {
            "rate": rate,
            "algo": "FISTA",
            "psnr": psnr_fista,
            "time": time_fista,
            "lambda": LAMBDA_REG,
            "measurement": MEASUREMENT_TYPE,
        }
    )
    reconstructions_dict[(rate, "FISTA")] = img_recon_fista
    results_dict[(rate, "FISTA")] = {"psnr": psnr_fista, "time": time_fista}

    # Save FISTA results
    save_image(
        img_recon_fista,
        os.path.join(
            RESULTS_DIR,
            "images",
            f"recon_fista_{MEASUREMENT_TYPE}_{rate * 100:.0f}.png",
        ),
    )
    fig_fista_recon = plot_results(
        img_true,
        img_recon_fista,
        MEASUREMENT_TYPE,
        rate,
        "FISTA",
        psnr_fista,
        time_fista,
    )
    fig_fista_recon.savefig(
        os.path.join(
            RESULTS_DIR, "plots", f"recon_fista_{MEASUREMENT_TYPE}_{rate * 100:.0f}.png"
        )
    )
    plt.close(fig_fista_recon)
    fig_fista_conv = plot_convergence(obj_hist_fista, "FISTA")
    fig_fista_conv.savefig(
        os.path.join(
            RESULTS_DIR, "plots", f"conv_fista_{MEASUREMENT_TYPE}_{rate * 100:.0f}.png"
        )
    )
    plt.close(fig_fista_conv)

    # Plot combined convergence
    fig_conv_comp, ax = plt.subplots(figsize=(8, 5))
    ax.plot(obj_hist_ista, label=f"ISTA (Final: {obj_hist_ista[-1]:.4e})")
    ax.plot(obj_hist_fista, label=f"FISTA (Final: {obj_hist_fista[-1]:.4e})")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective Function Value (log scale)")
    ax.set_yscale("log")
    ax.set_title(f"Convergence Comparison ({MEASUREMENT_TYPE}, {rate * 100:.0f}%)")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)  # Grid on log scale
    plt.tight_layout()
    fig_conv_comp.savefig(
        os.path.join(
            RESULTS_DIR,
            "plots",
            f"conv_compare_{MEASUREMENT_TYPE}_{rate * 100:.0f}.png",
        )
    )
    plt.close(fig_conv_comp)


# Print Summary
print("\n--- Results Summary")
print("Rate\tAlgo\tMeasure\tLambda\tPSNR (dB)\tTime (s)")
for res in results_summary:
    print(
        f"{res['rate'] * 100:.0f}%\t{res['algo']}\t{res['measurement']}\t{res['lambda']:.2e}\t{res['psnr']:.2f}\t\t{res['time']:.2f}"
    )

print(f"\nResults saved in: {RESULTS_DIR}")

# Generate Combined Summary Plot
print("\nGenerating combined summary plot...")

# Generate the plot using the collected data
num_rates = len(SAMPLING_RATES)
num_algos = 2  # ISTA and FISTA
algo_names = ["ISTA", "FISTA"]

# Create figure: 1 row for original + num_rates rows for results
# num_algos columns for reconstructions
fig_summary, axes = plt.subplots(
    num_rates + 1,
    num_algos,
    figsize=(5 * num_algos, 5 * (num_rates + 1)),
    squeeze=False,  # Ensure axes is always 2D array
)

# Plot Original Image
# Span the original image across the top row
gs = axes[0, 0].get_gridspec()
# Remove the underlying axes
for ax in axes[0, :]:
    ax.remove()
# Create a new axis spanning the first row
ax_orig = fig_summary.add_subplot(gs[0, :])
ax_orig.imshow(img_true, cmap="gray")
ax_orig.set_title(f"Original Image ({IMAGE_SIZE}x{IMAGE_SIZE})", fontsize=14)
ax_orig.axis("off")

# Plot Reconstructions
for i, rate in enumerate(SAMPLING_RATES):
    for j, algo_name in enumerate(algo_names):
        ax = axes[i + 1, j]  # Get the correct subplot axis (skip first row)
        img_recon = reconstructions_dict.get((rate, algo_name))
        result_metrics = results_dict.get((rate, algo_name))

        if img_recon is not None and result_metrics is not None:
            psnr = result_metrics["psnr"]
            time_taken = result_metrics["time"]
            ax.imshow(img_recon, cmap="gray", vmin=0, vmax=1)  # Ensure consistent range
            title = (
                f"{algo_name} ({rate * 100:.0f}% Sampling)\n"
                f"PSNR: {psnr:.2f} dB\nTime: {time_taken:.2f}s"
            )
            ax.set_title(title, fontsize=10)
        else:
            ax.text(
                0.5,
                0.5,
                "N/A",
                horizontalalignment="center",
                verticalalignment="center",
            )
            ax.set_title(f"{algo_name} ({rate * 100:.0f}%) - Error?", fontsize=10)

        ax.axis("off")

# Final Touches
fig_summary.suptitle(
    f"Compressed Sensing Reconstruction Summary\n"
    f"Measurement: {MEASUREMENT_TYPE.capitalize()}, Wavelet: {WAVELET_NAME}, Lambda: {LAMBDA_REG}",
    fontsize=16,
    y=0.99,
)  # Adjust y to prevent overlap

plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust layout to make room for suptitle

# Save the summary plot
summary_plot_path = os.path.join(
    RESULTS_DIR, "plots", f"summary_reconstructions_{MEASUREMENT_TYPE}.png"
)
fig_summary.savefig(summary_plot_path, dpi=150)  # Use decent DPI
print(f"Combined summary plot saved to: {summary_plot_path}")
plt.close(fig_summary)  # Close the figure to free memory
