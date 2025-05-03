import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio


def load_image(image_path, size=None):
    """Loads a grayscale image, resizes it, and normalizes to [0, 1]."""
    img = Image.open(image_path).convert("L")  # Convert to grayscale
    if size:
        if isinstance(size, int):
            size = (size, size)
        img = img.resize(size, Image.Resampling.LANCZOS)
    img_array = np.array(img, dtype=np.float64)
    img_array /= 255.0  # Normalize to [0, 1]
    return img_array


def calculate_psnr(img_true, img_recon, data_range=1.0):
    """Calculates the Peak Signal-to-Noise Ratio."""
    if img_true.shape != img_recon.shape:
        raise ValueError("Input images must have the same dimensions.")
    if img_true.dtype != img_recon.dtype:
        # Promote to common dtype if necessary, float64 is safe
        img_recon = img_recon.astype(img_true.dtype)

    # Ensure data range matches normalization (0-1 in our case)
    return peak_signal_noise_ratio(img_true, img_recon, data_range=data_range)


def plot_results(
    original,
    reconstructed,
    measurement_type,
    sampling_rate,
    algorithm_name,
    psnr,
    time_taken,
):
    """Plots the original and reconstructed images."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(reconstructed, cmap="gray")
    axes[1].set_title(
        f"{algorithm_name} Reconstruction\n"
        f"({measurement_type}, {sampling_rate * 100:.0f}% sampling)\n"
        f"PSNR: {psnr:.2f} dB, Time: {time_taken:.2f}s"
    )
    axes[1].axis("off")

    plt.tight_layout()
    return fig


def plot_convergence(objective_history, algorithm_name):
    """Plots the objective function value vs. iterations."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(objective_history)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective Function Value")
    ax.set_title(f"{algorithm_name} Convergence")
    ax.grid(True)
    plt.tight_layout()
    return fig


def save_image(image_array, path):
    """Saves a numpy array as a grayscale image."""
    img = Image.fromarray(np.clip(image_array * 255, 0, 255).astype(np.uint8))
    img.save(path)
