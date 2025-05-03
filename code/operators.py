from abc import ABC, abstractmethod

import numpy as np
import pywt
from scipy.fft import fft2, ifft2


# Base Classes
class LinearOperator(ABC):
    """Abstract base class for linear operators."""

    @abstractmethod
    def forward(self, x):
        """Applies the operator: A @ x"""
        pass

    @abstractmethod
    def adjoint(self, y):
        """Applies the adjoint (conjugate transpose) operator: A.H @ y"""
        pass

    def __matmul__(self, x):
        return self.forward(x)


# Measurement Operators
class GaussianMeasurementOperator(LinearOperator):
    """Random Gaussian measurement operator."""

    def __init__(self, num_measurements, signal_size):
        self.m = num_measurements
        self.n = signal_size
        # Generate matrix once for reproducibility within an instance
        self.matrix = np.random.randn(self.m, self.n) / np.sqrt(
            self.m
        )  # Normalize rows

    def forward(self, x):
        # Assumes x is a flattened vector
        if x.shape != (self.n,):
            x = x.ravel()
        if x.shape != (self.n,):
            raise ValueError(
                f"Input vector shape mismatch. Expected ({self.n},), got {x.shape}"
            )
        return self.matrix @ x

    def adjoint(self, y):
        # Assumes y is the measurement vector
        if y.shape != (self.m,):
            raise ValueError(
                f"Measurement vector shape mismatch. Expected ({self.m},), got {y.shape}"
            )
        return self.matrix.T @ y  # Adjoint for real matrix is transpose


class PartialFourierMeasurementOperator(LinearOperator):
    """Partial 2D Fourier measurement operator."""

    def __init__(self, num_measurements, img_shape):
        self.m = num_measurements
        self.img_shape = img_shape
        self.n = np.prod(img_shape)

        if self.m > self.n:
            raise ValueError(
                "Number of measurements cannot exceed total number of pixels."
            )

        # Randomly select frequency indices
        # Ensure indices are within the valid range for fft2 output
        indices = np.random.choice(self.n, self.m, replace=False)
        self.mask = np.zeros(self.n, dtype=bool)
        self.mask[indices] = True
        self.mask_2d = self.mask.reshape(self.img_shape)

    def forward(self, x):
        # Assumes x is a flattened image vector
        if x.shape != (self.n,):
            x = x.ravel()
        if x.shape != (self.n,):
            raise ValueError(
                f"Input vector shape mismatch. Expected ({self.n},), got {x.shape}"
            )

        img = x.reshape(self.img_shape)
        fft_coeffs = fft2(img, norm="ortho")  # Use orthonormal FFT
        # Select measurements based on the mask
        measured_coeffs = fft_coeffs[self.mask_2d]
        return measured_coeffs.ravel()  # Return as a flat vector

    def adjoint(self, y):
        # Assumes y is the measurement vector (subset of FFT coeffs)
        if y.shape != (self.m,):
            raise ValueError(
                f"Measurement vector shape mismatch. Expected ({self.m},), got {y.shape}"
            )

        # Place measured coefficients into a full FFT grid
        full_coeffs = np.zeros(self.img_shape, dtype=np.complex128)
        full_coeffs[self.mask_2d] = y.reshape(-1)  # Reshape y just in case

        # Inverse FFT to get back to image domain
        img_recon = ifft2(full_coeffs, norm="ortho")  # Use orthonormal IFFT

        # Return the real part as a flattened vector
        # (assuming original signal was real)
        return img_recon.real.ravel()


# Transform Operator
class WaveletOperator(LinearOperator):
    """
    2D Wavelet Transform operator using PyWavelets.
    Uses wavedec2/waverec2 with a specified mode.
    Implements MANUAL vectorization/de-vectorization.
    """

    def __init__(self, img_shape, wavelet="db4", level=None, mode="symmetric"):
        self.img_shape = img_shape
        self.n_pixels = np.prod(img_shape)
        self.wavelet = wavelet
        self.mode = mode

        # Perform a dummy transform to determine the structure and total size
        dummy_img = np.zeros(img_shape)
        try:
            coeffs_structure = pywt.wavedec2(
                dummy_img, wavelet=self.wavelet, level=level, mode=self.mode
            )
            # Store the determined level if it was None
            self.level = len(coeffs_structure) - 1
        except ValueError as e:
            print(f"ERROR during pywt.wavedec2 in __init__ with mode='{self.mode}'.")
            print(f"  Image shape: {img_shape}, Wavelet: {wavelet}, Level: {level}")
            raise e

        # Manually calculate total coefficient count and store shapes/slices info
        self.coeffs_shapes = []
        self.coeffs_slices = []
        current_pos = 0
        # Approximation coefficients
        app_shape = coeffs_structure[0].shape
        app_size = coeffs_structure[0].size
        self.coeffs_shapes.append(app_shape)
        self.coeffs_slices.append(slice(current_pos, current_pos + app_size))
        current_pos += app_size

        # Detail coefficients
        for detail_level in coeffs_structure[1:]:
            level_shapes = []
            level_slices = []
            # Details are usually (cH, cV, cD) tuples
            if isinstance(detail_level, tuple):
                for detail_coeffs in detail_level:
                    shape = detail_coeffs.shape
                    size = detail_coeffs.size
                    level_shapes.append(shape)
                    level_slices.append(slice(current_pos, current_pos + size))
                    current_pos += size
            elif isinstance(detail_level, dict):  # Handle dict format if necessary
                for key in sorted(detail_level.keys()):  # Ensure consistent order
                    detail_coeffs = detail_level[key]
                    shape = detail_coeffs.shape
                    size = detail_coeffs.size
                    level_shapes.append(shape)
                    level_slices.append(slice(current_pos, current_pos + size))
                    current_pos += size
            else:
                raise TypeError(
                    f"Unexpected detail coefficient format: {type(detail_level)}"
                )

            self.coeffs_shapes.append(
                tuple(level_shapes)
            )  # Store shapes for this level
            self.coeffs_slices.append(
                tuple(level_slices)
            )  # Store slices for this level

        self.coeffs_vec_len = current_pos  # Total length of the flattened vector

        # Sanity check: total length should match pixel count for orthogonal wavelets
        if not pywt.Wavelet(self.wavelet).orthogonal:
            print(
                f"Warning: Wavelet '{self.wavelet}' is not orthogonal. Coefficient count ({self.coeffs_vec_len}) might not equal pixel count ({self.n_pixels})."
            )
        elif self.coeffs_vec_len != self.n_pixels:
            print(
                f"WARNING in __init__: Manual coefficient count ({self.coeffs_vec_len}) != pixel count ({self.n_pixels}) for mode='{self.mode}'. Check decomposition logic."
            )
        else:
            print(
                f"WaveletOperator: Manual vectorization setup complete. Mode='{self.mode}'. Coefficient vector length: {self.coeffs_vec_len}"
            )

    def _coeffs_to_vector(self, coeffs_structure):
        """Manually flattens coefficient structure to a vector."""
        vector = np.zeros(self.coeffs_vec_len, dtype=coeffs_structure[0].dtype)
        # Approximation
        vector[self.coeffs_slices[0]] = coeffs_structure[0].ravel()
        # Details
        for i, detail_level_slices in enumerate(self.coeffs_slices[1:]):
            level_coeffs = coeffs_structure[i + 1]  # Get coeffs for this level
            if isinstance(level_coeffs, tuple):
                for j, detail_slice in enumerate(detail_level_slices):
                    vector[detail_slice] = level_coeffs[j].ravel()
            elif isinstance(level_coeffs, dict):
                # Assuming keys were sorted during slice creation
                keys = sorted(level_coeffs.keys())
                for j, detail_slice in enumerate(detail_level_slices):
                    vector[detail_slice] = level_coeffs[keys[j]].ravel()
        return vector

    def _vector_to_coeffs(self, vector):
        """Manually reconstructs coefficient structure from a vector."""
        coeffs_structure = []
        # Approximation
        app_coeffs = vector[self.coeffs_slices[0]].reshape(self.coeffs_shapes[0])
        coeffs_structure.append(app_coeffs)
        # Details
        for i, detail_level_slices in enumerate(self.coeffs_slices[1:]):
            level_shapes = self.coeffs_shapes[i + 1]
            reconstructed_details = []
            if isinstance(
                detail_level_slices, tuple
            ):  # Should correspond to tuple of shapes
                for j, detail_slice in enumerate(detail_level_slices):
                    detail_coeffs = vector[detail_slice].reshape(level_shapes[j])
                    reconstructed_details.append(detail_coeffs)
                coeffs_structure.append(tuple(reconstructed_details))
            else:
                # Adapt if dict structure was stored differently
                raise NotImplementedError(
                    "Dict structure reconstruction not fully implemented if needed"
                )

        return coeffs_structure

    def forward(self, s_vec):
        """Inverse Wavelet Transform (Synthesis): Psi @ s"""
        if not isinstance(s_vec, np.ndarray):
            s_vec = np.asarray(s_vec)
        if s_vec.size != self.coeffs_vec_len:
            raise ValueError(
                f"Coeff vector size mismatch in forward(). Expected {self.coeffs_vec_len}, got {s_vec.size}"
            )

        # Reconstruct structure from vector
        coeffs_recon = self._vector_to_coeffs(s_vec.reshape(self.coeffs_vec_len))

        # Perform inverse transform
        try:
            img_recon = pywt.waverec2(
                coeffs_recon, wavelet=self.wavelet, mode=self.mode
            )
        except Exception as e:
            print("\nERROR: Unexpected error during pywt.waverec2 in forward()!")
            print(f"  Mode: {self.mode}")
            raise e

        # Ensure output image has the correct shape
        h, w = self.img_shape
        h_recon, w_recon = img_recon.shape
        if h_recon != h or w_recon != w:
            img_recon_adjusted = np.zeros(self.img_shape, dtype=img_recon.dtype)
            min_h = min(h, h_recon)
            min_w = min(w, w_recon)
            img_recon_adjusted[:min_h, :min_w] = img_recon[:min_h, :min_w]
            img_recon = img_recon_adjusted

        return img_recon.ravel()

    def adjoint(self, x_vec):
        """Forward Wavelet Transform (Analysis): Psi.H @ x"""
        if not isinstance(x_vec, np.ndarray):
            x_vec = np.asarray(x_vec)
        if x_vec.size != self.n_pixels:
            raise ValueError(
                f"Image vector size mismatch in adjoint(). Expected {self.n_pixels}, got {x_vec.size}"
            )

        img = x_vec.reshape(self.img_shape)

        # Perform forward transform
        try:
            coeffs_structure = pywt.wavedec2(
                img, wavelet=self.wavelet, level=self.level, mode=self.mode
            )
        except ValueError as e:
            print(f"ERROR during pywt.wavedec2 in adjoint() with mode='{self.mode}'.")
            raise e

        # Flatten structure to vector
        s_vec = self._coeffs_to_vector(coeffs_structure)

        # Ensure correct length (should be guaranteed by _coeffs_to_vector)
        if len(s_vec) != self.coeffs_vec_len:
            # This should ideally not happen with manual vectorization
            print(
                f"CRITICAL WARNING in adjoint(): Manual vectorization length ({len(s_vec)}) != expected ({self.coeffs_vec_len})."
            )
            # Attempt to fix, but indicates a bug in _coeffs_to_vector or __init__
            if len(s_vec) > self.coeffs_vec_len:
                s_vec = s_vec[: self.coeffs_vec_len]
            else:
                s_vec_padded = np.zeros(self.coeffs_vec_len, dtype=s_vec.dtype)
                s_vec_padded[: len(s_vec)] = s_vec
                s_vec = s_vec_padded

        return s_vec
