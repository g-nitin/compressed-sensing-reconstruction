import time

import numpy as np
from tqdm import tqdm  # Progress bar


def soft_threshold(x, threshold):
    """Soft thresholding operator."""
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


def calculate_objective(A_op, Psi_op, s, y, lambda_reg):
    """Calculates the LASSO objective function value."""
    residual = A_op.forward(Psi_op.forward(s)) - y
    smooth_term = 0.5 * np.linalg.norm(residual) ** 2
    sparse_term = lambda_reg * np.linalg.norm(s, 1)
    return smooth_term + sparse_term


def ista(
    A_op,
    Psi_op,
    y,
    lambda_reg,
    max_iter=100,
    tol=1e-4,
    step_size=None,
    use_backtracking=True,
    eta=0.8,
    verbose=True,
):
    """
    Iterative Shrinkage-Thresholding Algorithm (ISTA) for LASSO:
    min_s (1/2) ||A@Psi@s - y||^2 + lambda ||s||_1
    """
    start_time = time.time()

    # Initialization
    s = np.zeros(Psi_op.coeffs_vec_len)  # Start with zero coefficients
    objective_history = []
    times_history = []

    # Determine step size (Lipschitz constant estimation or backtracking)
    if step_size is None and not use_backtracking:
        # Estimate Lipschitz constant L using power iteration (can be slow)
        # Or use a safe guess (e.g., 1.0, but might diverge)
        # For simplicity, we'll rely on backtracking or a user-provided guess
        print(
            "Warning: No step_size provided and backtracking disabled. Using default step_size=1.0, which might be unstable."
        )
        step_size = 1.0  # Potentially unstable default

    current_step_size = (
        step_size if step_size is not None else 1.0
    )  # Initial guess for backtracking

    print(f"--- Running ISTA (lambda={lambda_reg}) ---")
    pbar = tqdm(range(max_iter), disable=not verbose)
    for k in pbar:
        s_old = s.copy()

        # Gradient calculation: grad = Psi.H @ A.H @ (A@Psi@s - y)
        residual = A_op.forward(Psi_op.forward(s)) - y
        grad = Psi_op.adjoint(A_op.adjoint(residual))

        # Backtracking line search for step size (optional but recommended)
        if use_backtracking:
            current_step_size *= 1.1  # Cautious increase from previous iter
            f_s = 0.5 * np.linalg.norm(residual) ** 2  # Smooth part value at s
            while True:
                s_update_trial = s - current_step_size * grad
                prox_update = soft_threshold(
                    s_update_trial, current_step_size * lambda_reg
                )
                # Check condition: f(prox_update) <= f(s) + <grad, prox_update - s> + (1/(2*step))||prox_update - s||^2
                residual_prox = A_op.forward(Psi_op.forward(prox_update)) - y
                f_prox = 0.5 * np.linalg.norm(residual_prox) ** 2
                rhs = (
                    f_s
                    + np.dot(grad, prox_update - s)
                    + (0.5 / current_step_size) * np.linalg.norm(prox_update - s) ** 2
                )

                if f_prox <= rhs:
                    break  # Step size is accepted
                current_step_size *= eta  # Reduce step size
                if current_step_size < 1e-10:  # Avoid infinite loop
                    print("Warning: Backtracking step size became too small.")
                    break
            step_size = (
                current_step_size  # Use the accepted step size for this iteration
            )
        else:  # Fixed step size
            s_update_trial = s - step_size * grad
            prox_update = soft_threshold(s_update_trial, step_size * lambda_reg)

        # Update step
        s = prox_update

        # Calculate objective and store history
        obj = calculate_objective(A_op, Psi_op, s, y, lambda_reg)
        objective_history.append(obj)
        times_history.append(time.time() - start_time)

        # Check convergence
        diff = np.linalg.norm(s - s_old) / (np.linalg.norm(s_old) + 1e-10)
        if verbose:
            pbar.set_description(
                f"Iter {k + 1}/{max_iter}, Obj: {obj:.4e}, Diff: {diff:.4e}, Step: {step_size:.2e}"
            )

        if diff < tol:
            print(f"\nISTA converged after {k + 1} iterations.")
            break
    else:  # Loop finished without break
        print(f"\nISTA reached max iterations ({max_iter}).")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"ISTA finished in {total_time:.2f} seconds.")

    # Reconstruct image from final coefficients
    x_recon_vec = Psi_op.forward(s)
    x_recon = x_recon_vec.reshape(Psi_op.img_shape)

    return x_recon, s, objective_history, times_history, total_time


def fista(
    A_op,
    Psi_op,
    y,
    lambda_reg,
    max_iter=100,
    tol=1e-4,
    step_size=None,
    use_backtracking=True,
    eta=0.8,
    verbose=True,
):
    """
    Fast Iterative Shrinkage-Thresholding Algorithm (FISTA) for LASSO:
    min_s (1/2) ||A@Psi@s - y||^2 + lambda ||s||_1
    """
    start_time = time.time()

    # Initialization
    s = np.zeros(Psi_op.coeffs_vec_len)  # Primal variable
    z = s.copy()  # Extrapolation sequence
    t = 1.0  # Momentum term
    objective_history = []
    times_history = []

    # Determine step size (Lipschitz constant estimation or backtracking)
    if step_size is None and not use_backtracking:
        print(
            "Warning: No step_size provided and backtracking disabled. Using default step_size=1.0, which might be unstable."
        )
        step_size = 1.0  # Potentially unstable default

    current_step_size = (
        step_size if step_size is not None else 1.0
    )  # Initial guess for backtracking

    print(f"--- Running FISTA (lambda={lambda_reg}) ---")
    pbar = tqdm(range(max_iter), disable=not verbose)
    for k in pbar:
        s_old = s.copy()

        # Gradient calculation at extrapolated point z: grad = Psi.H @ A.H @ (A@Psi@z - y)
        residual_z = A_op.forward(Psi_op.forward(z)) - y
        grad_z = Psi_op.adjoint(A_op.adjoint(residual_z))

        # Backtracking line search for step size (optional but recommended)
        # Note: FISTA backtracking condition is slightly different
        if use_backtracking:
            current_step_size *= 1.1  # Cautious increase
            f_z = 0.5 * np.linalg.norm(residual_z) ** 2  # Smooth part value at z
            while True:
                s_update_trial = z - current_step_size * grad_z
                prox_update = soft_threshold(
                    s_update_trial, current_step_size * lambda_reg
                )
                # Check condition: f(prox_update) <= f(z) + <grad_z, prox_update - z> + (1/(2*step))||prox_update - z||^2
                residual_prox = A_op.forward(Psi_op.forward(prox_update)) - y
                f_prox = 0.5 * np.linalg.norm(residual_prox) ** 2
                rhs = (
                    f_z
                    + np.dot(grad_z, prox_update - z)
                    + (0.5 / current_step_size) * np.linalg.norm(prox_update - z) ** 2
                )

                if f_prox <= rhs:
                    break  # Step size is accepted
                current_step_size *= eta  # Reduce step size
                if current_step_size < 1e-10:
                    print("Warning: Backtracking step size became too small.")
                    break
            step_size = current_step_size  # Use the accepted step size
        else:  # Fixed step size
            s_update_trial = z - step_size * grad_z
            prox_update = soft_threshold(s_update_trial, step_size * lambda_reg)

        # Update steps
        s = prox_update
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t**2)) / 2.0
        z = s + ((t - 1.0) / t_new) * (s - s_old)
        t = t_new

        # Calculate objective (at s, not z) and store history
        obj = calculate_objective(A_op, Psi_op, s, y, lambda_reg)
        objective_history.append(obj)
        times_history.append(time.time() - start_time)

        # Check convergence (on s)
        diff = np.linalg.norm(s - s_old) / (np.linalg.norm(s_old) + 1e-10)
        if verbose:
            pbar.set_description(
                f"Iter {k + 1}/{max_iter}, Obj: {obj:.4e}, Diff: {diff:.4e}, Step: {step_size:.2e}"
            )

        if diff < tol and k > 1:  # Check diff after first few iters
            print(f"\nFISTA converged after {k + 1} iterations.")
            break
    else:  # Loop finished without break
        print(f"\nFISTA reached max iterations ({max_iter}).")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"FISTA finished in {total_time:.2f} seconds.")

    # Reconstruct image from final coefficients (s)
    x_recon_vec = Psi_op.forward(s)
    x_recon = x_recon_vec.reshape(Psi_op.img_shape)

    return x_recon, s, objective_history, times_history, total_time
