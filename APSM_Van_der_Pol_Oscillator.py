# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import expm
import os



def gradient_descent_update(A_k, x_k, y_k, learning_rate):
    """
    Gradient descent update for the system matrix A_k, 
    using a single state vector x_k and target output y_k.

    Parameters
    ----------
    A_k : numpy.ndarray
        Current system matrix to be updated.
    x_k : numpy.ndarray
        State vector (column vector).
    y_k : numpy.ndarray
        Target output vector.
    learning_rate : float
        Learning rate controlling the update step size.

    Returns
    -------
    A_k : numpy.ndarray
        Updated system matrix after one gradient descent step.
    """
    # Compute the prediction error (residual)
    error = y_k - A_k.dot(x_k)

    # Compute gradient of the loss function with respect to A_k
    grad_A_k = -2 * error * x_k.T

    # Update A_k using gradient descent
    A_k -= learning_rate * grad_A_k

    return A_k



def bilinear_continuous_to_discrete_A(A_c, dt):
    """
    Convert a continuous-time system matrix A_c to its discrete-time equivalent A_d
    using the bilinear transformation (Tustin's method).

    Parameters
    ----------
    A_c : numpy.ndarray
        Continuous-time system matrix.
    dt : float
        Discretization time step.

    Returns
    -------
    A_d : numpy.ndarray
        Discrete-time system matrix obtained via bilinear transformation.
    """
    # Identity matrix with the same size as A_c
    I = np.eye(A_c.shape[0])

    # Apply bilinear (Tustin) transformation:
    # A_d = (I - (dt/2) * A_c)^(-1) * (I + (dt/2) * A_c)
    A_d = np.linalg.inv(I - (dt/2) * A_c).dot(I + (dt/2) * A_c)

    return A_d



def evaluate_predictions(actual, predicted):
    """
    Evaluate the prediction quality of a model using MSE, NMSE, and R^2 metrics.
    Handles complex-valued inputs by using only their real parts.

    Parameters:
    - actual (np.ndarray): The ground truth data, potentially complex-valued.
    - predicted (np.ndarray): The predicted output data, potentially complex-valued.

    Returns:
    - mse (float): Mean Squared Error between actual and predicted values.
    - nmse (float): Normalized Mean Squared Error (MSE divided by variance of actual values).
    - r2 (float): Coefficient of Determination (R^2 score).
    """
    # Convert to real-valued arrays (if complex)
    actual_real = np.real(actual)
    predicted_real = np.real(predicted)

    # Compute Mean Squared Error (MSE)
    mse = mean_squared_error(actual_real, predicted_real)

    # Compute Normalized MSE (NMSE)
    variance_actual = np.var(actual_real)
    nmse = mse / variance_actual

    # Compute R^2 score
    r2 = r2_score(actual_real, predicted_real)

    return mse, nmse, r2




def bilinear_discrete_to_continuous_A(A_d, dt):
    """
    Recovers the continuous dynamic system matrices from the discrete ones using the Bilinear Transformation method.
    
    Args:
    Ad (ndarray): Discrete system matrix.
    Bd (ndarray): Discrete input matrix.
    T (float): Sampling time used for discretization.
    
    Returns:
    A (ndarray): Continuous system matrix.
    B (ndarray): Continuous input matrix.
    """
    n = A_d.shape[0]  # Assuming Ad is square and represents the state dimension
    I = np.eye(n)
    
    # Bilinear transformation inverse
    A = (2/dt) * np.linalg.inv(A_d + I) @ (A_d-I)
    return A



def jacobian(x_k, t):
    """
    Compute the Jacobian matrix for the modified nonlinear system.
    
    Parameters:
    ----------
    x_k : numpy.ndarray
        State vector [x1, x2] (can be column vector)
    t : float
        Current time for time-varying terms
    
    Returns:
    -------
    numpy.ndarray
        2x2 Jacobian matrix
    """
    # Extract states
    x1, x2 = x_k.flatten()  # ensure 1D

    # Compute matrix elements
    J = np.array([
        [0, 1],
        [-2.4 * x1 * x2 - (1 - 0.15 * np.sin(0.3 * t)), 1.2 * (1 - x1**2)]
    ])

    return J

def frobenius_norm(A, B=None):
    """
    Calculate the Frobenius norm of one or two matrices.
    
    Parameters:
    A (numpy.ndarray): First matrix
    B (numpy.ndarray, optional): Second matrix (default is None)
    
    Returns:
    tuple: Frobenius norm of A, and if B is provided, Frobenius norm of A + B
    """
    # Compute Frobenius norm of A
    frobenius_norm_A = np.linalg.norm(A, 'fro')
    
    # If B is provided, compute Frobenius norm of A + B
    if B is not None:
        frobenius_norm_A_plus_B = np.linalg.norm(A + B, 'fro')
        return frobenius_norm_A, frobenius_norm_A_plus_B
    else:
        return frobenius_norm_A

def bilinear_continuous_to_discrete(A_cont, delta_t):
    """
    Convert continuous-time state-space matrices to discrete-time using the bilinear (Tustin) transform.

    Parameters:
    - A_cont (np.ndarray): Continuous-time system matrix A
    - B_cont (np.ndarray): Continuous-time input matrix B
    - delta_t (float): Sampling time interval

    Returns:
    - A_d (np.ndarray): Discrete-time system matrix (Phi)
    - B_d (np.ndarray): Discrete-time input matrix (Gamma)
    """
    I = np.eye(A_cont.shape[0])  # Identity matrix of the same size as A_cont

    # Discrete-time A matrix using the bilinear (Tustin) transformation
    A_d = np.linalg.inv(I - (delta_t / 2) * A_cont) @ (I + (delta_t / 2) * A_cont)


    return A_d


def APSM_grad(combined_matrix_noise, A_initial, learning_rate, iterations, dt=0.005):
    """
    Adaptive Physics-based Structured Model (APSM) calibration with gradient descent.

    This function updates the system matrix A_k iteratively using noisy state data,
    compares it with the discretized Jacobian at each step, and records the deviation.

    Parameters
    ----------
    combined_matrix_noise : np.ndarray
        Noisy state trajectory matrix, each column is a state vector at a time step.
    A_initial : np.ndarray
        Initial system matrix A_k.
    learning_rate : float
        Gradient descent learning rate.
    iterations : int
        Number of gradient descent steps (should be time steps - 1).
    dt : float, default=0.005
        Sampling interval.

    Returns
    -------
    A_k : np.ndarray
        Updated system matrix after gradient descent.
    y_estimates : np.ndarray
        Predicted state vectors over time (shape: n × iterations).
    Frobenius_norm : list of float
        Frobenius norm of (J_k_discrete - A_k) at each iteration.
    """
    n = combined_matrix_noise.shape[0]
    A_k = A_initial

    y_estimates = np.zeros((n, iterations))
    Frobenius_norm = []

    for idx in range(iterations):
        # Current time
        t = idx * dt  

        # Current state and the next state
        x_k = combined_matrix_noise[:, idx:idx+1]    # shape (n, 1)
        y_k = combined_matrix_noise[:, idx+1:idx+2]  # shape (n, 1)

        # Compute Jacobian and its discrete equivalent
        J_k = jacobian(x_k, t)  
        J_k_discrete = bilinear_continuous_to_discrete(J_k, dt)

        # Update A_k using gradient descent
        A_k = gradient_descent_update(A_k, x_k, y_k, learning_rate)

        # Predict the next state
        y_k_pred = A_k @ x_k

        # Save predicted state and Frobenius norm
        y_estimates[:, idx] = y_k_pred.flatten()
        Frobenius_norm.append(frobenius_norm(J_k_discrete - A_k))

    return A_k, y_estimates, Frobenius_norm




def add_nonstationary_gaussian_noise(signal, noise_ratio):
    """
    Add non-stationary Gaussian noise to a signal. The noise added to each sample is proportional
    to the magnitude of the signal at that point.

    Parameters:
    - signal (np.ndarray): The original signal.
    - noise_ratio (float): The ratio of the noise amplitude to the signal amplitude.

    Returns:
    - noisy_signal (np.ndarray): Signal with added non-stationary Gaussian noise.
    """
    # Calculate noise standard deviation for each sample
    noise_std_per_sample = np.abs(signal) * noise_ratio

    # Generate non-stationary Gaussian noise
    noise = noise_std_per_sample * np.random.normal(0, 1, signal.shape)

    # Add noise to the original signal
    noisy_signal = signal + noise
    return noisy_signal



def plot_frobenius_norm_vs_time(Frobenius_norms, sample_interval=0.005, save_path=None):
    """
    Plot the Frobenius Norm as a function of time.

    Parameters
    ----------
    Frobenius_norms : list or np.ndarray
        Sequence of Frobenius norms.
    sample_interval : float, default=0.005
        Sampling interval (seconds).
    save_path : str or None
        Directory to save the PDF plot. If None, the plot is only displayed.
    """
    # Set global font to Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'

    # Time vector
    time = np.arange(0, len(Frobenius_norms) * sample_interval, sample_interval)

    # Plot curve
    plt.figure(figsize=(8, 6))
    plt.plot(time, Frobenius_norms, 'b-', linewidth=2)
    plt.xlabel('Time (s)', fontsize=20)
    plt.ylabel('Frobenius Norm', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save as PDF if path is provided
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        filename = os.path.join(save_path, "Frobenius_norm_plot.pdf")
        plt.savefig(filename, format='pdf', bbox_inches='tight')
        print(f"Plot saved at {filename}")

    plt.show()


# Set random seed for reproducibility
np.random.seed(0)

# Load state data from the current script directory
state_load_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state_data.npy")
X = np.load(state_load_path).T

# Learning rate for gradient-based optimization
learning_rate_grad = 0.005  

# Sampling time step
dt = 0.005  

# Add non-stationary Gaussian noise to the training data
noise_ratio = 0  # Set > 0 to inject noise
X_noisy = add_nonstationary_gaussian_noise(X, noise_ratio)

# Define prediction horizon
k_steps = 0  # Number of prediction steps
x_0 = X[:, -1 - k_steps]  # Use the (k+1)-th last time point as the initial state

# Split dataset into training and testing segments
X_train = X_noisy[:, :-1 - k_steps]   # Training states (noisy)
X_test = X[:, -k_steps:]              # Ground-truth test states
X_test_noisy = X_noisy[:, -k_steps:]  # Noisy test states (optional)

# Number of gradient descent iterations
iterations = X_train.shape[1] - 1

# ==============================================================
# Case 1: Initialize A with a zero matrix (assuming B is known)
# ==============================================================

A_initial_grad_0 = np.zeros((2, 2))

A_k_pure_grad_0, X_pure_pred_0, Frobenius_norms_grad_0 = APSM_grad(
    X_train, A_initial_grad_0, learning_rate_grad, iterations
)

# Plot Frobenius norm between A_k and discretized Jacobian over time
plot_frobenius_norm_vs_time(Frobenius_norms_grad_0, save_path=None)

# ==============================================================
# Case 2: Initialize A with the Jacobian matrix at the first state
# ==============================================================

x_0 = X_train[:, 0:1]
J_k = jacobian(x_0, 0)

A_initial_grad_J = bilinear_continuous_to_discrete_A(J_k, dt)

A_k_pure_grad_J, X_pure_pred_J, Frobenius_norms_grad_J = APSM_grad(
    X_train, A_initial_grad_J, learning_rate_grad, iterations
)

# Plot Frobenius norm for the Jacobian-initialized case
plot_frobenius_norm_vs_time(Frobenius_norms_grad_J, save_path=None)








































