import os
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Core: Temperature/Stiffness & Van der Pol (VdP) Dynamics
# =========================
def temperature(t, T0, DeltaT, Omega_T):
    return T0 + DeltaT * np.sin(Omega_T * t)

def stiffness_from_T(T, k0, T0, alpha_T):
    return k0 * (1.0 - alpha_T * (T - T0))

def damping_from_x1(mu, x1):
    """Nonlinear damping coefficient c(t, x1)"""
    return mu * (1 - x1**2)

def vdp_dynamics(state, t, mu, m, k0, T0, DeltaT, Omega_T, alpha_T):
    x1, x2 = state
    T_t = temperature(t, T0, DeltaT, Omega_T)
    k_t = stiffness_from_T(T_t, k0, T0, alpha_T)
    dx1 = x2
    dx2 = mu * (1.0 - x1**2) * x2 - (k_t / m) * x1
    return np.array([dx1, dx2], dtype=float)

def simulate_vdp(mu=1.2, k0=1.0, m=1.0, T0=0.0, DeltaT=0.5, Omega_T=0.02, alpha_T=0.3,
                 dt=0.01, t_end=200.0, x10=1.0, x20=0.0):
    """Fourth-order RK integration, return time_span, sol(x1,x2), T(t), k(t), c(t)"""
    time_span = np.arange(0.0, t_end + dt, dt)
    n = len(time_span)
    sol = np.zeros((n, 2))
    T_t = np.zeros(n)
    k_t = np.zeros(n)
    c_t = np.zeros(n)

    # Initial condition
    sol[0, 0], sol[0, 1] = x10, x20

    # RK4 integration
    for i in range(n - 1):
        t = time_span[i]
        x1 = sol[i, 0]

        # Update temperature, stiffness, and damping
        T_t[i] = temperature(t, T0, DeltaT, Omega_T)
        k_t[i] = stiffness_from_T(T_t[i], k0, T0, alpha_T)
        c_t[i] = damping_from_x1(mu, x1)

        # RK4 step
        s = sol[i].copy()
        k1 = vdp_dynamics(s, t, mu, m, k0, T0, DeltaT, Omega_T, alpha_T)
        k2 = vdp_dynamics(s + 0.5*dt*k1, t+0.5*dt, mu, m, k0, T0, DeltaT, Omega_T, alpha_T)
        k3 = vdp_dynamics(s + 0.5*dt*k2, t+0.5*dt, mu, m, k0, T0, DeltaT, Omega_T, alpha_T)
        k4 = vdp_dynamics(s + dt*k3, t+dt, mu, m, k0, T0, DeltaT, Omega_T, alpha_T)
        sol[i+1] = s + (dt/6.0)*(k1+2*k2+2*k3+k4)

    # Last time step
    T_t[-1] = temperature(time_span[-1], T0, DeltaT, Omega_T)
    k_t[-1] = stiffness_from_T(T_t[-1], k0, T0, alpha_T)
    c_t[-1] = damping_from_x1(mu, sol[-1, 0])

    return time_span, sol, T_t, k_t, c_t

# =========================
# Plotting functions (following your style)
# =========================
# Displacement vs Time
# ===== Set global font to Times New Roman =====
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'  # Math formulas still use LaTeX CM font

def plot_displacement(time_span, solution, save_path=None):
    """Plot displacement x1 vs time"""
    x1 = solution[:, 0]
    plt.figure(figsize=(8, 6))
    plt.plot(time_span, x1, 'b-', linewidth=1.5)
    plt.xlabel('Time (s)', fontsize=20)
    plt.ylabel('Displacement', fontsize=20)
    plt.xticks(fontsize=18); plt.yticks(fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        file_name = os.path.join(save_path, "displacement_time_plot.pdf")
        plt.savefig(file_name)
    plt.show()

# Velocity vs Time
def plot_velocity(time_span, solution, save_path=None):
    """Plot velocity x2 vs time"""
    x2 = solution[:, 1]
    plt.figure(figsize=(8, 6))
    plt.plot(time_span, x2, 'b-', linewidth=1.5)
    plt.xlabel('Time (s)', fontsize=20)
    plt.ylabel('Velocity', fontsize=20)
    plt.xticks(fontsize=18); plt.yticks(fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        file_name = os.path.join(save_path, "velocity_time_plot.pdf")
        plt.savefig(file_name)
    plt.show()

# Phase portrait (x1-x2)
def plot_phase_portrait(solution, save_path=None):
    """Plot phase portrait (x1 vs x2)"""
    x1 = solution[:, 0]; x2 = solution[:, 1]
    plt.figure(figsize=(8, 6))
    plt.plot(x1, x2, 'b-', linewidth=1.5)
    plt.xlabel('Displacement', fontsize=20)
    plt.ylabel('Velocity', fontsize=20)
    plt.xticks(fontsize=18); plt.yticks(fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        file_name = os.path.join(save_path, "phase_portrait_plot.pdf")
        plt.savefig(file_name)
    plt.show()

# Temperature vs Time
def plot_temperature(time_span, T_t, save_path=None):
    """Plot temperature T(t) vs time"""
    plt.figure(figsize=(8, 6))
    plt.plot(time_span, T_t, 'b-', linewidth=1.5)
    plt.xlabel('Time (s)', fontsize=20)
    plt.ylabel('Temperature', fontsize=20)
    plt.xticks(fontsize=18); plt.yticks(fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        file_name = os.path.join(save_path, "temperature_time_plot.pdf")
        plt.savefig(file_name)
    plt.show()

# Stiffness vs Time
def plot_stiffness(time_span, k_t, save_path=None):
    """Plot stiffness k(t) vs time"""
    plt.figure(figsize=(8, 6))
    plt.plot(time_span, k_t, 'b-', linewidth=1.5)
    plt.xlabel('Time (s)', fontsize=20)
    plt.ylabel('Stiffness', fontsize=20)
    plt.xticks(fontsize=18); plt.yticks(fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        file_name = os.path.join(save_path, "stiffness_time_plot.pdf")
        plt.savefig(file_name)
    plt.show()

# Damping vs Time
def plot_damping(time_span, c_t, save_path=None):
    """Plot damping c(t) vs time"""
    plt.figure(figsize=(8, 6))
    plt.plot(time_span, c_t, 'b-', linewidth=1.5)
    plt.xlabel('Time (s)', fontsize=20)
    plt.ylabel('Damping', fontsize=20)
    plt.xticks(fontsize=18); plt.yticks(fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        file_name = os.path.join(save_path, "damping_time_plot.pdf")
        plt.savefig(file_name)
        print(f"Damping plot saved to: {file_name}")
    plt.show()

# =========================
# Example usage

def save_state(solution, save_path):
    """
    Save state vector [x1, x2] to a specified directory as a NumPy .npy file.

    Parameters
    ----------
    solution : numpy.ndarray
        State vector array with shape (n_steps, 2) containing [x1, x2].
    save_path : str
        Directory to save the file.
    """
    # Ensure directory exists
    os.makedirs(save_path, exist_ok=True)

    # File path
    file_path = os.path.join(save_path, "state_data.npy")

    # Save state
    np.save(file_path, solution)

    print(f"State vector saved to: {file_path}")


t, sol, T_t, k_t, c_t = simulate_vdp(
    mu=1.2, k0=1.0, m=1.0,
    T0=0.0, DeltaT=0.5, Omega_T=0.1, alpha_T=0.3,
    dt=0.005, t_end=200.0, x10=1.0, x20=0.0
)

# Call plotting functions and save as PDF
plot_displacement(t, sol, save_path=None)        # Displacement vs Time
plot_velocity(t, sol, save_path=None)            # Velocity vs Time
plot_phase_portrait(sol, save_path=None)         # Phase portrait
plot_temperature(t, T_t, save_path=None)         # Temperature vs Time
plot_stiffness(t, k_t, save_path=None)           # Stiffness vs Time
plot_damping(t, c_t, save_path=None)             # Damping vs Time



save_dir = os.path.dirname(os.path.abspath(__file__))

# Save state vector
save_state(sol, save_dir)















