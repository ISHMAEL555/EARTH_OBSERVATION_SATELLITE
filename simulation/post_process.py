"""
simulation/post_process.py

Post-processing and visualization utilities for ADCS simulation results.
"""

import numpy as np
import matplotlib.pyplot as plt
from config import SAVE_PLOTS, SHOW_PLOTS
import os


def save_plot(fig, filename: str):
    """Save plot to plots/ directory"""
    os.makedirs("plots", exist_ok=True)
    path = f"plots/{filename}"
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Plot saved: {path}")


def plot_attitude_error(time, pointing_error_deg, title="Attitude Pointing Error"):
    """Plot 3σ pointing error"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time, pointing_error_deg, 'b-', linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Pointing Error (deg) - 3σ')
    ax.set_title(title)
    ax.grid(True)
    
    # Requirement lines
    ax.axhline(0.1, color='g', linestyle='--', label='Nominal Requirement (0.1°)')
    ax.axhline(0.2, color='orange', linestyle='--', label='Post-Slew Requirement (0.2°)')
    ax.axhline(0.5, color='r', linestyle='--', label='Dumping Requirement (0.5°)')
    
    ax.legend()
    if SAVE_PLOTS:
        save_plot(fig, f"{title.lower().replace(' ', '_')}.png")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def plot_momentum(time, h_total_history, title="Total Stored Momentum"):
    """Plot momentum vector and magnitude"""
    h_hist = np.array(h_total_history)
    h_mag = np.linalg.norm(h_hist, axis=1)
    
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    
    # Components
    axs[0].plot(time, h_hist[:,0], label='h_x')
    axs[0].plot(time, h_hist[:,1], label='h_y')
    axs[0].plot(time, h_hist[:,2], label='h_z')
    axs[0].set_ylabel('Momentum (Nms)')
    axs[0].grid(True)
    axs[0].legend()
    
    # Magnitude
    axs[1].plot(time, h_mag, 'r-', linewidth=2)
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('|h| (Nms)')
    axs[1].grid(True)
    axs[1].set_title(title)
    
    if SAVE_PLOTS:
        save_plot(fig, f"{title.lower().replace(' ', '_')}.png")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def plot_rates(time, omega_history):
    """Plot body rates"""
    omega_hist = np.array(omega_history)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time, np.rad2deg(omega_hist[:,0]), label='ω_x')
    ax.plot(time, np.rad2deg(omega_hist[:,1]), label='ω_y')
    ax.plot(time, np.rad2deg(omega_hist[:,2]), label='ω_z')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Body Rate (deg/s)')
    ax.set_title('Body Angular Rates')
    ax.grid(True)
    ax.legend()
    
    if SAVE_PLOTS:
        save_plot(fig, "body_rates.png")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def print_performance_metrics(results):
    """Print key performance metrics"""
    pointing_err = np.array(results['pointing_error'])
    h_final = np.array(results['h_total'][-1])
    h_max = np.max(np.linalg.norm(np.array(results['h_total']), axis=1))
    
    print("\n" + "="*60)
    print("PERFORMANCE METRICS")
    print("="*60)
    print(f"Max 3σ Pointing Error : {pointing_err.max():.4f}°")
    print(f"Final 3σ Pointing Error: {pointing_err[-1]:.4f}°")
    print(f"Peak Stored Momentum  : {h_max:.3f} Nms")
    print(f"Final Momentum Magnitude: {np.linalg.norm(h_final):.3f} Nms")
    print("="*60)


def generate_all_plots(results, scenario_name: str):
    """Generate all standard plots for a scenario"""
    print(f"\nGenerating plots for {scenario_name}...")
    
    plot_attitude_error(results['time'], results['pointing_error'], 
                       f"Pointing Error - {scenario_name}")
    
    plot_momentum(results['time'], results['h_total'], 
                 f"Momentum History - {scenario_name}")
    
    plot_rates(results['time'], results['omega'])
    
    print_performance_metrics(results)