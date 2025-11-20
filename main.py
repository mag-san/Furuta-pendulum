import logging
import math
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from euler_backward import euler_backward
from euler_forward import euler_forward
from runge_kutta_4 import runge_kutta_4
from system_ode import furuta_pendulum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_timestep_comparison(h, params, x0, t_span):
    """Run all solvers for a given timestep and return results."""
    t_eval = np.arange(t_span[0], t_span[1], h)

    logger.info(f"\n{'='*60}")
    logger.info(f"Running comparison for timestep h = {h}")
    logger.info(f"Time span: {t_span}, Steps: {len(t_eval)}")
    logger.info(f"{'='*60}")

    # Solve using RK45
    start_time = time.perf_counter()
    sol = solve_ivp(
        fun=lambda t, x: furuta_pendulum(t, x, params),
        t_span=t_span,
        y0=x0,
        t_eval=t_eval,
        method="RK45",
    )
    rk45_time = time.perf_counter() - start_time
    logger.info(f"RK45 (scipy) completed in {rk45_time:.6f} seconds")

    # Solve using Euler forward
    start_time = time.perf_counter()
    t_euler_fwd, y_euler_fwd = euler_forward(
        f=lambda t, x: furuta_pendulum(t, x, params), t_span=t_span, y0=x0, h=h
    )
    euler_fwd_time = time.perf_counter() - start_time
    logger.info(f"Euler Forward completed in {euler_fwd_time:.6f} seconds")

    # Solve using Euler backward
    start_time = time.perf_counter()
    t_euler_bwd, y_euler_bwd = euler_backward(
        f=lambda t, x: furuta_pendulum(t, x, params), t_span=t_span, y0=x0, h=h
    )
    euler_bwd_time = time.perf_counter() - start_time
    logger.info(f"Euler Backward completed in {euler_bwd_time:.6f} seconds")

    # Solve using RK4
    start_time = time.perf_counter()
    t_rk4, y_rk4 = runge_kutta_4(
        f=lambda t, x: furuta_pendulum(t, x, params), t_span=t_span, y0=x0, h=h
    )
    rk4_time = time.perf_counter() - start_time
    logger.info(f"RK4 completed in {rk4_time:.6f} seconds")

    # Summary
    logger.info("-" * 60)
    logger.info("Timing Summary:")
    logger.info(f"  RK45 (scipy):    {rk45_time:.6f} seconds")
    logger.info(f"  Euler Forward:   {euler_fwd_time:.6f} seconds")
    logger.info(f"  Euler Backward:  {euler_bwd_time:.6f} seconds")
    logger.info(f"  RK4:             {rk4_time:.6f} seconds")

    return {
        'sol': sol,
        't_euler_fwd': t_euler_fwd,
        'y_euler_fwd': y_euler_fwd,
        't_euler_bwd': t_euler_bwd,
        'y_euler_bwd': y_euler_bwd,
        't_rk4': t_rk4,
        'y_rk4': y_rk4,
        'times': {
            'rk45': rk45_time,
            'euler_fwd': euler_fwd_time,
            'euler_bwd': euler_bwd_time,
            'rk4': rk4_time
        }
    }


def plot_comparison(results, h, save_dir="plots"):
    """Create and save comparison plots for a given timestep."""
    import os
    os.makedirs(save_dir, exist_ok=True)

    sol = results['sol']
    t_rk4 = results['t_rk4']
    y_rk4 = results['y_rk4']
    t_euler_fwd = results['t_euler_fwd']
    y_euler_fwd = results['y_euler_fwd']
    t_euler_bwd = results['t_euler_bwd']
    y_euler_bwd = results['y_euler_bwd']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Furuta Pendulum Solver Comparison (h = {h})', fontsize=16, fontweight='bold')

    axes[0, 0].plot(sol.t, sol.y[0], label="RK45", linewidth=2, alpha=0.6)
    axes[0, 0].plot(t_rk4, y_rk4[:, 0], label="RK4", linewidth=2)
    axes[0, 0].plot(
        t_euler_fwd, y_euler_fwd[:, 0], "--", alpha=0.7, label="Euler Forward"
    )
    axes[0, 0].plot(
        t_euler_bwd, y_euler_bwd[:, 0], ":", alpha=0.7, label="Euler Backward"
    )
    axes[0, 0].set_xlabel("Time [s]")
    axes[0, 0].set_ylabel("θ [rad]")
    axes[0, 0].set_title("Arm angle")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(sol.t, sol.y[1], label="RK45", linewidth=2, alpha=0.6)
    axes[0, 1].plot(t_rk4, y_rk4[:, 1], label="RK4", linewidth=2)
    axes[0, 1].plot(
        t_euler_fwd, y_euler_fwd[:, 1], "--", alpha=0.7, label="Euler Forward"
    )
    axes[0, 1].plot(
        t_euler_bwd, y_euler_bwd[:, 1], ":", alpha=0.7, label="Euler Backward"
    )
    axes[0, 1].set_xlabel("Time [s]")
    axes[0, 1].set_ylabel("θ̇ [rad/s]")
    axes[0, 1].set_title("Arm angular velocity")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(sol.t, sol.y[2], label="RK45", linewidth=2, alpha=0.6)
    axes[1, 0].plot(t_rk4, y_rk4[:, 2], label="RK4", linewidth=2)
    axes[1, 0].plot(
        t_euler_fwd, y_euler_fwd[:, 2], "--", alpha=0.7, label="Euler Forward"
    )
    axes[1, 0].plot(
        t_euler_bwd, y_euler_bwd[:, 2], ":", alpha=0.7, label="Euler Backward"
    )
    axes[1, 0].set_xlabel("Time [s]")
    axes[1, 0].set_ylabel("α [rad]")
    axes[1, 0].set_title("Pendulum angle")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(sol.t, sol.y[3], label="RK45", linewidth=2, alpha=0.6)
    axes[1, 1].plot(t_rk4, y_rk4[:, 3], label="RK4", linewidth=2)
    axes[1, 1].plot(
        t_euler_fwd, y_euler_fwd[:, 3], "--", alpha=0.7, label="Euler Forward"
    )
    axes[1, 1].plot(
        t_euler_bwd, y_euler_bwd[:, 3], ":", alpha=0.7, label="Euler Backward"
    )
    axes[1, 1].set_xlabel("Time [s]")
    axes[1, 1].set_ylabel("α̇ [rad/s]")
    axes[1, 1].set_title("Pendulum angular velocity")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the figure
    h_str = f"{h:.4f}".replace(".", "_")
    filename = f"comparison_h_{h_str}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    logger.info(f"Plot saved to: {filepath}")

    plt.close(fig)


def main():
    params = {
        "m_alpha": 0.1,  # pendulum mass [kg]
        "m_theta": 0.2,  # arm mass [kg]
        "L": 0.3,  # pendulum length [m]
        "r": 0.15,  # arm length [m]
        "g": 9.81,  # gravity [m/s²]
        "tau": 0.0,  # input torque [N·m]
    }
    # initial values (theta, dottheta, alpha, dotalpha)
    x0 = [0, 0, math.pi / 2, 0]
    t_span = (0, 10)

    # Define timesteps to compare
    timesteps = [0.1, 0.01, 0.0001]

    logger.info("\n" + "="*60)
    logger.info("FURUTA PENDULUM - TIMESTEP COMPARISON STUDY")
    logger.info("="*60)
    logger.info(f"Parameters: {params}")
    logger.info(f"Initial conditions: θ={x0[0]}, θ̇={x0[1]}, α={x0[2]:.4f}, α̇={x0[3]}")
    logger.info(f"Time span: {t_span}")
    logger.info(f"Timesteps to compare: {timesteps}")
    logger.info("="*60)

    # Run comparisons for each timestep
    all_results = {}
    for h in timesteps:
        results = run_timestep_comparison(h, params, x0, t_span)
        all_results[h] = results
        plot_comparison(results, h)

    # Final summary
    logger.info("\n" + "="*60)
    logger.info("FINAL SUMMARY - ALL TIMESTEPS")
    logger.info("="*60)
    for h in timesteps:
        logger.info(f"\nTimestep h = {h}:")
        times = all_results[h]['times']
        logger.info(f"  RK45 (scipy):    {times['rk45']:.6f} seconds")
        logger.info(f"  Euler Forward:   {times['euler_fwd']:.6f} seconds")
        logger.info(f"  Euler Backward:  {times['euler_bwd']:.6f} seconds")
        logger.info(f"  RK4:             {times['rk4']:.6f} seconds")
    logger.info("="*60)
    logger.info("\nAll plots have been saved to the 'plots' directory")
    logger.info("="*60)


if __name__ == "__main__":
    main()
