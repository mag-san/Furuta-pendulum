import csv
import logging
import math
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from euler_backward import euler_backward
from euler_forward import euler_forward
from runge_kutta_4 import runge_kutta_4
from system_ode import furuta_pendulum

# Configure logging
os.makedirs("logs", exist_ok=True)
log_filename = f"logs/solver_timing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler(log_filename),  # File output
    ],
)
logger = logging.getLogger(__name__)
logger.info(f"Logging to file: {log_filename}")


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
        "sol": sol,
        "t_euler_fwd": t_euler_fwd,
        "y_euler_fwd": y_euler_fwd,
        "t_euler_bwd": t_euler_bwd,
        "y_euler_bwd": y_euler_bwd,
        "t_rk4": t_rk4,
        "y_rk4": y_rk4,
        "times": {
            "rk45": rk45_time,
            "euler_fwd": euler_fwd_time,
            "euler_bwd": euler_bwd_time,
            "rk4": rk4_time,
        },
    }


def plot_comparison(results, h, save_dir="plots", exclude_euler_fwd=False):
    """Create and save comparison plots for a given timestep."""
    import os

    os.makedirs(save_dir, exist_ok=True)

    sol = results["sol"]
    t_rk4 = results["t_rk4"]
    y_rk4 = results["y_rk4"]
    t_euler_fwd = results["t_euler_fwd"]
    y_euler_fwd = results["y_euler_fwd"]
    t_euler_bwd = results["t_euler_bwd"]
    y_euler_bwd = results["y_euler_bwd"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    suffix = " (without Euler Forward)" if exclude_euler_fwd else ""
    fig.suptitle(
        f"Furuta Pendulum Solver Comparison (h = {h}){suffix}",
        fontsize=16,
        fontweight="bold",
    )

    axes[0, 0].plot(sol.t, sol.y[0], label="RK45", linewidth=2, alpha=0.6)
    axes[0, 0].plot(t_rk4, y_rk4[:, 0], label="RK4", linewidth=2)
    if not exclude_euler_fwd:
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
    if not exclude_euler_fwd:
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
    if not exclude_euler_fwd:
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
    if not exclude_euler_fwd:
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
    suffix_filename = "_no_euler_fwd" if exclude_euler_fwd else ""
    filename = f"comparison_h_{h_str}{suffix_filename}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    logger.info(f"Plot saved to: {filepath}")

    plt.close(fig)


def save_timing_results(all_results, params, x0, t_span, save_dir="logs"):
    """Save timing results to CSV file for analysis."""
    os.makedirs(save_dir, exist_ok=True)

    csv_filename = (
        f"{save_dir}/solver_timing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )

    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Write header with metadata
        writer.writerow(["# Furuta Pendulum Solver Timing Results"])
        writer.writerow(["# Generated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
        writer.writerow(["# Time span:", f"{t_span[0]} to {t_span[1]} seconds"])
        writer.writerow(
            [
                "# Initial conditions:",
                f"θ={x0[0]}, θ̇={x0[1]}, α={x0[2]:.4f}, α̇={x0[3]}",
            ]
        )
        writer.writerow(["# Parameters:", str(params)])
        writer.writerow([])

        # Write column headers
        writer.writerow(
            [
                "timestep_h",
                "num_steps",
                "rk45_time_sec",
                "euler_fwd_time_sec",
                "euler_bwd_time_sec",
                "rk4_time_sec",
            ]
        )

        # Write data for each timestep
        for h, results in sorted(all_results.items()):
            num_steps = len(results["sol"].t)
            times = results["times"]
            writer.writerow(
                [
                    h,
                    num_steps,
                    f"{times['rk45']:.6f}",
                    f"{times['euler_fwd']:.6f}",
                    f"{times['euler_bwd']:.6f}",
                    f"{times['rk4']:.6f}",
                ]
            )

    logger.info(f"Timing results saved to: {csv_filename}")
    return csv_filename


def main():
    params = {
        "m_alpha": 0.1,  # pendulum mass [kg]
        "m_theta": 0.2,  # arm mass [kg]
        "L": 0.095,  # pendulum length [m]
        "r": 0.095,  # arm length [m]
        "g": 9.81,  # gravity [m/s²]
        "tau": 0.0,  # input torque [N·m]
    }
    # initial values (theta, dottheta, alpha, dotalpha)
    x0 = [0, 0, math.pi / 2, 0]
    t_span = (0, 10)

    # Define timesteps to compare
    timesteps = [0.1, 0.01, 0.0001, 0.00001]

    logger.info("\n" + "=" * 60)
    logger.info("FURUTA PENDULUM - TIMESTEP COMPARISON STUDY")
    logger.info("=" * 60)
    logger.info(f"Parameters: {params}")
    logger.info(f"Initial conditions: θ={x0[0]}, θ̇={x0[1]}, α={x0[2]:.4f}, α̇={x0[3]}")
    logger.info(f"Time span: {t_span}")
    logger.info(f"Timesteps to compare: {timesteps}")
    logger.info("=" * 60)

    # Run comparisons for each timestep
    all_results = {}
    for h in timesteps:
        results = run_timestep_comparison(h, params, x0, t_span)
        all_results[h] = results
        plot_comparison(results, h)

        # For h=0.1 and h=0.01, also create a version without Euler Forward
        if h == 0.1 or h == 0.01:
            plot_comparison(results, h, exclude_euler_fwd=True)

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("FINAL SUMMARY - ALL TIMESTEPS")
    logger.info("=" * 60)
    for h in timesteps:
        logger.info(f"\nTimestep h = {h}:")
        times = all_results[h]["times"]
        logger.info(f"  RK45 (scipy):    {times['rk45']:.6f} seconds")
        logger.info(f"  Euler Forward:   {times['euler_fwd']:.6f} seconds")
        logger.info(f"  Euler Backward:  {times['euler_bwd']:.6f} seconds")
        logger.info(f"  RK4:             {times['rk4']:.6f} seconds")
    logger.info("=" * 60)

    # Save timing results to CSV
    csv_file = save_timing_results(all_results, params, x0, t_span)

    logger.info("\nAll plots have been saved to the 'plots' directory")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
