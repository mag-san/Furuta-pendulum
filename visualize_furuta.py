import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp

from system_ode import furuta_pendulum

"""
this code was generated using claude code. It uses the outputs from solve_ivp to vizualize
the pendulumswings over time
"""


def compute_positions(theta, alpha, r, L):
    """
    Compute 3D positions of arm end and pendulum bob.

    Args:
        theta: Arm angle [rad]
        alpha: Pendulum angle from vertical [rad]
        r: Arm length [m]
        L: Pendulum length [m]

    Returns:
        arm_end: (x, y, z) position of arm end
        pendulum_bob: (x, y, z) position of pendulum bob
    """
    # Arm end position (rotating in horizontal plane)
    arm_x = r * np.cos(theta)
    arm_y = r * np.sin(theta)
    arm_z = 0.0
    arm_end = (arm_x, arm_y, arm_z)

    # Pendulum bob position (swinging from arm end)
    # The pendulum swings in a plane that rotates with the arm
    bob_x = arm_x + L * np.sin(alpha) * np.cos(theta)
    bob_y = arm_y + L * np.sin(alpha) * np.sin(theta)
    bob_z = -L * np.cos(alpha)
    pendulum_bob = (bob_x, bob_y, bob_z)

    return arm_end, pendulum_bob


class FurutaPendulumAnimation:
    """Animated visualization of the Furuta pendulum."""

    def __init__(self, sol, params, trail_length=50):
        """
        Initialize the animation.

        Args:
            sol: Solution object from solve_ivp
            params: Dictionary of pendulum parameters
            trail_length: Number of previous positions to show as trail
        """
        self.sol = sol
        self.params = params
        self.r = params["r"]
        self.L = params["L"]
        self.trail_length = trail_length

        # Extract solution data
        self.t = sol.t
        self.theta = sol.y[0]  # Arm angle
        self.alpha = sol.y[2]  # Pendulum angle

        # Setup figure and 3D axis
        self.fig = plt.figure(figsize=(14, 6))
        self.ax = self.fig.add_subplot(121, projection="3d")

        # Setup plot limits
        max_reach = self.r + self.L
        self.ax.set_xlim([-max_reach, max_reach])
        self.ax.set_ylim([-max_reach, max_reach])
        self.ax.set_zlim([-self.L * 1.2, self.L * 0.2])

        self.ax.set_xlabel("X [m]")
        self.ax.set_ylabel("Y [m]")
        self.ax.set_zlabel("Z [m]")
        self.ax.set_title("Furuta Pendulum Animation")

        # Initialize plot elements
        (self.arm_line,) = self.ax.plot([], [], [], "b-", linewidth=3, label="Arm")
        (self.pendulum_line,) = self.ax.plot(
            [], [], [], "r-", linewidth=2, label="Pendulum"
        )
        (self.arm_end_point,) = self.ax.plot([], [], [], "bo", markersize=8)
        (self.bob_point,) = self.ax.plot([], [], [], "ro", markersize=10)
        (self.trail_line,) = self.ax.plot([], [], [], "r--", alpha=0.3, linewidth=1)

        # Add base pivot
        self.ax.plot([0], [0], [0], "ko", markersize=12)
        self.ax.legend()

        # Setup time series plots
        self.ax_theta = self.fig.add_subplot(322)
        self.ax_alpha = self.fig.add_subplot(324)
        self.ax_energy = self.fig.add_subplot(326)

        # Plot full time series as background
        self.ax_theta.plot(self.t, self.theta, "b-", alpha=0.3)
        self.ax_theta.set_ylabel("θ [rad]")
        self.ax_theta.set_title("Arm Angle")
        self.ax_theta.grid(True)

        self.ax_alpha.plot(self.t, self.alpha, "r-", alpha=0.3)
        self.ax_alpha.set_ylabel("α [rad]")
        self.ax_alpha.set_title("Pendulum Angle")
        self.ax_alpha.grid(True)

        # Current position markers on time series
        (self.theta_marker,) = self.ax_theta.plot([], [], "bo", markersize=8)
        (self.alpha_marker,) = self.ax_alpha.plot([], [], "ro", markersize=8)

        # Energy plot (kinetic + potential)
        self.energy = self._compute_energy()
        self.ax_energy.plot(self.t, self.energy, "g-", alpha=0.3)
        self.ax_energy.set_xlabel("Time [s]")
        self.ax_energy.set_ylabel("Energy [J]")
        self.ax_energy.set_title("Total Energy")
        self.ax_energy.grid(True)
        (self.energy_marker,) = self.ax_energy.plot([], [], "go", markersize=8)

        # Time text
        self.time_text = self.ax.text2D(0.05, 0.95, "", transform=self.ax.transAxes)

        # Trail storage
        self.trail_x = []
        self.trail_y = []
        self.trail_z = []

    def _compute_energy(self):
        """Compute total mechanical energy at each time step."""
        m_alpha = self.params["m_alpha"]
        m_theta = self.params["m_theta"]
        g = self.params["g"]

        theta_dot = self.sol.y[1]
        alpha_dot = self.sol.y[3]

        # Simplified energy calculation (approximate)
        # Kinetic energy: rotational + pendulum
        KE = 0.5 * m_theta * (self.r * theta_dot) ** 2
        KE += 0.5 * m_alpha * (self.r * theta_dot) ** 2
        KE += 0.5 * m_alpha * (self.L * alpha_dot) ** 2

        # Potential energy
        PE = -m_alpha * g * self.L * np.cos(self.alpha)

        return KE + PE

    def init(self):
        """Initialize animation."""
        self.arm_line.set_data([], [])
        self.arm_line.set_3d_properties([])
        self.pendulum_line.set_data([], [])
        self.pendulum_line.set_3d_properties([])
        self.arm_end_point.set_data([], [])
        self.arm_end_point.set_3d_properties([])
        self.bob_point.set_data([], [])
        self.bob_point.set_3d_properties([])
        self.trail_line.set_data([], [])
        self.trail_line.set_3d_properties([])
        self.theta_marker.set_data([], [])
        self.alpha_marker.set_data([], [])
        self.energy_marker.set_data([], [])
        self.time_text.set_text("")

        return (
            self.arm_line,
            self.pendulum_line,
            self.arm_end_point,
            self.bob_point,
            self.trail_line,
            self.theta_marker,
            self.alpha_marker,
            self.energy_marker,
            self.time_text,
        )

    def update(self, frame):
        """Update animation frame."""
        # Get current state
        theta = self.theta[frame]
        alpha = self.alpha[frame]

        # Compute positions
        arm_end, pendulum_bob = compute_positions(theta, alpha, self.r, self.L)

        # Update arm (from origin to arm end)
        self.arm_line.set_data([0, arm_end[0]], [0, arm_end[1]])
        self.arm_line.set_3d_properties([0, arm_end[2]])

        # Update pendulum (from arm end to bob)
        self.pendulum_line.set_data(
            [arm_end[0], pendulum_bob[0]], [arm_end[1], pendulum_bob[1]]
        )
        self.pendulum_line.set_3d_properties([arm_end[2], pendulum_bob[2]])

        # Update points
        self.arm_end_point.set_data([arm_end[0]], [arm_end[1]])
        self.arm_end_point.set_3d_properties([arm_end[2]])

        self.bob_point.set_data([pendulum_bob[0]], [pendulum_bob[1]])
        self.bob_point.set_3d_properties([pendulum_bob[2]])

        # Update trail
        self.trail_x.append(pendulum_bob[0])
        self.trail_y.append(pendulum_bob[1])
        self.trail_z.append(pendulum_bob[2])

        if len(self.trail_x) > self.trail_length:
            self.trail_x.pop(0)
            self.trail_y.pop(0)
            self.trail_z.pop(0)

        self.trail_line.set_data(self.trail_x, self.trail_y)
        self.trail_line.set_3d_properties(self.trail_z)

        # Update time series markers
        self.theta_marker.set_data([self.t[frame]], [theta])
        self.alpha_marker.set_data([self.t[frame]], [alpha])
        self.energy_marker.set_data([self.t[frame]], [self.energy[frame]])

        # Update time text
        self.time_text.set_text(f"Time: {self.t[frame]:.2f} s")

        return (
            self.arm_line,
            self.pendulum_line,
            self.arm_end_point,
            self.bob_point,
            self.trail_line,
            self.theta_marker,
            self.alpha_marker,
            self.energy_marker,
            self.time_text,
        )

    def animate(self, interval=20, save_path=None):
        """
        Create and display/save the animation.

        Args:
            interval: Time between frames in milliseconds
            save_path: If provided, save animation to this path (e.g., 'animation.mp4')
        """
        anim = FuncAnimation(
            self.fig,
            self.update,
            init_func=self.init,
            frames=len(self.t),
            interval=interval,
            blit=True,
        )

        plt.tight_layout()

        if save_path:
            print(f"Saving animation to {save_path}...")
            anim.save(save_path, writer="pillow", fps=30)
            print("Animation saved!")
        else:
            plt.show()

        return anim


def main():
    """Run the Furuta pendulum visualization."""
    # Pendulum parameters
    params = {
        "m_alpha": 0.50,  # pendulum mass [kg]
        "m_theta": 0.1,  # arm mass [kg]
        "L": 0.095,  # pendulum length [m]
        "r": 0.095,  # arm length [m]
        "g": 9.81,  # gravity [m/s²]
        "tau": 0.0,  # input torque [N·m]
    }

    # Initial conditions: [theta, theta_dot, alpha, alpha_dot]
    # Starting with pendulum slightly off from inverted position
    x0 = [0.0, 0, 2, 0]

    # Time span
    t_span = (0, 10)
    t_eval = np.linspace(0, 10, 500)

    print("Solving ODE system...")
    sol = solve_ivp(
        fun=lambda t, x: furuta_pendulum(t, x, params),
        t_span=t_span,
        y0=x0,
        t_eval=t_eval,
        method="RK45",
    )

    print("Creating animation...")
    animator = FurutaPendulumAnimation(sol, params, trail_length=50)

    # To save animation, uncomment and provide path:
    # animator.animate(interval=20, save_path='furuta_pendulum.gif')

    # Display animation
    animator.animate(interval=20)


if __name__ == "__main__":
    main()
