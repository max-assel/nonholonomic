import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, sin
from matplotlib.animation import FuncAnimation

from common import plot_car, update_state, calc_lambda_dot, update_lambda, draw_initial


class KinematicsController(object):
    def __init__(self):

        # self.MAX_STEP = 100
        self.dt = 0.01 # time step
        self.tf = 8 # final time
        self.clambda = 1
        self.cp = 1.0
        self.cd = 2.0
        self.epsilon = 0.01
        self.lambda_init = 0.5

        assert(self.cp > 0), ("cp must be positive, but got {}".format(self.cp))
        assert(self.cd > 0), ("cd must be positive, but got {}".format(self.cd))
        assert(self.clambda > 0), ("clambda must be positive, but got {}".format(self.clambda))
        assert(self.epsilon > 0), ("epsilon must be positive, but got {}".format(self.epsilon))
        assert(self.lambda_init > 0), ("lambda_init must be positive, but got {}".format(self.lambda_init))
        assert(self.lambda_init > self.epsilon), ("lambda_init must be larger than epsilon, but got lambda_init = {} and epsilon = {}".format(self.lambda_init, self.epsilon))

        assert(self.tf > 0), ("tf must be positive, but got {}".format(self.tf))
        assert(self.dt > 0), ("dt must be positive, but got {}".format(self.dt))
        assert(self.dt < self.tf), ("dt must be smaller than tf, but got dt = {} and tf = {}".format(self.dt, self.tf))

        # self.lambda_ = 0.5

        # self.initialize_xi()
        # self.initialize_lambda()

        self.x0 = np.zeros(2)
        N = len(np.arange(0, self.tf, self.dt))
        self.xi_hist = np.empty((3, N))
        self.tau_hist = np.empty((2, N))
        self.t_hist = np.empty(N)


    def initialize_xi(self):
        xi = np.empty(3)
        xi[0] = 2             # x-position
        xi[1] = 3             # y-position
        xi[2] = np.pi * 7 / 4 # theta
        return xi
    
    def initialize_lambda(self):
        lambda_ = self.lambda_init
        return lambda_

    def dynamics(self, xi, u):
        f = np.empty(3)

        f[0] = cos(xi[2]) * u[0] # x-velocity
        f[1] = sin(xi[2]) * u[0] # y-velocity
        f[2] = u[1]                   # angular velocity  
        return f

    def feedback_control_law(self, xi, lambda_, x0):
        x = xi[0:2]
        theta = xi[2]
        rot_l_inv = np.empty((2, 2))
        rot_l_inv[0, 0] = cos(theta)
        rot_l_inv[0, 1] = sin(theta)
        rot_l_inv[1, 0] = -sin(theta) / lambda_
        rot_l_inv[1, 1] = cos(theta) / lambda_
        q = x + lambda_ * np.array([cos(theta), sin(theta)])
        lambda_dot = calc_lambda_dot(lambda_, self.clambda, self.epsilon)
        e1 = np.zeros(2)
        e1[0] = 1
        tau = -self.cp * rot_l_inv @ (q - x0) - lambda_dot * e1
        return tau

    def run(self):
        xi = self.initialize_xi()
        lambda_ = self.initialize_lambda()

        # save trajectory
        for (step, t) in enumerate(np.arange(0, self.tf, self.dt)):
            tau = self.feedback_control_law(xi, lambda_, self.x0)

            xi = update_state(lambda t, xi: self.dynamics(xi, tau), (t, t + self.dt), xi)

            lambda_ = update_lambda(lambda t, l_: calc_lambda_dot(l_, self.clambda, self.epsilon), (t, t + self.dt), lambda_)

            self.xi_hist[:, step] = xi
            self.tau_hist[:, step] = tau
            self.t_hist[step] = t

    def update(self, iter):
        # global lambda_, xi
        t = self.dt * iter
        tau = self.feedback_control_law(self.xi_vis, self.lambda_vis, self.x0)

        new_xi_vis = self.xi_vis
        new_xi_vis = update_state(lambda t, new_xi_vis: self.dynamics(new_xi_vis, tau), (t, t + self.dt), new_xi_vis)
        self.xi_vis = new_xi_vis

        new_lambda_vis = self.lambda_vis
        new_lambda_vis = update_lambda(lambda t, l_: calc_lambda_dot(l_, self.clambda, self.epsilon), (t, t + self.dt), new_lambda_vis)
        self.lambda_vis = new_lambda_vis

        (
            car_x,
            car_y,
            car_angle_x,
            car_angle_y,
            left_tire_x,
            left_tire_y,
            right_tire_x,
            right_tire_y,
        ) = plot_car(self.xi_vis[0:3])

        self.plots["car"].set_data(car_x, car_y)
        self.plots["car_angle"].set_data(car_angle_x, car_angle_y)
        self.plots["lefT_tire"].set_data(left_tire_x, left_tire_y)
        self.plots["right_tire"].set_data(right_tire_x, right_tire_y)
        self.plots["tau0"].set_offsets((t, tau[0]))
        self.plots["tau1"].set_offsets((t, tau[1]))


    def visualize(self):
        # reset for animation
        
        self.xi_vis = self.initialize_xi()
        self.lambda_vis = self.initialize_lambda()

        fig, axes = plt.subplots(1, 2)
        ax1, ax2 = axes

        ax1.set_xlim(0, 3.25)
        ax1.set_ylim(0, 3.25)
        ax1.set_aspect("equal")

        self.plots = draw_initial(axes)
        fanm = FuncAnimation(fig, self.update, interval=10, frames=500)

        ax1.plot(self.xi_hist[0, :], self.xi_hist[1, :], zorder=-1)
        ax2.plot(self.t_hist, self.tau_hist[0, :], self.t_hist, self.tau_hist[1, :], zorder=-1)
        ax2.legend(["v", "$\omega$"])  # noqa: W605
        # plt.show()
        fanm.save("movie.mp4", "ffmpeg")

print("Kinematics Controller")
controller = KinematicsController()

print("Running simulation")
controller.run()

print("Visualizing results")
controller.visualize()