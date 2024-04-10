import numpy as np
import torch
from stochastic.processes.continuous import FractionalBrownianMotion

class Simulation():
    """
    Generate survival data with different settings
    - The longitudinal features (time series) are sample path  of a fractional
    Brownian motion.
    - The time-of-event is defined as the time when trajectory
    (which is the output of specific differential equations) cross the
    threshold during the observation period.
    """
    def __init__(self):
        pass

    def get_path(self, n_samples, n_sampling_times, end_time, dim=3, hurst=0.7, intercept=True):
        """
        Generate the longitudinal features as the sample path  of a fractional
        Brownian motion.

        Parameters
        ----------

        n_samples : `int`
            The number of individual

        n_sampling_times : `int`
            The number of measurement time points between 0 to end_time

        end_time : `float`
            The end time of observation period.

        dim : `int`
            The dimension of sample path (longitudinal features)
            (include the time channel)

        hurst : `float`
            Hurst parameter

        intercept : `bool`, default = True
            Whether to add random intercept to sample paths

        Returns
        -------
        paths : : `Tensor`, shape=(n_samples, n_sampling_times, dim)
            The simulated longitudinal features

        sampling_times : `np.ndarray`, shape=(n_sampling_times, )
            The measurement time points
        """
        step = end_time / n_sampling_times
        sampling_times = torch.arange(0, end=end_time, step=step)
        n_sampling_times = sampling_times.shape[0]
        paths = torch.empty(n_samples, n_sampling_times, dim)
        paths[:, :, 0] = sampling_times
        fbm = FractionalBrownianMotion(t=end_time, hurst=hurst, rng=np.random.default_rng(0))
        for j in torch.arange(n_samples):
            for d in torch.arange(1, dim):
                paths[j, :, d] = torch.tensor(fbm.sample(n_sampling_times - 1))

        if intercept:
            torch.manual_seed(0)
            intercept = torch.randn_like(paths[:, 0, 1:])
            intercepts = torch.repeat_interleave(intercept, repeats=n_sampling_times, axis=0)
            paths[:, :, 1:] += intercepts.reshape(n_samples, -1, dim - 1)

        sampling_times = np.array(sampling_times)

        return paths, sampling_times

    def get_trajectory(self, path):
        pass

    def get_survival_label(self, paths, end_time, threshold=2.5):
        """
        Generates survival times based on paths. The individual experiences an
        event if its trajectory crosses the threshold value. We censor
        individuals whose trajectory does not cross the threshold during the
        observation period. This means that individuals are never censored
        during the observation period, but only at the end.

        Parameters
        ----------
        paths : : `Tensor`, shape=(n_samples, n_sampling_times, dim)
            The simulated longitudinal features

        end_time : `float`
            The end time of observation period.

        threshold : `float`, default = 2.5
            The threshold.

        Returns
        -------

        surv_time : `np.ndarray`, shape=(n_samples, )
            Survival times, equal to end_time if the individual is censored

        surv_ind : `np.ndarray`, shape=(n_samples, )
            Boolean indicator, equal to True if the individual is uncensored.
        """

        trajectories = self.get_trajectory(paths)
        max_paths = trajectories - threshold
        idxs = np.argmax(max_paths > 0, axis=1)
        sampling_times_ext = np.array(paths[:, :, 0])
        surv_times = np.take(sampling_times_ext, idxs)
        surv_times[surv_times == 0] = end_time
        surv_inds = (idxs != 0)

        return np.array(surv_times), np.array(surv_inds)

    def generate_simulated_dataset(self, n_samples, n_sampling_times, end_time, dim=3,
                                   hurst=0.7, intercept=True, threshold=5):
        """
        Wrapper function to generate the survival data

        Parameters
        ----------

        n_samples : `int`
            The number of individual

        n_sampling_times : `int`
            The number of measurement time points between 0 to end_time

        end_time : `float`
            The end time of observation period.

        dim : `int`
            The dimension of sample path (longitudinal features)
            (include the time channel)

        hurst : `float`
            Hurst parameter

        intercept : `bool`, default = True
            Whether to add random intercept to sample paths

        threshold : `float`, default = 2.5
            The threshold.

        Returns
        -------
        paths : : `Tensor`, shape=(n_samples, n_sampling_times, dim)
            The simulated longitudinal features

        sampling_times : `np.ndarray`, shape=(n_sampling_times, )
            The measurement time points

        surv_time : `np.ndarray`, shape=(n_samples, )
            Survival times, equal to end_time if the individual is censored

        surv_ind : `np.ndarray`, shape=(n_samples, )
            Boolean indicator, equal to True if the individual is uncensored.
        """
        paths, sampling_times = self.get_path(n_samples, n_sampling_times,
                                              end_time, dim, hurst, intercept)
        surv_times, surv_inds = self.get_survival_label(paths, end_time, threshold)

        return paths, sampling_times, surv_times, surv_inds

class Ornstein_Uhlenbeck(Simulation):
    """
    Simulate the trajectory by Ornstein Uhlenbeck Stochastic Differential Equation
    """

    def get_trajectory(self, path: torch.Tensor) -> torch.Tensor:

        theta = 0.1
        mu = 0.1
        sigma = 1.
        n_samples, n_sampling_times, _ = path.shape
        trajectories = np.ones((n_samples, n_sampling_times))
        time_step = path[0, 1, 0] - path[0, 0, 0]
        seed = 0
        for i in np.arange(n_samples):
            for t in np.arange(1, n_sampling_times):
                random_increment = path[i, t, :] - path[i, t - 1, :]
                prev_traj = trajectories[i, t - 1]
                rnd = np.random.default_rng(seed).normal(0, 1)
                trajectories[i, t] = prev_traj + random_increment.sum() \
                                     - theta * (prev_traj - mu) * time_step \
                                     + np.sqrt(time_step) * sigma * rnd
                seed += 1

        return trajectories

class TumorGrowth(Simulation):
    """
    Simulate the trajectory by Differential Equation (Simeoni et al, 2004)
    representing tumor growth dynamics.
    """
    def __init__(self, lambda_0: float = 0.9, lambda_1: float = 0.7,
                 k_1: float = 10., k_2: float = 0.15, psi: int = 20):
        self.lambda_0 = lambda_0
        self.lambda_1 = lambda_1
        self.k_1 = k_1
        self.k_2 = k_2
        self.psi = psi

    def tumor_growth(self, u: torch.Tensor, y: float, x: float) -> np.ndarray:
        assert u.ndim == 1, "u must be a one dimensional array of length 4"
        assert u.shape[0] == 4, "u must be a one dimensional array of length 4"

        assert isinstance(x, float), "x must be a float"
        assert isinstance(y, float), "y must be a float"

        tmp = (1 + (self.lambda_0 / self.lambda_1 * y) ** self.psi)
        du_1 = (self.lambda_0 * u[0] * tmp) ** (-1 / self.psi) - self.k_2 * x * u[0]
        du_2 = self.k_2 * x * u[0] - self.k_1 * u[1]
        du_3 = self.k_1 * (u[1] - u[2])
        du_4 = self.k_1 * (u[2] - u[3])

        return np.array([du_1, du_2, du_3, du_4])

    def get_trajectory(self, path: torch.Tensor) -> torch.Tensor:

        n_samples, n_sampling_times, _ = path.shape
        trajectories = np.ones((n_samples, n_sampling_times))
        time_step = float(path[0, 1, 0] - path[0, 0, 0])
        drugs = path[:,:,1]

        for i in np.arange(n_samples):
            # tumor growth
            y_prev = trajectories[i, 0]
            u_init = np.array([0.8, 0, 0, 0])
            u_prev = u_init
            drug = drugs[i]
            for j in np.arange(1, n_sampling_times):
                x = drug[j]
                u = u_prev + self.tumor_growth(u_prev, y_prev, float(x)) * time_step
                trajectories[i, j] = np.sum(u)
                u_prev = u
                y_prev = trajectories[i, j]

        return trajectories