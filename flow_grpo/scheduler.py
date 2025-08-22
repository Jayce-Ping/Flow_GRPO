from typing import Optional
import torch
from diffusers import FlowMatchEulerDiscreteScheduler

class FlowMatchSlidingWindowScheduler(FlowMatchEulerDiscreteScheduler):
    """
        A scheduler with noise level provided only within the given window.
        The window is set by `window_size` and `left_boundary`.
        For example, given `window_size=2` and `left_boundary=3`,
        the noise window is [3, 4], and `right_boundary=5` is not included in the window.
    """
    def __init__(
        self,
        noise_level : float = 0.9,
        window_size: int = 1000,
        left_boundary : int = 0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._window_size = min(window_size, self.config.num_train_timesteps)
        self.noise_level = noise_level
        self._left_boundary = left_boundary

        assert self.noise_level >= 0 and self.noise_level <= 1, "Noise level must be between 0 and 1."
        assert self._window_size > 0, "Window size must be greater than 0."
        assert self._left_boundary >= 0, "Left boundary must be non-negative."

        self.cur_timestep = self._left_boundary
        self.cur_iter_in_group = 0

    @property
    def window_size(self):
        if self._window_size > len(self.timesteps) - self._left_boundary:
            self._window_size = len(self.timesteps) - self._left_boundary

        return self._window_size

    @property
    def left_boundary(self):
        if self._left_boundary > len(self.timesteps):
            # Reset left boundary to zero
            print("Left boundary exceeds the number of timesteps. Resetting to zero.")
            self._left_boundary = 0

        return self._left_boundary

    @property
    def right_boundary(self):
        return self.left_boundary + self.window_size
    
    def get_window_timesteps(self) -> torch.Tensor:
        return self.timesteps[self.left_boundary:self.right_boundary]

    def get_window_sigmas(self) -> torch.Tensor:
        return self.sigmas[self.left_boundary:self.right_boundary]

    def get_noise_levels(self) -> torch.Tensor:
        """ Returns noise levels on all timesteps, where noise level is non-zero only within the current window. """
        window_indices = [self.index_for_timestep(t) for t in self.get_window_timesteps()]
        noise_levels = torch.zeros_like(self.timesteps, dtype=torch.float32)
        noise_levels[window_indices] = self.noise_level
        return noise_levels
    
    def get_noise_level_for_timestep(self, time_step) -> float:
        """
            Return the noise level for a specific timestep.
        """
        time_step_index = self.index_for_timestep(time_step)
        if self.left_boundary <= time_step_index < self.right_boundary:
            return self.noise_level

        return 0.0