import math
from typing import Optional, Union
import torch

from diffusers.utils.torch_utils import randn_tensor
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

    def get_window_timesteps(self, left_boundary : Optional[int] = None) -> torch.Tensor:
        """
            Returns timesteps within the current window.
            If `left_boundary` is provided, use it instead of the current left boundary.
        """
        if left_boundary is None:
            left_boundary = self.left_boundary

        return self.timesteps[left_boundary:self.right_boundary]

    def get_window_sigmas(self, left_boundary : Optional[int] = None) -> torch.Tensor:
        """
            Returns sigmas within the current window.
            If `left_boundary` is provided, use it instead of the current left boundary.
        """
        if left_boundary is None:
            left_boundary = self.left_boundary

        return self.sigmas[left_boundary:self.right_boundary]

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
    
    
    def step(
        self,
        model_output: torch.FloatTensor,
        sample: torch.FloatTensor,
        timestep: Optional[Union[list[float], torch.FloatTensor]],
        noise_level: Union[int, float, list[float], torch.FloatTensor] = 0.7,
        prev_sample: Optional[torch.FloatTensor] = None,
        generator: Optional[Union[torch.Generator, list[torch.Generator]]] = None
    ):
        """
        Predict the sample from the previous timestep by **reversing** the SDE. This function propagates the flow
        process from the learned model outputs (most often the predicted velocity). Specially, when noise_level is zero, the process becomes deterministic.

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned flow model.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            timestep (`float` | `torch.FloatTensor`, *optional*):
                The current discrete timestep(s) in the diffusion chain, with batch dimension. It has shape of (batch_size,).
            noise_level (`int` | `float` | `list[float]` | `torch.FloatTensor`, *optional*, defaults to 0.7):
                The noise level parameter, can be different for each sample in the batch. This parameter controls the standard deviation of the noise added to the denoised sample.
            prev_sample (`torch.FloatTensor`):
                The next insance of the sample. If given, calculate the log_prob using given `prev_sample` as predicted value.
            generator (`torch.Generator`, *optional*):
                A random number generator for SDE solving. If not given, a random generator will be used.
        """
        # bf16 can overflow here when compute prev_sample_mean, we must convert all variable to fp32
        model_output = model_output.float()
        sample = sample.float()
        if prev_sample is not None:
            prev_sample = prev_sample.float()
        
        if timestep is not None:
            if isinstance(timestep, float) or isinstance(timestep, int):
                # Make timestep a list (Batch size)
                timestep = [timestep] * model_output.shape[0]

        # Convert noise_level to a tensor with shape (batch_size, 1, 1)
        if isinstance(noise_level, float) or isinstance(noise_level, int):
            noise_level = torch.tensor([noise_level], device=sample.device, dtype=sample.dtype).repeat(sample.shape[0])
        elif isinstance(noise_level, list):
            noise_level = torch.tensor(noise_level, device=sample.device, dtype=sample.dtype)


        step_index = [self.index_for_timestep(t) for t in timestep]
        prev_step_index = [step + 1 for step in step_index]
        # sigmas is a decreasing sequence from 1 to 0, sigma=1 means pure noise, sigma=0 means pure data
        # sigma here has shape (batch_size, 1, 1)
        sigma = self.sigmas[step_index].view(-1, *([1] * (len(sample.shape) - 1)))
        sigma_prev = self.sigmas[prev_step_index].view(-1, *([1] * (len(sample.shape) - 1)))
        sigma_max = self.sigmas[1].item()
        dt = sigma_prev - sigma # dt is negative, (batch_size, 1, 1)

        noise_level = noise_level.view(-1, *([1] * (len(sample.shape) - 1)))

        std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma))) * noise_level # (batch_size, 1, 1)
        
        # our sde
        # Equation (9):
        #              sigma <-> t
        #        noise_level <-> a below Equation (9) - gives sigma_t = sqrt(t/(1-t))*a in the paper - corresponsds to std_dev_t = sqrt(sigma/(1-sigma))*noise_level here
        #                 dt <-> -\delta_t
        #       model_output <-> v_\theta(x_t, t)
        #             sample <-> x_t
        #        prev_sample <-> x_{t+\delta_t}
        #          std_dev_t <-> sigma_t

        prev_sample_mean = sample * (1 + std_dev_t**2 / (2 * sigma) * dt) + model_output * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt
        
        if prev_sample is None:
            # Non-determistic step, add noise to it
            variance_noise = randn_tensor(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            # Last term of Equation (9)
            prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-1 * dt) * variance_noise

        log_prob = (
            -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * ((std_dev_t * torch.sqrt(-1 * dt)) ** 2))
            - torch.log(std_dev_t * torch.sqrt(-1 * dt))
            - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
        )

        # mean along all but batch dimension
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
        

        # Returns x_{t+\delta_t}, log_prob, x_{t+\delta_t} mean, sigma_t
        return prev_sample, log_prob, prev_sample_mean, std_dev_t