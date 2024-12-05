import torch

class NoiseScheduler:
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        """
        Initializes the Noise Scheduler for a denoising diffusion model.

        Args:
            num_timesteps (int): Number of timesteps for the diffusion process.
            beta_start (float): Starting value of beta (noise variance).
            beta_end (float): Ending value of beta (noise variance).
        """
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        # Linear schedule for betas
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat((torch.tensor([1.0]), self.alphas_cumprod[:-1]))

        # Precompute terms for reverse diffusion
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def add_noise(self, x_start, t):
        """
        Adds noise to the original sample x_start at a given timestep t.

        Args:
            x_start (torch.Tensor): The original clean data.
            t (torch.Tensor): Timesteps for which to add noise.

        Returns:
            torch.Tensor: The noised sample x_t.
        """
        noise = torch.randn_like(x_start)
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, *([1] * (x_start.dim() - 1)))
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(
            -1, *([1] * (x_start.dim() - 1))
        )
        x_t = sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise
        return x_t, noise

    def reverse_noise(self, x_t, t, predicted_noise):
        """
        Removes noise from the noised sample x_t to approximate x_start.

        Args:
            x_t (torch.Tensor): The noised data.
            t (torch.Tensor): Timesteps at which to reverse noise.
            predicted_noise (torch.Tensor): Predicted noise from the model.

        Returns:
            torch.Tensor: The denoised sample x_start_hat.
        """
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, *([1] * (x_t.dim() - 1)))
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(
            -1, *([1] * (x_t.dim() - 1))
        )
        x_start_hat = (x_t - sqrt_one_minus_alpha_cumprod_t * predicted_noise) / sqrt_alpha_cumprod_t
        return x_start_hat
