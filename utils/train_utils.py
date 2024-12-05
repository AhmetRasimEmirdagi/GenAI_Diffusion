import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from diffusers import UNet2DConditionModel
from tqdm import tqdm

class GlobalNormalizationLayer(nn.Module):
    """
    A global normalization layer to normalize input data.
    """
    def __init__(self):
        super(GlobalNormalizationLayer, self).__init__()

    def forward(self, x):
        return (x - x.mean(dim=(1, 2, 3), keepdim=True)) / (x.std(dim=(1, 2, 3), keepdim=True) + 1e-6)


class Trainer:
    def __init__(self, model, dataloader, optimizer, scheduler, device="cuda", num_timesteps=1000):
        """
        Initializes the Trainer class for training a diffusion model.

        Args:
            model (nn.Module): The diffusion model (e.g., UNet2DConditionModel).
            dataloader (DataLoader): DataLoader for training data.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            scheduler (NoiseScheduler): Noise scheduler for diffusion process.
            device (str): Device to use ("cuda" or "cpu").
            num_timesteps (int): Number of timesteps for the diffusion process.
        """
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_timesteps = num_timesteps
        self.loss_fn = nn.MSELoss()  # Loss function
        self.normalization_layer = GlobalNormalizationLayer().to(device)

    def train(self, num_epochs):
        """
        Train the diffusion model.

        Args:
            num_epochs (int): Number of epochs to train the model.

        Returns:
            None
        """
        self.model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0
            tqdm_loader = tqdm(self.dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
            for batch in tqdm_loader:
                # Extract data and conditions
                x_start, rx, tx = batch["data"], batch["rx"], batch["tx"]
                distance = torch.sqrt((rx[:, 0] - tx[:, 0]) ** 2 + (rx[:, 1] - tx[:, 1]) ** 2)
                angle = torch.atan2(rx[:, 1] - tx[:, 1], rx[:, 0] - tx[:, 0])
                conditions = torch.stack([distance, angle], dim=1).to(self.device)

                # Move data to the device and normalize
                x_start = self.normalization_layer(x_start.to(self.device))

                # Sample random timesteps
                t = torch.randint(0, self.num_timesteps, (x_start.size(0),), device=self.device)

                # Add noise
                x_t, noise = self.scheduler.add_noise(x_start, t)

                # Predict noise
                predicted_noise = self.model(x_t, timesteps=t, class_labels=conditions).sample

                # Compute loss
                loss = self.loss_fn(predicted_noise, noise)
                epoch_loss += loss.item()

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(self.dataloader)}")

