import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from diffusers import UNet2DConditionModel
from tqdm import tqdm
import wandb


class GlobalNormalizationLayer(nn.Module):
    """
    A global normalization layer to normalize input data.
    """
    def __init__(self):
        super(GlobalNormalizationLayer, self).__init__()

    def forward(self, x):
        """
        Normalize input data globally across each sample.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, length).

        Returns:
            torch.Tensor: Globally normalized tensor.
        """
        return (x - x.mean(dim=(1, 2), keepdim=True)) / (x.std(dim=(1, 2), keepdim=True) + 1e-6)


class Trainer:
    def __init__(
        self, 
        model, 
        dataloader, 
        optimizer, 
        scheduler, 
        device="cuda", 
        num_timesteps=1000, 
        val_dataloader=None, 
        checkpoint_dir="checkpoints",
        wandb_logger=None,
        gradient_clipping=1.0,
    ):
        """
        Initializes the Trainer class for training a diffusion model.

        Args:
            model (nn.Module): The diffusion model (e.g., UNet2DConditionModel).
            dataloader (DataLoader): DataLoader for training data.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            scheduler (NoiseScheduler): Noise scheduler for diffusion process.
            device (str): Device to use ("cuda" or "cpu").
            num_timesteps (int): Number of timesteps for the diffusion process.
            val_dataloader (DataLoader): Optional DataLoader for validation data.
            checkpoint_dir (str): Directory to save checkpoints.
            wandb_logger: W&B logger for tracking experiments.
            gradient_clipping (float): Maximum gradient norm for clipping.
        """
        self.model = model
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_timesteps = num_timesteps
        self.loss_fn = nn.MSELoss()  # Loss function
        self.normalization_layer = GlobalNormalizationLayer().to(device)
        self.checkpoint_dir = checkpoint_dir
        self.wandb_logger = wandb_logger
        self.gradient_clipping = gradient_clipping

        # Create checkpoint directory if it doesn't exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train(self, num_epochs, save_every=5):
        """
        Train the diffusion model.

        Args:
            num_epochs (int): Number of epochs to train the model.
            save_every (int): Frequency (in epochs) to save model checkpoints.

        Returns:
            None
        """
        self.model.train()
        best_val_loss = float("inf")  # Track best validation loss for saving the best model

        for epoch in range(num_epochs):
            epoch_loss = 0
            tqdm_loader = tqdm(self.dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
            for batch in tqdm_loader:
                try:
                    # Extract data and conditions
                    x_start = batch["data"].to(self.device)  # Input signals
                    rx = batch["rx"].to(self.device)        # Receiver coordinates
                    tx = batch["tx"].to(self.device)        # Transmitter coordinates

                    # Compute distance and angle (polar coordinates)
                    distance = torch.sqrt((rx[:, 0] - tx[:, 0]) ** 2 + (rx[:, 1] - tx[:, 1]) ** 2)
                    angle = torch.atan2(rx[:, 1] - tx[:, 1], rx[:, 0] - tx[:, 0])
                    conditions = torch.stack([distance, angle], dim=1).to(self.device)

                    # Normalize the input data
                    x_start = self.normalization_layer(x_start)

                    # Sample random timesteps
                    t = torch.randint(0, self.num_timesteps, (x_start.size(0),), device=self.device)

                    # Add noise
                    x_t, noise = self.scheduler.add_noise(x_start, t)

                    # Predict noise
                    predicted_noise = self.model(x_t, timesteps=t, encoder_hidden_states=conditions).sample

                    # Compute loss
                    loss = self.loss_fn(predicted_noise, noise)
                    epoch_loss += loss.item()

                    # Backpropagation
                    self.optimizer.zero_grad()
                    loss.backward()

                    # Gradient clipping
                    if self.gradient_clipping:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)

                    self.optimizer.step()

                    # Update progress bar
                    tqdm_loader.set_postfix({"Loss": loss.item()})

                except Exception as e:
                    print(f"Error during training step: {e}")

            # Log training loss to W&B
            epoch_loss /= len(self.dataloader)
            if self.wandb_logger:
                self.wandb_logger.log({"train_loss": epoch_loss})

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.6f}")

            # Validation step
            if self.val_dataloader:
                val_loss = self.validate()
                if self.wandb_logger:
                    self.wandb_logger.log({"val_loss": val_loss})
                print(f"Validation Loss: {val_loss:.6f}")

                # Save the best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_path = os.path.join(self.checkpoint_dir, "best_model.pth")
                    torch.save(self.model.state_dict(), best_model_path)
                    print(f"Best model saved to {best_model_path}")

            # Save model checkpoint periodically
            if (epoch + 1) % save_every == 0:
                checkpoint_path = os.path.join(self.checkpoint_dir, f"model_epoch_{epoch + 1}.pth")
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")

    def validate(self):
        """
        Validate the model on the validation dataset.

        Returns:
            float: Average validation loss.
        """
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validating"):
                x_start = batch["data"].to(self.device)
                rx = batch["rx"].to(self.device)
                tx = batch["tx"].to(self.device)

                # Compute distance and angle (polar coordinates)
                distance = torch.sqrt((rx[:, 0] - tx[:, 0]) ** 2 + (rx[:, 1] - tx[:, 1]) ** 2)
                angle = torch.atan2(rx[:, 1] - tx[:, 1], rx[:, 0] - tx[:, 0])
                conditions = torch.stack([distance, angle], dim=1).to(self.device)

                # Normalize the input data
                x_start = self.normalization_layer(x_start)

                # Sample random timesteps
                t = torch.randint(0, self.num_timesteps, (x_start.size(0),), device=self.device)

                # Add noise
                x_t, noise = self.scheduler.add_noise(x_start, t)

                # Predict noise
                predicted_noise = self.model(x_t, timesteps=t, encoder_hidden_states=conditions).sample

                # Compute loss
                loss = self.loss_fn(predicted_noise, noise)
                val_loss += loss.item()

        return val_loss / len(self.val_dataloader) if self.val_dataloader else 0.0
