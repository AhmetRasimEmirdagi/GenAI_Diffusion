import argparse
import torch
import wandb
from data_loader import get_dataloader
from train_utils import Trainer
from inference_utils import InferencePipeline
from noise_scheduler import NoiseScheduler
from diffusers import UNet2DConditionModel

def main():
    parser = argparse.ArgumentParser(description="Train or run inference on a diffusion model.")
    parser.add_argument("--mode", type=str, choices=["train", "inference"], required=True, help="Mode to run: train or inference.")
    parser.add_argument("--data_folder", type=str, required=True, help="Folder containing .pt signal data files.")
    parser.add_argument("--rx_file", type=str, required=True, help="Path to the file containing receiver coordinates.")
    parser.add_argument("--config_file", type=str, required=True, help="Path to the JSON config file containing transmitter coordinates.")
    parser.add_argument("--model_path", type=str, default="model_checkpoint.pth", help="Path to save/load the model checkpoint.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training or inference.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs for training.")
    parser.add_argument("--num_timesteps", type=int, default=1000, help="Number of timesteps for the diffusion process.")
    parser.add_argument("--signal_length", type=int, default=250, help="Length of the 1D signal.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training or inference.")
    parser.add_argument("--project_name", type=str, default="diffusion-model", help="W&B project name.")
    args = parser.parse_args()

    if args.mode == "train":
        # Initialize Weights & Biases
        wandb.init(
            project=args.project_name,
            config={
                "batch_size": args.batch_size,
                "num_epochs": args.num_epochs,
                "learning_rate": 1e-4,
                "num_timesteps": args.num_timesteps,
                "signal_length": args.signal_length,
            },
        )

        # Create DataLoader for training
        dataloader = get_dataloader(
            data_folder=args.data_folder,
            rx_file=args.rx_file,
            config_file=args.config_file,
            batch_size=args.batch_size,
        )

        # Model
        model = UNet2DConditionModel(
            sample_size=args.signal_length,
            in_channels=1,  # Single channel for 1D signals
            out_channels=1,
            cross_attention_dim=2,  # Polar coordinates: r and theta
        ).to(args.device)

        # Noise Scheduler
        scheduler = NoiseScheduler(num_timesteps=args.num_timesteps)

        # Optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Trainer
        trainer = Trainer(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=args.device,
            wandb_logger=wandb,  # Pass W&B logger to the trainer
        )

        # Train the model
        trainer.train(num_epochs=args.num_epochs)

        # Save the model
        torch.save(model.state_dict(), args.model_path)
        wandb.save(args.model_path)  # Save model checkpoint to W&B
        print(f"Model saved to {args.model_path}")

    elif args.mode == "inference":
        # Load the trained model
        pipeline = InferencePipeline(model_path=args.model_path, device=args.device)

        # Create DataLoader for inference
        dataloader = get_dataloader(
            data_folder=args.data_folder,
            rx_file=args.rx_file,
            config_file=args.config_file,
            batch_size=args.batch_size,
        )

        # Run inference
        outputs = pipeline.run_inference(dataloader)

        # Postprocess results
        results = pipeline.postprocess(outputs)
        print("Inference completed. Postprocessed results:")
        print(results)

if __name__ == "__main__":
    main()
