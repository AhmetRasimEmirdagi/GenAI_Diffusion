import argparse
import os
import torch
from data_utils import DatasetLoader
from train_utils import Trainer
from inference_utils import InferencePipeline
from noise_scheduler import NoiseScheduler
from diffusers import UNet2DConditionModel

def main():
    parser = argparse.ArgumentParser(description="Train or run inference on a diffusion model.")
    parser.add_argument("--mode", type=str, choices=["train", "inference"], required=True, help="Mode to run: train or inference.")
    parser.add_argument("--data_folder", type=str, default=None, help="Folder containing .pt data files.")
    parser.add_argument("--model_path", type=str, default="model_checkpoint.pth", help="Path to save/load the model checkpoint.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training or inference.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs for training.")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to generate if data_folder is None.")
    parser.add_argument("--image_size", type=int, default=64, help="Height and width of generated images.")
    parser.add_argument("--channels", type=int, default=3, help="Number of channels in generated images.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training or inference.")
    args = parser.parse_args()

    if args.mode == "train":
        # Load dataset
        loader = DatasetLoader(
            data_folder=args.data_folder,
            signal_length=args.signal_length,
            channels=args.channels,
            device=args.device,
        )
        dataloader = loader.get_dataloader(batch_size=args.batch_size)
    
        # Model
        model = UNet2DConditionModel(
            sample_size=args.signal_length,  # Length of the 1D signals
            in_channels=args.channels,      # 1D signal input
            out_channels=args.channels,     # Output matches input
            cross_attention_dim=2,          # Polar coordinates (r, theta)
        ).to(args.device)
    
        # Noise Scheduler
        scheduler = NoiseScheduler(num_timesteps=1000)
    
        # Optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
        # Trainer
        trainer = Trainer(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=args.device,
        )

        # Train the model
        trainer.train(num_epochs=args.num_epochs)

        # Save the model
        torch.save(model, args.model_path)
        print(f"Model saved to {args.model_path}")

    elif args.mode == "inference":
        # Load the trained model
        pipeline = InferencePipeline(model_path=args.model_path, device=args.device)

        # Load dataset for inference
        loader = DatasetLoader(data_folder=args.data_folder)
        dataloader = loader.get_dataloader(batch_size=args.batch_size, shuffle=False)

        # Run inference
        outputs = pipeline.run_inference(dataloader)

        # Postprocess results
        results = pipeline.postprocess(outputs)
        print("Inference completed. Postprocessed results:")
        print(results)

if __name__ == "__main__":
    main()
