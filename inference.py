import argparse
import torch
from utils.inference_utils import InferencePipeline
from utils.data_utils import DatasetLoader
from utils.visualization_utils import plot_real_vs_generated_signals, plot_tx_rx_map


def main(args):
    # Set device
    device = "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    print(f"Using device: {device}")

    # Load dataset
    print("Loading dataset...")
    dataset_loader = DatasetLoader(
        data_folder=args.data_folder,
        batch_size=args.batch_size,
        shuffle=False,
        normalize=True
    )
    data_loader = dataset_loader.get_dataloader()

    # Initialize inference pipeline
    print("Loading model...")
    inference_pipeline = InferencePipeline(model_path=args.model_path, device=device)

    # Run inference
    print("Running inference...")
    outputs = inference_pipeline.run_inference(data_loader)

    # Visualize results
    if args.visualize:
        print("Visualizing results...")
        for batch_idx, batch in enumerate(data_loader):
            real_data = batch["data"]
            generated_data = outputs[batch_idx]

            # Plot TX/RX map
            if "rx" in batch and "tx" in batch:
                plot_tx_rx_map(batch["rx"], batch["tx"], args.tx_loc)

            # Compare real and generated signals
            plot_real_vs_generated_signals(real_data[0], generated_data[0])  # Visualize the first item in the batch

            if batch_idx == args.num_visualizations - 1:
                break  # Stop after visualizing a limited number of batches


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference pipeline for the diffusion model.")
    parser.add_argument("--data_folder", type=str, required=True, help="Path to the folder containing .pt files.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for inference ('cuda' or 'cpu').")
    parser.add_argument("--visualize", action="store_true", help="Whether to visualize the results.")
    parser.add_argument("--tx_loc", type=list, default=[0, 0], help="TX location for visualization (e.g., [x, y]).")
    parser.add_argument("--num_visualizations", type=int, default=5, help="Number of batches to visualize.")

    args = parser.parse_args()
    main(args)
