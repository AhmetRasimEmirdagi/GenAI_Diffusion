import torch
import os
from data_utils import DatasetLoader

class InferencePipeline:
    """
    Pipeline for performing inference using a trained diffusion model.
    """
    def __init__(self, model_path, device="cuda"):
        """
        Initialize the pipeline with a trained model.

        Args:
            model_path (str): Path to the trained model checkpoint.
            device (str): Device to use for inference.
        """
        self.device = device
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        """
        Load the trained model from a checkpoint.

        Args:
            model_path (str): Path to the model checkpoint.

        Returns:
            torch.nn.Module: Loaded model.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
        model = torch.load(model_path, map_location=self.device)
        model.eval()  # Set to evaluation mode
        return model

    def preprocess(self, input_data):
        """
        Preprocess input data for inference.

        Args:
            input_data (torch.Tensor): Raw input data.

        Returns:
            torch.Tensor: Preprocessed data.
        """
        self.mean = input_data.mean()
        self.std = input_data.std()
        return (input_data - self.mean) / (self.std + 1e-6)  # Global normalization

    def run_inference(self, data_loader):
        """
        Run inference on a DataLoader.

        Args:
            data_loader (DataLoader): DataLoader containing the input data.

        Returns:
            list: List of outputs from the model.
        """
        outputs = []
        with torch.no_grad():
            for batch in data_loader:
                data = batch["data"].to(self.device)
                preprocessed_data = self.preprocess(data)
                output = self.model(preprocessed_data)
                outputs.append(output.cpu())
        return outputs

    def postprocess(self, outputs):
        """
        Postprocess model outputs for interpretability by reversing normalization.

        Args:
            outputs (list): Model outputs.

        Returns:
            list: Postprocessed results.
        """
        # Inverse the global normalization
        return [(output * self.std + self.mean).numpy() for output in outputs]
