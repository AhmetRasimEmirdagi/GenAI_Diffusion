import os
import torch
from torch.utils.data import Dataset, DataLoader


class PolarCoordinatesDataset(Dataset):
    """
    Custom dataset for training a diffusion model. Each sample includes data, RX, and TX coordinates.

    Args:
        data (torch.Tensor): Tensor of input data of shape (N, C, H, W).
        rx_coords (torch.Tensor): Tensor of RX coordinates of shape (N, 2).
        tx_coords (torch.Tensor): Tensor of TX coordinates of shape (N, 2).
    """
    def __init__(self, data, rx_coords, tx_coords):
        super().__init__()
        self.data = data
        self.rx_coords = rx_coords
        self.tx_coords = tx_coords

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "data": self.data[idx],
            "rx": self.rx_coords[idx],
            "tx": self.tx_coords[idx],
        }


class DatasetLoader:
    """
    Handles dataset loading and management from various sources (e.g., `.pt` files or generated data).
    """
    def __init__(self, data_folder=None, num_samples=100, image_size=(64, 64), channels=3, device="cpu"):
        """
        Initializes the dataset loader.

        Args:
            data_folder (str, optional): Folder containing `.pt` files. If None, generates a new dataset.
            num_samples (int): Number of samples to generate if `data_folder` is None.
            image_size (tuple): Size of the generated images (height, width).
            channels (int): Number of channels in the generated images.
            device (str): Device for storing tensors.
        """
        self.data_folder = data_folder
        self.num_samples = num_samples
        self.image_size = image_size
        self.channels = channels
        self.device = device

    def get_pt_files(self):
        """Get a list of all .pt files in the specified folder."""
        if not self.data_folder or not os.path.exists(self.data_folder):
            raise RuntimeError(f"Data folder '{self.data_folder}' does not exist.")
        return [f for f in os.listdir(self.data_folder) if f.endswith(".pt")]

    def read_pt_file(self, file_path):
        """Load a .pt file and return its content."""
        try:
            data = torch.load(file_path)
            return data
        except Exception as e:
            raise RuntimeError(f"Error reading {file_path}: {str(e)}")

    def load_dataset_from_pt_files(self):
        """
        Load a dataset from `.pt` files in the data folder.

        Returns:
            PolarCoordinatesDataset: A dataset loaded from the `.pt` files.
        """
        pt_files = self.get_pt_files()
        data_list = []
        rx_coords = []
        tx_coords = []

        for pt_file in pt_files:
            file_path = os.path.join(self.data_folder, pt_file)
            try:
                file_data = self.read_pt_file(file_path)
                data_list.append(file_data["data"])
                rx_coords.append(file_data["rx"])
                tx_coords.append(file_data["tx"])
            except Exception as e:
                print(f"Skipping file {file_path} due to error: {str(e)}")

        if not data_list:
            raise RuntimeError(f"No valid .pt files found in {self.data_folder}")

        # Concatenate all data into single tensors
        data = torch.cat(data_list, dim=0)
        rx_coords = torch.cat(rx_coords, dim=0)
        tx_coords = torch.cat(tx_coords, dim=0)

        return PolarCoordinatesDataset(data, rx_coords, tx_coords)

    def generate_polar_dataset(self):
        """
        Generate a dataset with random data and random TX/RX coordinates.

        Returns:
            PolarCoordinatesDataset: A dataset with the generated samples.
        """
        data = torch.rand((self.num_samples, self.channels, *self.image_size), device=self.device)
        rx_coords = torch.rand((self.num_samples, 2), device=self.device)  # RX coordinates (x, y)
        tx_coords = torch.rand((self.num_samples, 2), device=self.device)  # TX coordinates (x, y)
        return PolarCoordinatesDataset(data, rx_coords, tx_coords)

    def get_dataloader(self, batch_size=8, shuffle=True):
        """
        Create a DataLoader for the dataset.

        Args:
            batch_size (int): Batch size for the DataLoader.
            shuffle (bool): Whether to shuffle the DataLoader.

        Returns:
            DataLoader: A DataLoader instance for the dataset.
        """
        if self.data_folder:
            dataset = self.load_dataset_from_pt_files()
        else:
            dataset = self.generate_polar_dataset()

        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def summarize_dataset(data_folder):
    """
    Summarize the dataset by inspecting all .pt files in the folder.

    Parameters:
    - data_folder: Path to the folder containing .pt files.

    Returns:
    - summary: List of dictionaries containing information about each file.
    """
    loader = DatasetLoader(data_folder=data_folder)
    pt_files = loader.get_pt_files()
    summary = []

    for pt_file in pt_files:
        file_path = os.path.join(data_folder, pt_file)
        try:
            data = loader.read_pt_file(file_path)
            summary.append({
                "file": pt_file,
                "type": type(data).__name__,
                "keys": list(data.keys()),
                "data_shape": data["data"].shape if "data" in data else None,
                "rx_shape": data["rx"].shape if "rx" in data else None,
                "tx_shape": data["tx"].shape if "tx" in data else None,
            })
        except Exception as e:
            summary.append({
                "file": pt_file,
                "error": str(e)
            })

    return summary
