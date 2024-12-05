import os
import torch
import json
from torch.utils.data import Dataset, DataLoader
from data_utils import read_pt_file


class SignalDataset(Dataset):
    """
    Dataset class for loading 1D signals and their associated TX and RX coordinates.
    """
    def __init__(self, data_folder, rx_file, config_file):
        """
        Initialize the dataset.

        Args:
            data_folder (str): Path to the folder containing .pt signal files.
            rx_file (str): Path to the file containing receiver coordinates.
            config_file (str): Path to the JSON config file containing TX coordinates.
        """
        self.data_folder = data_folder
        self.rx_coordinates = self.load_coordinates(rx_file)
        self.tx_coordinate = self.load_tx_from_config(config_file)
        self.signal_files = self.get_signal_files(data_folder)

        # Ensure all components have the correct length
        if len(self.signal_files) != len(self.rx_coordinates):
            raise ValueError("Number of signals does not match number of RX coordinates.")

    @staticmethod
    def load_coordinates(file_path):
        """
        Load coordinates from a .pt file.

        Args:
            file_path (str): Path to the .pt file containing coordinates.

        Returns:
            torch.Tensor: Tensor of coordinates.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Coordinates file not found: {file_path}")
        return torch.load(file_path)  # Assumes .pt file contains tensor of shape (N, 2)

    @staticmethod
    def load_tx_from_config(config_file):
        """
        Load the TX coordinates from a JSON configuration file.

        Args:
            config_file (str): Path to the JSON config file.

        Returns:
            torch.Tensor: Tensor containing the TX coordinate.
        """
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file not found: {config_file}")
        with open(config_file, "r") as f:
            config = json.load(f)
        
        tx_coordinate = config.get("tx")
        if tx_coordinate is None or not isinstance(tx_coordinate, list) or len(tx_coordinate) != 2:
            raise ValueError("TX coordinate is missing or invalid in the config file.")
        
        return torch.tensor(tx_coordinate, dtype=torch.float32)

    @staticmethod
    def get_signal_files(data_folder):
        """
        Get a list of all .pt signal files in the specified folder.

        Args:
            data_folder (str): Path to the folder containing .pt files.

        Returns:
            list: List of paths to .pt files.
        """
        return [
            os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith(".pt")
        ]

    def __len__(self):
        """
        Get the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.signal_files)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: A dictionary containing the signal, RX coordinate, and TX coordinate.
        """
        signal_file = self.signal_files[idx]
        signal = read_pt_file(signal_file)  # Assumes signal is stored in .pt file

        rx_coordinate = self.rx_coordinates[idx]
        tx_coordinate = self.tx_coordinate  # TX is global and constant

        return {
            "data": signal,  # 1D signal
            "rx": rx_coordinate,  # Receiver coordinate
            "tx": tx_coordinate,  # Transmitter coordinate
        }


def get_dataloader(data_folder, rx_file, config_file, batch_size, shuffle=True):
    """
    Create a DataLoader for the SignalDataset.

    Args:
        data_folder (str): Path to the folder containing .pt signal files.
        rx_file (str): Path to the .pt file containing receiver coordinates.
        config_file (str): Path to the JSON config file containing TX coordinates.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle the data.

    Returns:
        DataLoader: DataLoader for the SignalDataset.
    """
    dataset = SignalDataset(data_folder=data_folder, rx_file=rx_file, config_file=config_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
