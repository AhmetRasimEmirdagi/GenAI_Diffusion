# Diffusion Model Project

## Project Overview

This project implements a diffusion model with a `UNet2DCondition` backbone, designed to work with datasets involving polar coordinates. The project includes:

- Dataset loading and preprocessing (`utils.data_utils`).
- Training with a noise scheduler (`utils.train_utils`).
- Inference with preprocessing and postprocessing (`utils.inference_utils`).
- Visualization tools (`utils.visualization_utils`).

---

## File Structure

```
.
├── main.py                    # Main script for training and inference
├── requirements.txt           # Python dependencies
├── utils/                     # Utility functions and modules
│   ├── __init__.py            # Makes `utils` a Python package
│   ├── data_utils.py          # Dataset management utilities
│   ├── inference_utils.py     # Inference pipeline and utilities
│   ├── train_utils.py         # Training utilities and Trainer class
│   ├── noise_scheduler.py     # Noise scheduler for diffusion training
│   ├── visualization_utils.py # Visualization tools
├── data/                      # Folder for dataset .pt files
└── README.md                  # Documentation
```

---

## Installation

Run the following command to install the dependencies:

```
pip install -r requirements.txt
```

**Dependencies**:
- `torch`: PyTorch for model development.
- `diffusers`: Prebuilt diffusion models like `UNet2DCondition`.
- `matplotlib`: Visualization of results.

---

## Usage

### 1. Dataset Preparation

#### Loading `.pt` Files

Place your `.pt` files in the `data/` folder. Each file should be structured like this:

```
{
    "data": torch.Tensor,  # Input data (e.g., signals or images)
    "rx": torch.Tensor,    # RX coordinates (e.g., [x, y])
    "tx": torch.Tensor     # TX coordinates (e.g., [x, y])
}
```

#### Generating Synthetic Data

If `.pt` files are unavailable, synthetic data will be generated dynamically using the `DatasetLoader` in `utils.data_utils`.

---

### 2. Training

Run the following command to train the model:

```
python main.py --mode train --data_folder data --num_epochs 20 --batch_size 16 --model_path model_checkpoint.pth
```

**Arguments**:
- `--mode train`: Train the diffusion model.
- `--data_folder`: Path to the folder containing `.pt` files.
- `--num_epochs`: Number of training epochs (default: `10`).
- `--batch_size`: Training batch size (default: `8`).
- `--model_path`: Path to save the trained model.

---

### 3. Inference

Run the following command to perform inference:

```
python main.py --mode inference --data_folder data --model_path model_checkpoint.pth
```

**Arguments**:
- `--mode inference`: Run the inference pipeline.
- `--data_folder`: Path to the folder containing `.pt` files.
- `--model_path`: Path to the trained model checkpoint.

---

### 4. Visualization

The `utils.visualization_utils` module includes the following tools:

#### Plot RX/TX Locations:

```python
from utils.visualization_utils import plot_tx_rx_map
plot_tx_rx_map(rx_x, rx_y, tx_loc)
```

#### Compare Real vs. Generated Signals:

```python
from utils.visualization_utils import plot_signals
plot_signals(real_signal, generated_signal)
```

---

## Example Workflow

1. **Prepare Dataset**:
   - Place `.pt` files in the `data/` folder, or let synthetic data be generated.

2. **Train the Model**:
   ```
   python main.py --mode train --data_folder data --num_epochs 20 --batch_size 16
   ```

3. **Run Inference**:
   ```
   python main.py --mode inference --data_folder data --model_path model_checkpoint.pth
   ```

4. **Visualize Results**:
   ```python
   from utils.visualization_utils import plot_signals, plot_tx_rx_map
   plot_signals(real_signal, generated_signal)
   plot_tx_rx_map(rx_x, rx_y, tx_loc)
   ```

---

## Key Components

### `DatasetLoader` (`utils.data_utils`)

Handles loading `.pt` files or generating synthetic data dynamically. Provides:
- Preprocessing.
- Normalization.
- PyTorch `DataLoader` creation.

### `NoiseScheduler` (`utils.noise_scheduler`)

Manages Gaussian noise schedules, enabling the diffusion process:
- Linear scheduling of `beta` values.
- Precomputed cumulative products for efficient sampling.

### `Trainer` (`utils.train_utils`)

Encapsulates the training process:
- Applies noise to the input via `NoiseScheduler`.
- Optimizes the `UNet2DCondition` model using AdamW.
- Saves checkpoints for later use.

### `InferencePipeline` (`utils.inference_utils`)

Performs:
- Input preprocessing (global normalization).
- Model inference.
- Output postprocessing (inverse normalization).

### Visualization Tools (`utils.visualization_utils`)

Includes tools for:
- **Mapping RX/TX Locations** (`plot_tx_rx_map`).
- **Comparing Signals** (`plot_signals`).

---

## Future Improvements

1. **Logging**:
   - Integrate TensorBoard or Weights & Biases for training/inference tracking.

2. **Advanced Noise Schedules**:
   - Add cosine or exponential noise schedules for better performance.

3. **Pretrained Models**:
   - Integrate transfer learning using pretrained diffusion models.

4. **Hyperparameter Tuning**:
   - Add support for automated hyperparameter optimization (e.g., with `Optuna`).

5. **Enhanced Visualization**:
   - Use interactive libraries like `Plotly` or `Bokeh` for visualizing RX/TX maps and signal comparisons.

---
