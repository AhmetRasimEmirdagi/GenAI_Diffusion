import gradio as gr
import torch
from utils.inference_utils import InferencePipeline
from utils.noise_scheduler import NoiseScheduler

# Load the trained model and initialize the inference pipeline
MODEL_PATH = "path_to_trained_model.pth"  # Replace with your model's path
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

pipeline = InferencePipeline(model_path=MODEL_PATH, device=DEVICE)

def generate_channel_response(r, theta):
    """
    Generate channel response for given polar coordinates (r, theta).
    
    Args:
        r (float): Radius in polar coordinates.
        theta (float): Angle in polar coordinates (in degrees).

    Returns:
        List[float]: Generated channel response.
    """
    # Convert polar coordinates to Cartesian (if needed for model input)
    r_tensor = torch.tensor([[r]], dtype=torch.float32, device=DEVICE)
    theta_tensor = torch.tensor([[theta]], dtype=torch.float32, device=DEVICE)
    
    # Combine into model input format (customize based on your model)
    input_data = torch.cat((r_tensor, theta_tensor), dim=-1)
    
    # Perform inference
    generated_response = pipeline.model(input_data).cpu().detach().numpy()
    
    # Return the generated channel response as a list
    return generated_response.tolist()

# Gradio Interface
description = """
# Channel Response Generator
Enter the polar coordinates (r and theta) to generate the corresponding channel response using a trained diffusion model.
"""

# Input and output components
inputs = [
    gr.inputs.Number(label="Radius (r)"),
    gr.inputs.Number(label="Angle (theta, in degrees)")
]
outputs = gr.outputs.Textbox(label="Generated Channel Response")

# Create Gradio interface
interface = gr.Interface(
    fn=generate_channel_response,
    inputs=inputs,
    outputs=outputs,
    title="Channel Response Generator",
    description=description,
    theme="default"
)

# Launch the Gradio app
if __name__ == "__main__":
    interface.launch(share = True)
