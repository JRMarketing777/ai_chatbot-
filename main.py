import os
from huggingface_hub import InferenceApi
import gradio as gr

# Initialize the Inference API with your model and API token
API_TOKEN = os.getenv("HF_API_TOKEN")  # Store your API token in an environment variable for security
inference = InferenceApi(repo_id="meta-llama/Meta-Llama-3.1-405B", token=API_TOKEN)

# Function to perform inference using the Inference API
def perform_inference(input_text):
    result = inference(inputs=input_text)
    return result

# Define a Gradio interface
def gradio_interface(input_text):
    result = perform_inference(input_text)
    return result

# Create the Gradio UI
iface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.inputs.Textbox(lines=2, placeholder="Enter your text here..."),
    outputs="text",
    title="Hugging Face Inference API",
    description="Enter text to get predictions from the Hugging Face model."
)

# Launch the Gradio interface
if __name__ == "__main__":
    iface.launch()
