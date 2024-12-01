import controlnet_hinter
import gradio as gr
import torch
from diffusers import DiffusionPipeline, LCMScheduler,DDIMScheduler
from diffusers import AutoPipelineForText2Image, StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image
from PIL import Image

# Mapping for ControlNet Model path and hinter function
CONTROLNET_MAPPING = {
    "canny_edge": {
        "model_id": "lllyasviel/sd-controlnet-canny",
        "hinter": controlnet_hinter.hint_canny
    },
    "pose": {
        "model_id": "lllyasviel/sd-controlnet-openpose",
        "hinter": controlnet_hinter.hint_openpose
    },
    "depth": {
        "model_id": "lllyasviel/sd-controlnet-depth",
        "hinter": controlnet_hinter.hint_depth
    },
    "scribble": {
        "model_id": "lllyasviel/sd-controlnet-scribble",
        "hinter": controlnet_hinter.hint_scribble,
    },
    "segmentation": {
        "model_id": "lllyasviel/sd-controlnet-seg",
        "hinter": controlnet_hinter.hint_segmentation,
    },
    "normal": {
        "model_id": "lllyasviel/sd-controlnet-normal",
        "hinter": controlnet_hinter.hint_normal,
    },
    "hed": {
        "model_id": "lllyasviel/sd-controlnet-hed",
        "hinter": controlnet_hinter.hint_hed,
    },
    "hough": {
        "model_id": "lllyasviel/sd-controlnet-mlsd",
        "hinter": controlnet_hinter.hint_hough,
    }
}
no_of_steps = 20

# how much would the prompt affect the final output.
# higher guidance scale means more preference given to the prompt.
guidace_scale = 7.0

# how much final output would follow the control image
controlnet_conditioning_scale=1.0


# Base model for Stable Diffusion
#base_model_path = "runwayml/stable-diffusion-v1-5"
base_model_path =  "digiplay/Juggernaut_final"

pipe=None
torch.cuda.empty_cache()
device = "cuda"

# ControlNet model setup
controlnet_type = "canny_edge"  # or other types depending on your requirement

# Loading the base model with ControlNet
controlnet = ControlNetModel.from_pretrained(CONTROLNET_MAPPING[controlnet_type]["model_id"], torch_dtype=torch.float16).to(device)
pipe = StableDiffusionControlNetPipeline.from_pretrained(base_model_path, controlnet=controlnet,torch_dtype=torch.float16).to(device)

# Describe the prompt for the logo
prompt = "Colorful, jungle surrounding, trees, natural, detailed, hd, 4k, best quality, extremely detailed"
negative_prompt = "nsfw, longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"




def generate_image(input_image, prompt, negative_prompt, num_steps=20, guidance_scale=7.5, controlnet_conditioning_scale=1.0):
    # Convert the uploaded image into a format suitable for ControlNet (e.g., canny edge)
    control_image = CONTROLNET_MAPPING[controlnet_type]["hinter"](input_image)

    # Perform the img2img generation
    output= pipe(
    prompt=prompt,
    width=512,
    height=512,
    negative_prompt=negative_prompt,
    image=control_image,
    controlnet_conditioning_scale=controlnet_conditioning_scale,
    num_inference_steps=no_of_steps,
    guidance_scale=guidace_scale,
)
    return output.images[0]


# Create Gradio interface
iface = gr.Interface(
    fn=generate_image,  # The function to wrap
    inputs=[
        gr.Image(type="pil", label="Upload Input Image"),  # Image upload
        gr.Textbox(label="Prompt", value="Colorful, jungle surrounding, trees, natural, detailed, hd, 4k, best quality, extremely detailed"),  # Text input for prompt
        gr.Textbox(label="Negative Prompt", value="nsfw, longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"),  # Text input for negative prompt
        gr.Dropdown(choices=list(CONTROLNET_MAPPING.keys()), value="canny_edge", label="Select ControlNet Type"),  # Dropdown for ControlNet type

    ],
    outputs=gr.Image(type="pil", label="Generated Image"),  # Output an image
    title="Image-to-Image Generator with Stable Diffusion and ControlNet",
    description="Upload an image, provide a text prompt, and adjust parameters to generate a modified image using Stable Diffusion and ControlNet.",
)

# Launch the app
iface.launch()
