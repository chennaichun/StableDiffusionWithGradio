# **Image-to-Image Generator with Stable Diffusion and ControlNet**

![Generated Output](https://github.com/chennaichun/StableDiffusionWithGradio/blob/main/output2.png)

This project demonstrates the integration of **Stable Diffusion** with **ControlNet** for advanced image-to-image generation. Users can upload an image, provide text prompts, and adjust parameters to create customized outputs using **Gradio** as the interface.

---

## **Usage**
1. Clone this repository and install the dependencies.
2. Run the Python script to start the Gradio interface.
3. Upload an input image, provide a descriptive prompt, and adjust parameters to generate your desired output.

---

## **Features**
- **ControlNet Integration**: Generate images with specific control conditions like canny edge, depth, pose, and more.
- **Customizable Prompts**: Fine-tune the generated output using text-based prompts and negative prompts.
- **Interactive UI**: Utilize a user-friendly Gradio interface for real-time interaction.
- **Base Model Support**: Uses the pre-trained `digiplay/Juggernaut_final` or other compatible Stable Diffusion models.

---

## **How It Works**

### **ControlNet Setup**
1. Choose a ControlNet type (e.g., `canny_edge`, `pose`, `depth`, etc.).
2. The system maps the input image to a control format using `controlnet_hinter`.

### **Image Generation**
- Generates high-quality images based on the uploaded input, text prompts, and control conditions.
- Parameters like `guidance scale` and `control conditioning scale` allow for output customization.

### **Gradio Interface**
1. Upload an input image.
2. Provide a **Prompt** (descriptive text) and **Negative Prompt** (undesirable attributes).
3. Select the desired **ControlNet Type** from a dropdown menu.

---

## **Code Overview**

### **ControlNet Types**
The following ControlNet types are available:

| ControlNet Type | Model Path                              | Hinter Function              |
|------------------|----------------------------------------|------------------------------|
| `canny_edge`     | `lllyasviel/sd-controlnet-canny`       | `hint_canny`                 |
| `pose`           | `lllyasviel/sd-controlnet-openpose`    | `hint_openpose`              |
| `depth`          | `lllyasviel/sd-controlnet-depth`       | `hint_depth`                 |
| `scribble`       | `lllyasviel/sd-controlnet-scribble`    | `hint_scribble`              |
| `segmentation`   | `lllyasviel/sd-controlnet-seg`         | `hint_segmentation`          |
| `normal`         | `lllyasviel/sd-controlnet-normal`      | `hint_normal`                |
| `hed`            | `lllyasviel/sd-controlnet-hed`        | `hint_hed`                   |
| `hough`          | `lllyasviel/sd-controlnet-mlsd`       | `hint_hough`                 |

---

## **Gradio Interface**
The Gradio interface allows you to:
- Upload an input image.
- Enter prompts and negative prompts.
- Select the desired **ControlNet Type**.
- View the generated output.

---

## **Customization**
You can customize the base model, prompts, and ControlNet types to suit your specific use case. Adjust the following parameters in the script:

- **Base Model Path**: Modify `base_model_path` to use a different pre-trained Stable Diffusion model.
- **ControlNet Type**: Change the `controlnet_type` variable to select a different mapping.
- **Prompts**: Customize the `prompt` and `negative_prompt` strings.

---

## **License**
This project uses open-source models from Hugging Face and ControlNet. Ensure compliance with the respective licenses when using the models.
