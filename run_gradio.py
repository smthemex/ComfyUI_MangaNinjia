import gradio as gr
import numpy as np
from PIL import Image, ImageDraw
import cv2
import gradio as gr
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
import numpy as np
import os
import re
from PIL import Image, ImageDraw
import cv2
#
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
import torch.nn as nn
from inference.manganinjia_pipeline import MangaNinjiaPipeline
from diffusers import (
    ControlNetModel,
    DiffusionPipeline,
    DDIMScheduler,
    AutoencoderKL,
)
from src.models.mutual_self_attention_multi_scale import ReferenceAttentionControl
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.refunet_2d_condition import RefUNet2DConditionModel
from src.point_network import PointNet
from src.annotator.lineart import BatchLineartDetector
val_configs = OmegaConf.load('./configs/inference.yaml')
# === load the checkpoint ===
pretrained_model_name_or_path = val_configs.model_path.pretrained_model_name_or_path
refnet_clip_vision_encoder_path = val_configs.model_path.clip_vision_encoder_path
controlnet_clip_vision_encoder_path = val_configs.model_path.clip_vision_encoder_path
controlnet_model_name_or_path = val_configs.model_path.controlnet_model_name
annotator_ckpts_path = val_configs.model_path.annotator_ckpts_path

output_root = val_configs.inference_config.output_path
device = val_configs.inference_config.device
preprocessor = BatchLineartDetector(annotator_ckpts_path)
in_channels_reference_unet = 4
in_channels_denoising_unet = 4
in_channels_controlnet = 4
noise_scheduler = DDIMScheduler.from_pretrained(pretrained_model_name_or_path,subfolder='scheduler')
vae = AutoencoderKL.from_pretrained(
    pretrained_model_name_or_path,
    subfolder='vae'
)

denoising_unet = UNet2DConditionModel.from_pretrained(
    pretrained_model_name_or_path,subfolder="unet",
    in_channels=in_channels_denoising_unet,
    low_cpu_mem_usage=False,
    ignore_mismatched_sizes=True
)
        
reference_unet = RefUNet2DConditionModel.from_pretrained(
    pretrained_model_name_or_path,subfolder="unet",
    in_channels=in_channels_reference_unet,
    low_cpu_mem_usage=False,
    ignore_mismatched_sizes=True
)
refnet_tokenizer = CLIPTokenizer.from_pretrained(refnet_clip_vision_encoder_path)
refnet_text_encoder = CLIPTextModel.from_pretrained(refnet_clip_vision_encoder_path)
refnet_image_enc = CLIPVisionModelWithProjection.from_pretrained(refnet_clip_vision_encoder_path)
       
controlnet = ControlNetModel.from_pretrained(
    controlnet_model_name_or_path,
    in_channels=in_channels_controlnet,
    low_cpu_mem_usage=False,
    ignore_mismatched_sizes=True
)
controlnet_tokenizer = CLIPTokenizer.from_pretrained(controlnet_clip_vision_encoder_path)
controlnet_text_encoder = CLIPTextModel.from_pretrained(controlnet_clip_vision_encoder_path)
controlnet_image_enc = CLIPVisionModelWithProjection.from_pretrained(controlnet_clip_vision_encoder_path)
      

point_net=PointNet()
reference_control_writer = ReferenceAttentionControl(
            reference_unet,
            do_classifier_free_guidance=False,
            mode="write",
            fusion_blocks="full",
        )
reference_control_reader = ReferenceAttentionControl(
    denoising_unet,
    do_classifier_free_guidance=False,
    mode="read",
    fusion_blocks="full",
)



controlnet.load_state_dict(
        torch.load(val_configs.model_path.manga_control_model_path, map_location="cpu"),
        strict=False,
        )
point_net.load_state_dict(
        torch.load(val_configs.model_path.point_net_path, map_location="cpu"),
        strict=False,
        )
reference_unet.load_state_dict(
        torch.load(val_configs.model_path.manga_reference_model_path, map_location="cpu"),
        strict=False,
        )
denoising_unet.load_state_dict(
        torch.load(val_configs.model_path.manga_main_model_path, map_location="cpu"),
        strict=False,
        )
pipe = MangaNinjiaPipeline(
        reference_unet=reference_unet,
        controlnet=controlnet,
        denoising_unet=denoising_unet,  
        vae=vae,
        refnet_tokenizer=refnet_tokenizer,
        refnet_text_encoder=refnet_text_encoder,
        refnet_image_encoder=refnet_image_enc,
        controlnet_tokenizer=controlnet_tokenizer,
        controlnet_text_encoder=controlnet_text_encoder,
        controlnet_image_encoder=controlnet_image_enc,
        scheduler=noise_scheduler,
        point_net=point_net
    )
pipe = pipe.to(torch.device(device))
def string_to_np_array(coord_string):
    coord_string = coord_string.strip('[]')
    coords = re.findall(r'\d+', coord_string)
    coords = list(map(int, coords))
    coord_array = np.array(coords).reshape(-1, 2)
    return coord_array
def infer_single(is_lineart, ref_image, target_image, output_coords_ref, output_coords_base, seed = -1, num_inference_steps=20, guidance_scale_ref = 9, guidance_scale_point =15 ):
    """
    mask: 0/1 1-channel  np.array
    image: rgb           np.array
    """
    generator = torch.cuda.manual_seed(seed)
    matrix1 = np.zeros((512, 512), dtype=np.uint8)
    matrix2 = np.zeros((512, 512), dtype=np.uint8)
    output_coords_ref = string_to_np_array(output_coords_ref)
    output_coords_base = string_to_np_array(output_coords_base)
    for index, (coords_ref,coords_base) in enumerate(zip(output_coords_ref,output_coords_base)):
        y1, x1 = coords_ref 
        y2, x2 = coords_base
        matrix1[y1, x1] = index + 1
        matrix2[y2, x2] = index + 1
    point_ref = torch.from_numpy(matrix1).unsqueeze(0).unsqueeze(0)
    point_main = torch.from_numpy(matrix2).unsqueeze(0).unsqueeze(0)
    preprocessor.to(device,dtype=torch.float32) 
    pipe_out = pipe(
        is_lineart,
        ref_image,
        target_image,
        target_image,
        denosing_steps=num_inference_steps,
        processing_res=512,
        match_input_res=True,
        batch_size=1,
        show_progress_bar=True,
        guidance_scale_ref=guidance_scale_ref,
        guidance_scale_point=guidance_scale_point,
        preprocessor=preprocessor,
        generator=generator,
        point_ref=point_ref,  
        point_main=point_main,  
    )
    return pipe_out


def inference_single_image(ref_image, 
                           tar_image, 
                           ddim_steps, 
                           scale_ref,
                           scale_point, 
                           seed,
                           output_coords1,
                           output_coords2,
                           is_lineart
                           ):
    if seed == -1:
        seed = np.random.randint(10000)
    pipe_out = infer_single(is_lineart, ref_image, tar_image, output_coords_ref=output_coords1, output_coords_base=output_coords2,seed=seed ,num_inference_steps=ddim_steps, guidance_scale_ref = scale_ref, guidance_scale_point = scale_point
                           )
    return pipe_out
clicked_points_img1 = []
clicked_points_img2 = []
current_img_idx = 0  
max_clicks = 14  
point_size = 8  
colors = [(255, 0, 0), (0, 255, 0)] 

# Process images: resizing them to 512x512
def process_image(ref, base):
    ref_resized = cv2.resize(ref, (512, 512))  # Note OpenCV resize order is (width, height)
    base_resized = cv2.resize(base, (512, 512))
    return ref_resized, base_resized

# Convert string to numpy array of coordinates
def string_to_np_array(coord_string):
    coord_string = coord_string.strip('[]')
    coords = re.findall(r'\d+', coord_string)
    coords = list(map(int, coords))
    coord_array = np.array(coords).reshape(-1, 2)
    return coord_array

# Function to handle click events
def get_select_coords(img1, img2, evt: gr.SelectData):
    global clicked_points_img1, clicked_points_img2, current_img_idx
    click_coords = (evt.index[1], evt.index[0])

    if current_img_idx == 0:
        clicked_points_img1.append(click_coords)
        if len(clicked_points_img1) > max_clicks:
            clicked_points_img1 = []
        current_img = img1
        clicked_points = clicked_points_img1
    else:
        clicked_points_img2.append(click_coords)
        if len(clicked_points_img2) > max_clicks:
            clicked_points_img2 = []
        current_img = img2
        clicked_points = clicked_points_img2

    current_img_idx = 1 - current_img_idx
    img_pil = Image.fromarray(current_img.astype('uint8'))
    draw = ImageDraw.Draw(img_pil)
    for idx, point in enumerate(clicked_points):
        x, y = point
        color = colors[current_img_idx]
        for dx in range(-point_size, point_size + 1):
            for dy in range(-point_size, point_size + 1):
                if 0 <= y + dy < img_pil.size[0] and 0 <= x + dx < img_pil.size[1]:
                    draw.point((y+dy, x+dx), fill=color)

    img_out = np.array(img_pil)
    coord_array = np.array([(x, y) for x, y in clicked_points])
    return img_out, coord_array

# Function to clear the clicked points
def undo_last_point(ref, base):
    global clicked_points_img1, clicked_points_img2, current_img_idx
    current_img_idx=1-current_img_idx
    if current_img_idx == 0 and clicked_points_img1:
        clicked_points_img1.pop()  # Undo last point in ref
    elif current_img_idx == 1 and clicked_points_img2:
        clicked_points_img2.pop()  # Undo last point in base

    # After removing the last point, redraw the image without it
    if current_img_idx == 0:
        current_img = ref
        current_img_other = base
        clicked_points = clicked_points_img1
        clicked_points_other = clicked_points_img2
    else:
        current_img = base
        current_img_other = ref
        clicked_points = clicked_points_img2
        clicked_points_other = clicked_points_img1

    # Redraw the image without the last point
    img_pil = Image.fromarray(current_img.astype('uint8'))
    draw = ImageDraw.Draw(img_pil)
    for idx, point in enumerate(clicked_points):
        x, y = point
        color = colors[current_img_idx]
        for dx in range(-point_size, point_size + 1):
            for dy in range(-point_size, point_size + 1):
                if 0 <= y + dy < img_pil.size[0] and 0 <= x + dx < img_pil.size[1]:
                    draw.point((y+dy, x+dx), fill=color)
    img_out = np.array(img_pil)


    img_pil_other = Image.fromarray(current_img_other.astype('uint8'),)
    draw_other = ImageDraw.Draw(img_pil_other)
    for idx, point in enumerate(clicked_points_other):
        x, y = point
        color = colors[1-current_img_idx]
        for dx in range(-point_size, point_size + 1):
            for dy in range(-point_size, point_size + 1):
                if 0 <= y + dy < img_pil.size[0] and 0 <= x + dx < img_pil.size[1]:
                    draw_other.point((y+dy, x+dx), fill=color)
    img_out_other = np.array(img_pil_other)

    coord_array = np.array([(x, y) for x, y in clicked_points])
    # Return the updated image and coordinates as text
    updated_coords = str(coord_array.tolist())
    
    # If current_img_idx is 0, it means we are working with ref, so return for ref
    if current_img_idx == 0:
        coord_array2 = np.array([(x, y) for x, y in clicked_points_img2])
        updated_coords2 = str(coord_array2.tolist())
        return img_out, updated_coords, img_out_other, updated_coords2  # for ref image
    else:
        coord_array1 = np.array([(x, y) for x, y in clicked_points_img1])
        updated_coords1 = str(coord_array1.tolist())
        return img_out_other, updated_coords1, img_out, updated_coords  # for base image


# Main function to run the image processing
def run_local(ref, base, *args):
    image = Image.fromarray(base)
    ref_image = Image.fromarray(ref)
    
    pipe_out = inference_single_image(ref_image.copy(), image.copy(), *args)
    to_save_dict = pipe_out.to_save_dict
    to_save_dict['edit2'] = pipe_out.img_pil
    return [to_save_dict['edit2'], to_save_dict['edge2_black']]

with gr.Blocks() as demo:
    with gr.Column():
        gr.Markdown("#  MangaNinja: Line Art Colorization with Precise Reference Following")
        
        with gr.Row():
            baseline_gallery = gr.Gallery(label='Output', show_label=True, elem_id="gallery", columns=1, height=768)
            
            with gr.Accordion("Advanced Option", open=True):
                num_samples = 1
                ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=50, step=1)
                scale_ref = gr.Slider(label="Guidance of ref", minimum=0, maximum=30.0, value=9, step=0.1)
                scale_point = gr.Slider(label="Guidance of points", minimum=0, maximum=30.0, value=15, step=0.1)
                is_lineart = gr.Checkbox(label="Input is lineart", value=False)
                seed = gr.Slider(label="Seed", minimum=-1, maximum=999999999, step=1, value=-1)
                
                gr.Markdown("### Tutorial")
                gr.Markdown("1. Upload the reference image and target image. Note that for the target image, there are two modes: you can upload an RGB image, and the model will automatically extract the line art; or you can directly upload the line art by checking the 'input is lineart' option.")
                gr.Markdown("2. Click 'Process Images' to resize the images to 512*512 resolution.")
                gr.Markdown("3. (Optional) **Starting from the reference image**, **alternately** click on the reference and target images in sequence to define matching points. Use 'Undo' to revert the last action.")
                gr.Markdown("4. Click 'Generate' to produce the result.")
        gr.Markdown("# Upload the reference image and target image")
        
        with gr.Row():
            ref = gr.Image(label="Reference Image",)
            base = gr.Image(label="Target Image",)
        gr.Button("Process Images").click(process_image, inputs=[ref, base], outputs=[ref, base])
            
        with gr.Row():
            output_img1 = gr.Image(label="Reference Output")
            output_coords1 = gr.Textbox(lines=2, label="Clicked Coordinates Image 1 (npy format)")
            output_img2 = gr.Image(label="Base Output")
            output_coords2 = gr.Textbox(lines=2, label="Clicked Coordinates Image 2 (npy format)")

        # Image click select functions
        ref.select(get_select_coords, [ref, base], [output_img1, output_coords1])
        base.select(get_select_coords, [ref, base], [output_img2, output_coords2])
        
        # Undo button
        undo_button = gr.Button("Undo")
        undo_button.click(undo_last_point, inputs=[ref, base], outputs=[output_img1, output_coords1, output_img2, output_coords2])

        run_local_button = gr.Button(label="Generate", value="Generate")
        
    with gr.Row():
        gr.Examples(
            examples=[
                ['test_cases/hz0.png', 'test_cases/hz1.png'],
                ['test_cases/more_cases/az0.png', 'test_cases/more_cases/az1.JPG'],
                ['test_cases/more_cases/hi0.png', 'test_cases/more_cases/hi1.jpg'],
                ['test_cases/more_cases/kn0.jpg', 'test_cases/more_cases/kn1.jpg'],
                ['test_cases/more_cases/rk0.jpg', 'test_cases/more_cases/rk1.jpg'],
            
            
            ],
            inputs=[ref, base],
            cache_examples=False,
            examples_per_page=100
        )
        

    run_local_button.click(fn=run_local, 
                           inputs=[ref, 
                                   base, 
                                   ddim_steps, 
                                   scale_ref,
                                   scale_point, 
                                   seed,
                                   output_coords1,
                                   output_coords2,
                                   is_lineart
                                  ], 
                           outputs=[baseline_gallery]
    )

demo.launch(server_name="0.0.0.0")
