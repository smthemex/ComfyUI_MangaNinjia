import logging
import os
import random
import numpy as np
import torch
import gc
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
import torch.nn as nn
from .src.manganinjia_pipeline import MangaNinjiaPipeline
from .src.image_util import resize_max_res,chw2hwc
from diffusers import (
    ControlNetModel,
    StableDiffusionPipeline,
    DDIMScheduler,
    AutoencoderKL,
)
from .src.models.mutual_self_attention_multi_scale import ReferenceAttentionControl
from .src.models.unet_2d_condition import UNet2DConditionModel
from .src.models.refunet_2d_condition import RefUNet2DConditionModel
from .src.point_network import PointNet
from .src.annotator.lineart import BatchLineartDetector
import folder_paths

# if "__main__" == __name__:
#     logging.basicConfig(level=logging.INFO)

#     # -------------------- Arguments --------------------
#     parser = argparse.ArgumentParser(
#         description="Run single-image MangaNinjia"
#     )
#     parser.add_argument(
#         "--output_dir", type=str, required=True, help="Output directory."
#     )

#     # inference setting
#     parser.add_argument(
#         "--denoise_steps",
#         type=int,
#         default=50,  # quantitative evaluation uses 50 steps
#         help="Diffusion denoising steps, more steps results in higher accuracy but slower inference speed.",
#     )

#     # resolution setting
#     parser.add_argument("--seed", type=int, default=None, help="Random seed.")

#     parser.add_argument(
#         "--repo",
#         type=str,
#         default=None,
#         required=True,
#         help="Path to pretrained model or model identifier from huggingface.co/models.",
#     )
#     parser.add_argument(
#         "--image_encoder_path",
#         type=str,
#         default=None,
#         required=True,
#         help="Path to pretrained model or model identifier from huggingface.co/models.",
#     )
#     parser.add_argument(
#         "--controlnet_model_name_or_path", type=str, required=True, help="Path to original controlnet."
#     )
#     parser.add_argument(
#         "--annotator_ckpts_path", type=str, required=True, help="Path to depth inpainting model."
#     )
#     parser.add_argument(
#         "--manga_reference_unet_path", type=str, required=True, help="Path to depth inpainting model."
#     )
#     parser.add_argument(
#         "--manga_main_model_path", type=str, required=True, help="Path to depth inpainting model."
#     )
#     parser.add_argument(
#         "--manga_controlnet_model_path", type=str, required=True, help="Path to depth inpainting model."
#     )
#     parser.add_argument(
#         "--point_net_path", type=str, required=True, help="Path to depth inpainting model."
#     )
#     parser.add_argument(
#         "--input_reference_paths",
#         nargs='+',
#         default=None,
#         help="input_image_paths",
#     )
#     parser.add_argument(
#         "--input_lineart_paths",
#         nargs='+',
#         default=None,
#         help="lineart_paths",
#     )
#     parser.add_argument(
#         "--point_ref_paths",
#         type=str,
#         default=None,
#         nargs="+",
#     )
#     parser.add_argument(
#         "--point_lineart_paths",
#         type=str,
#         default=None,
#         nargs="+",
#     )
#     parser.add_argument(
#         "--is_lineart",
#         action="store_true",
#         default=False
#     )
#     parser.add_argument(
#         "--guidance_scale_ref",
#         type=float,
#         default=1e-4,
#         help="guidance scale for reference image",
#     )
#     parser.add_argument(
#         "--guidance_scale_point",
#         type=float,
#         default=1e-4,
#         help="guidance scale for points",
#     )

def nijia_loader(MangaNinjia_weigths_path,repo,controlnet_model_name_or_path,image_encoder_path,device,
                 ckpt_path,original_config_file,sd_config):
    # -------------------- Model --------------------
    preprocessor = BatchLineartDetector(MangaNinjia_weigths_path)# 直接使用模型路径，cn的预处理器
    preprocessor.to(device,dtype=torch.float32) 
    in_channels_reference_unet = 4
    in_channels_denoising_unet = 4
    in_channels_controlnet = 4

    try:
        pipe = StableDiffusionPipeline.from_single_file(
            ckpt_path,config=sd_config, original_config=original_config_file)
    except:
        pipe = StableDiffusionPipeline.from_single_file(
            ckpt_path, config=sd_config,original_config_file=original_config_file)


    noise_scheduler = DDIMScheduler.from_pretrained(repo,subfolder='scheduler')
    vae=pipe.vae


    # vae = AutoencoderKL.from_pretrained(
    #     repo,
    #     subfolder='vae'
    # )


    # denoising_unet = UNet2DConditionModel.from_pretrained(
    #     repo,subfolder="unet",
    #     in_channels=in_channels_denoising_unet,
    #     low_cpu_mem_usage=False,
    #     ignore_mismatched_sizes=True
    # )
            
    # reference_unet = RefUNet2DConditionModel.from_pretrained(
    #     repo,subfolder="unet",
    #     in_channels=in_channels_reference_unet,
    #     low_cpu_mem_usage=False,
    #     ignore_mismatched_sizes=True
    # )

    Unet=pipe.unet

    denoising_unet = UNet2DConditionModel.from_config(
        repo,subfolder="unet",
        in_channels=in_channels_denoising_unet,
        low_cpu_mem_usage=False,
        ignore_mismatched_sizes=True
    )
    
    denoising_unet.load_state_dict(
        Unet.state_dict(),strict=False,)


    reference_unet = RefUNet2DConditionModel.from_config(
        repo,subfolder="unet",
        in_channels=in_channels_reference_unet,
        low_cpu_mem_usage=False,
        ignore_mismatched_sizes=True
    )
    
    reference_unet.load_state_dict(
        Unet.state_dict(),strict=False,)

    refnet_tokenizer = CLIPTokenizer.from_pretrained(image_encoder_path)
    refnet_text_encoder = CLIPTextModel.from_pretrained(image_encoder_path)
    refnet_image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path)
    print("load controlnet")
    controlnet=ControlNetModel.from_unet(Unet)

    if os.path.splitext(controlnet_model_name_or_path)[1].lower() == ".safetensors":
        import safetensors.torch
        cn_dict=safetensors.torch.load_file(controlnet_model_name_or_path)
    else:
        cn_dict=torch.load(controlnet_model_name_or_path)
    controlnet.load_state_dict(cn_dict, strict=False)
    # controlnet = ControlNetModel.from_pretrained(
    #     controlnet_model_name_or_path,
    #     in_channels=in_channels_controlnet,
    #     low_cpu_mem_usage=False,
    #     ignore_mismatched_sizes=True
    # )
    del cn_dict,Unet,pipe
    gc.collect()
    torch.cuda.empty_cache()

    # controlnet_tokenizer = CLIPTokenizer.from_pretrained(image_encoder_path)
    # controlnet_text_encoder = CLIPTextModel.from_pretrained(image_encoder_path)
    # controlnet_image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path)
        
    point_net=PointNet()

    cn_dict_= torch.load(os.path.join(MangaNinjia_weigths_path,"controlnet.pth"), map_location="cpu")
    controlnet.load_state_dict(cn_dict_,strict=False,)

    point_net_= torch.load(os.path.join(MangaNinjia_weigths_path,"point_net.pth"), map_location="cpu")
    point_net.load_state_dict(point_net_,strict=False,)

    refer_dict=torch.load(os.path.join(MangaNinjia_weigths_path,"reference_unet.pth"), map_location="cpu")
    reference_unet.load_state_dict(refer_dict,strict=False,)

    deno_dict=torch.load(os.path.join(MangaNinjia_weigths_path,"denoising_unet.pth"), map_location="cpu")
    denoising_unet.load_state_dict(deno_dict,strict=False,
                                   )
    del cn_dict_,point_net_,deno_dict,refer_dict
    gc.collect()
    torch.cuda.empty_cache()

    pipe = MangaNinjiaPipeline(
            reference_unet=reference_unet,
            controlnet=controlnet,
            denoising_unet=denoising_unet,  
            #ae=vae,
            # refnet_tokenizer=refnet_tokenizer,
            # refnet_text_encoder=refnet_text_encoder,
            # refnet_image_encoder=refnet_image_encoder,
            # controlnet_tokenizer=controlnet_tokenizer,
            # controlnet_text_encoder=controlnet_text_encoder,
            # controlnet_image_encoder=controlnet_image_encoder,
            scheduler=noise_scheduler,
            point_net=point_net
        )
    
    #pipe = pipe.to(torch.device(device))
    pipe.enable_xformers_memory_efficient_attention()
    return pipe,preprocessor,refnet_tokenizer,refnet_text_encoder,refnet_image_encoder,vae

def infer_main (model,ref_image_list,lineart_image_list,point_ref_paths,point_lineart_paths,denoise_steps,seed,is_lineart,guidance_scale_ref,guidance_scale_point,device):
     #pre data
    pipe=model.get("pipe")
    
    preprocessor=model.get("preprocessor")
    refnet_image_encoder=model.get("refnet_image_encoder")
    refnet_text_encoder=model.get("refnet_text_encoder")
    refnet_tokenizer=model.get("refnet_tokenizer")

    refnet_image_encoder=refnet_image_encoder.to(device)
    refnet_text_encoder=refnet_text_encoder.to(device)

    HALF=True
    dtype = torch.float16 if  HALF else torch.float32
    controlnet_uncond_encoder_hidden_states = prompt2embeds("", refnet_tokenizer, refnet_text_encoder,device,dtype)
    refnet_uncond_encoder_hidden_states= prompt2embeds("", refnet_tokenizer, refnet_text_encoder,device,dtype)

    controlnet_encoder_hidden_states=img2embeds(ref_image_list[0], refnet_image_encoder,device) #TO DO
    refnet_encoder_hidden_states=img2embeds(ref_image_list[0], refnet_image_encoder,device) # TO DO more image

    refnet_image_encoder.to("cpu")
    refnet_text_encoder.to("cpu")
    vae=model.get("vae")
    vae.to(device)
    processing_res=512
    rgb_latent_scale_factor = 0.18215
    
    gc.collect()
    torch.cuda.empty_cache()

    output_dir = folder_paths.get_output_directory()

    if seed is None:
        import time

        seed = int(time.time())
    generator = torch.cuda.manual_seed(seed)

    ref1_latents=encode_RGB(vae,ref_image_list[0],processing_res, generator,device,rgb_latent_scale_factor,torch.float32)

    # -------------------- Device --------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"device = {device}")
    pipe.to(device=device)
    

    # -------------------- Inference and saving --------------------
    with torch.no_grad():
        image_list=[]
        lineart_list=[]
        for i in range(len(ref_image_list)):
            # save path
           
            rgb_name_prefix  =''.join(random.choice("0123456789") for _ in range(6))
            pred_name_base = rgb_name_prefix + "_colorized"
            lineart_name_base = rgb_name_prefix + "_lineart"
            colored_save_path = os.path.join(
                output_dir, f"{pred_name_base}.png"
            )
            lineart_save_path = os.path.join(
                output_dir, f"{lineart_name_base}.png"
            )
            if point_ref_paths is not None:# 手动映射，比较讨厌
                point_ref_path = point_ref_paths[i]
                point_lineart_path = point_lineart_paths[i]
                point_ref = torch.from_numpy(np.load(point_ref_path)).unsqueeze(0).unsqueeze(0)
                point_main = torch.from_numpy(np.load(point_lineart_path)).unsqueeze(0).unsqueeze(0)
            else:
                matrix1 = np.zeros((512, 512), dtype=np.uint8)
                matrix2 = np.zeros((512, 512), dtype=np.uint8)
                point_ref = torch.from_numpy(matrix1).unsqueeze(0).unsqueeze(0)
                point_main = torch.from_numpy(matrix2).unsqueeze(0).unsqueeze(0)
     
            ref_image=ref_image_list[i]
            target_image =lineart_image_list[i]
            # ref_image.save(f"{i}_ref_image.png")
            # target_image.save(f"{i}_target_image.png")

            pipe_out = pipe(
                is_lineart,
                ref_image,
                target_image,
                target_image,
                denosing_steps=denoise_steps,
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
                controlnet_encoder_hidden_states=controlnet_encoder_hidden_states,
                controlnet_uncond_encoder_hidden_states=controlnet_uncond_encoder_hidden_states,
                refnet_encoder_hidden_states=refnet_encoder_hidden_states,
                refnet_uncond_encoder_hidden_states=refnet_uncond_encoder_hidden_states,
                ref1_latents=ref1_latents
            )

            if os.path.exists(colored_save_path):
                logging.warning(f"Existing file: '{colored_save_path}' will be overwritten")
            image = pipe_out.latent
            lineart = pipe_out.to_save_dict['edge2_black']
            image_list.append(image)
            lineart_list.append(lineart)
            #image.save(colored_save_path)
            #lineart.save(lineart_save_path)
    vae.to(device)
    out_list=[]
    for i in image_list:
        img=latent2pil(vae,i,rgb_latent_scale_factor)
        out_list.append(img)
    pipe.to("cpu") # move pipe to cpu 
    return out_list,lineart_list

from transformers import CLIPImageProcessor
clip_image_processor=CLIPImageProcessor()

def img2embeds(img, image_enc,device):

    clip_image = clip_image_processor.preprocess(
        img, return_tensors="pt"
    ).pixel_values
    clip_image_embeds = image_enc(
        clip_image.to(device, dtype=image_enc.dtype)
    ).image_embeds
    encoder_hidden_states = clip_image_embeds.unsqueeze(1)
    return encoder_hidden_states

def prompt2embeds(prompt, tokenizer, text_encoder,device,dtype):
    text_inputs = tokenizer(
        prompt,
        padding="do_not_pad",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(device) #[1,2]
    empty_text_embed = text_encoder(text_input_ids)[0].to(dtype)
    uncond_encoder_hidden_states = empty_text_embed.repeat((1, 1, 1))[:,0,:].unsqueeze(0)
    return uncond_encoder_hidden_states

def encode_RGB(vae,ref1,processing_res, generator,device,rgb_latent_scale_factor,dtype) -> torch.Tensor:
    """
    Encode RGB image into latent.

    Args:
        rgb_in (`torch.Tensor`):
            Input RGB image to be encoded.

    Returns:
        `torch.Tensor`: Image latent.
    """
    def resize_img(img):
            img = resize_max_res(img, max_edge_resolution=processing_res)
            return img
    ref1 = resize_img(ref1)
    def normalize_img(img):
        img = img.convert("RGB")
        img = np.array(img)

        # Normalize RGB Values.
        rgb = np.transpose(img,(2,0,1))
        rgb_norm = rgb / 255.0 * 2.0 - 1.0
        rgb_norm = torch.from_numpy(rgb_norm).to(dtype)
        rgb_norm = rgb_norm.to(device)
        img = rgb_norm
        assert img.min() >= -1.0 and img.max() <= 1.0
        return img
    ref1=normalize_img(ref1)
    # generator = None
    rgb_latent = vae.encode(ref1[None]).latent_dist.sample(generator)
    rgb_latent = rgb_latent * rgb_latent_scale_factor
    vae.to("cpu")
    return rgb_latent


def latent2pil(vae,rgb_latent,rgb_latent_scale_factor):
    def decode_RGB(vae, rgb_latent: torch.Tensor,rgb_latent_scale_factor) -> torch.Tensor:
        """
        Decode depth latent into depth map.

        Args:
            rgb_latent (`torch.Tensor`):
                Depth latent to be decoded.

        Returns:
            `torch.Tensor`: Decoded depth map.
        """

        rgb_latent = rgb_latent / rgb_latent_scale_factor
        rgb_out = vae.decode(rgb_latent, return_dict=False)[0]
        return rgb_out

    edit2 = decode_RGB(vae,rgb_latent,rgb_latent_scale_factor)
    img_pred = torch.clip(edit2, -1.0, 1.0)
    img_pred = img_pred.squeeze().cpu().numpy().astype(np.float32)
    img_pred_np = (((img_pred + 1.) / 2.) * 255).astype(np.uint8)
    img_pred_np = chw2hwc(img_pred_np)
    img_pred_pil = Image.fromarray(img_pred_np)
    return img_pred_pil
