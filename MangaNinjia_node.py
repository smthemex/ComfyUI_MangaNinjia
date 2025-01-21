# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import torch
import gc
import numpy as np
from .node_utils import  load_images,equalize_lists,tensor2pil_list
import folder_paths
from .infer import infer_main,nijia_loader

MAX_SEED = np.iinfo(np.int32).max
current_node_path = os.path.dirname(os.path.abspath(__file__))
device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

# add checkpoints dir
MangaNinjia_weigths_path = os.path.join(folder_paths.models_dir, "MangaNinjia")
if not os.path.exists(MangaNinjia_weigths_path):
    os.makedirs(MangaNinjia_weigths_path)
folder_paths.add_model_folder_path("MangaNinjia", MangaNinjia_weigths_path)


class MangaNinjiaLoader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "checkpoint": (["none"] + folder_paths.get_filename_list("checkpoints"),),
                "clip": ("STRING", {"default": "openai/clip-vit-large-patch14"}),
                "controlnet": (["none"] + folder_paths.get_filename_list("controlnet"),),
            },
        }

    RETURN_TYPES = ("MODEL_MangaNinjia",)
    RETURN_NAMES = ("model",)
    FUNCTION = "loader_main"
    CATEGORY = "MangaNinjia"

    def loader_main(self,checkpoint,clip,controlnet):
        original_config_file=os.path.join(folder_paths.models_dir,"configs","v1-inference.yaml")
        sd_config=os.path.join(current_node_path,"sd15_repo")
        if checkpoint!="none":
            ckpt_path=folder_paths.get_full_path("checkpoints",checkpoint)
        else:
            raise "no checkpoint"
        
        if controlnet!="none":
            controlnet_model_name_or_path=folder_paths.get_full_path("controlnet",controlnet)
        else:
            raise "no checkpoint"
        if not clip:    
            raise "no clip"
        print("***********Start MangaNinjia Loader**************")
        pipe,preprocessor=nijia_loader(MangaNinjia_weigths_path,sd_config,controlnet_model_name_or_path,clip,device,ckpt_path,original_config_file,sd_config)
        print("***********MangaNinjia Loader is Done**************")
        model={"pipe":pipe,"preprocessor":preprocessor}
        gc.collect()
        torch.cuda.empty_cache()
        return (model,)
    
class MangaNinjiaSampler:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL_MangaNinjia",),
                "image": ("IMAGE",),
                "lineart_image": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED}),
                "width": ("INT", {
                    "default": 512,
                    "min": 128,  # Minimum value
                    "max": 2048,  # Maximum value
                    "step": 64,  # Slider's step
                    "display": "number",  # Cosmetic only: display as "number" or "slider"
                }),
                "height": ("INT", {
                    "default": 512,
                    "min": 128,  # Minimum value
                    "max": 2048,  # Maximum value
                    "step": 64,  # Slider's step
                    "display": "number",  # Cosmetic only: display as "number" or "slider"
                }),
                "guidance_scale_ref": ("FLOAT", {
                    "default": 9.0,
                    "min": 0.0,
                    "max": 30.0,
                    "step": 0.1,
                    "display": "number",
                }),
                "guidance_scale_point": ("FLOAT", {
                    "default": 15.0,
                    "min": 0.0,
                    "max": 30.0,
                    "step": 0.1,
                    "display": "number",
                }),
                "steps": ("INT", {
                    "default": 25,
                    "min": 1,  # Minimum value
                    "max": 1024,  # Maximum value
                    "step": 1,  # Slider's step
                    "display": "number",  # Cosmetic only: display as "number" or "slider"
                }),
                "is_lineart": ("BOOLEAN", {"default": True},),
                         },
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "lineart")
    FUNCTION = "sampler_main"
    CATEGORY = "MangaNinjia"
    
    def sampler_main(self, model,image,lineart_image,seed,width,height,guidance_scale_ref,guidance_scale_point,steps,is_lineart):
        
        #pre data
        pipe=model.get("pipe")
        pipe.to(device=device)
        preprocessor=model.get("preprocessor")
       
        ref_image_list=tensor2pil_list(image,width,height)
        lineart_image_list=tensor2pil_list(lineart_image,width,height)
        ref_image_list,lineart_image_list=equalize_lists(ref_image_list,lineart_image_list)

        point_ref_paths,point_lineart_paths=None,None
        print("***********Start MangaNinjia Sampler**************")
        iamge,lineart=infer_main(pipe,preprocessor,ref_image_list,lineart_image_list,point_ref_paths,point_lineart_paths,steps,seed,is_lineart,guidance_scale_ref,guidance_scale_point)
        
        pipe.to("cpu")# move pipe to cpu 
        gc.collect()
        torch.cuda.empty_cache()
        return (load_images(iamge),load_images(lineart),)



NODE_CLASS_MAPPINGS = {
    "MangaNinjiaLoader":MangaNinjiaLoader,
    "MangaNinjiaSampler":MangaNinjiaSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MangaNinjiaLoader":"MangaNinjiaLoader",
    "MangaNinjiaSampler":"MangaNinjiaSampler",
}
