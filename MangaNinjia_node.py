# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import torch
import gc
import numpy as np
from PIL import Image, ImageOps
import hashlib
import traceback
import time
import json


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
                "clip": (["none"] + folder_paths.get_filename_list("clip"),),
                "controlnet": (["none"] + folder_paths.get_filename_list("controlnet"),),
            },
        }

    RETURN_TYPES = ("MODEL_MangaNinjia",)
    RETURN_NAMES = ("model",)
    FUNCTION = "loader_main"
    CATEGORY = "MangaNinjia"

    def loader_main(self,checkpoint,clip,controlnet):
        original_config_file=os.path.join(current_node_path,"configs","v1-inference.yaml")
        sd_config=os.path.join(current_node_path,"sd15_repo")
        if checkpoint!="none":
            ckpt_path=folder_paths.get_full_path("checkpoints",checkpoint)
        else:
            raise "no checkpoint"
        
        if controlnet!="none":
            controlnet_model_name_or_path=folder_paths.get_full_path("controlnet",controlnet)
        else:
            raise "no checkpoint"
        
        if clip!="none": 
            clip_path=folder_paths.get_full_path("clip",clip)
        else:   
            raise "no clip"
        print("***********Start MangaNinjia Loader**************")
        pipe,preprocessor,refnet_tokenizer,refnet_text_encoder,refnet_image_encoder,vae=nijia_loader(MangaNinjia_weigths_path,
        sd_config,controlnet_model_name_or_path,clip_path,device,ckpt_path,original_config_file,sd_config)
        print("***********MangaNinjia Loader is Done**************")
        model={"pipe":pipe,"preprocessor":preprocessor,"refnet_tokenizer":refnet_tokenizer,"refnet_text_encoder":refnet_text_encoder,"refnet_image_encoder":refnet_image_encoder,"vae":vae}
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
            "optional": {
                #"mask": ("MASK",),# B H W 
                "xy_data": ("MINJIA_DATA",),
            }

        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "lineart")
    FUNCTION = "sampler_main"
    CATEGORY = "MangaNinjia"
    
    def sampler_main(self, model,image,lineart_image,seed,width,height,guidance_scale_ref,guidance_scale_point,steps,is_lineart,**kwargs):
        
        xy_data=kwargs.get("xy_data")
        #lineart_mask=kwargs.get("lineart_mask")

        ref_image_list=tensor2pil_list(image,width,height)
        lineart_image_list=tensor2pil_list(lineart_image,width,height)
        ref_image_list,lineart_image_list=equalize_lists(ref_image_list,lineart_image_list)
        ref_value,lineart_value=None,None
        if isinstance(xy_data,dict) :
            if xy_data.get("lineart_value") is not None  and xy_data.get("ref_value") is not None :
                lineart_value=xy_data.get("lineart_value")
                ref_value=xy_data.get("ref_value")
                if len (lineart_value)!=len (ref_value):
                    min_length = min(len(lineart_value), len(ref_value))
                    lineart_value = lineart_value[:min_length]
                    ref_value = ref_value[:min_length]

        
            

        print("***********Start MangaNinjia Sampler**************")
        iamge,lineart=infer_main(model,ref_image_list,lineart_image_list,ref_value,lineart_value,steps,seed,is_lineart,guidance_scale_ref,guidance_scale_point,device)
        
        
        gc.collect()
        torch.cuda.empty_cache()
        return (load_images(iamge),load_images(lineart),)

class MarkImageNode:
    _canvas_cache = {
        'image': None,
        'mask': None,
        'cache_enabled': True,
        'data_flow_status': {},
        'persistent_cache': {},
        'last_execution_id': None
    }
    def __init__(self):
        pass
   
    def get_execution_id(self):
        """获取当前工作流执行ID"""
        try:
            # 可以使用时间戳或其他唯一标识
            return str(int(time.time() * 1000))
        except Exception as e:
            print(f"Error getting execution ID: {str(e)}")
            return None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "markimage_image": ("STRING", {"default": "canvas_image.png"}),
                "ref_coordinates": ("STRING", {"default": "canvas_image.json"}),
                "lineart_coordinates": ("STRING", {"default": "canvas_image.json"}),
                "trigger": ("INT", {"default": 0, "min": 0, "max": 99999999, "step": 1, "hidden": True}),
            },
        }
    
    RETURN_TYPES = ("IMAGE", "MINJIA_DATA")
    RETURN_NAMES = ("image", "xy_data")
    FUNCTION = "process_canvas_image"
    CATEGORY = "MangaNinjia"

    def process_canvas_image(self, markimage_image,ref_coordinates,lineart_coordinates,trigger):

        try:
            current_execution = self.get_execution_id()
            print(f"Processing canvas image, execution ID: {current_execution}")
            try:
                # 尝试读取画布图像
                path_image = folder_paths.get_annotated_filepath(markimage_image)
                i = Image.open(path_image)
               
                i = ImageOps.exif_transpose(i)
                if i.mode not in ['RGB', 'RGBA']:
                    i = i.convert('RGB')
                image = np.array(i).astype(np.float32) / 255.0
                if i.mode == 'RGBA':
                    rgb = image[..., :3]
                    alpha = image[..., 3:]
                    image = rgb * alpha + (1 - alpha) * 0.5
                processed_image = torch.from_numpy(image)[None,]
              
            except Exception as e:
                # 如果读取失败，创建白色画布
                processed_image = torch.ones((1, 512, 512, 3), dtype=torch.float32)


            return (processed_image,{"lineart_value":process_json(lineart_coordinates),"ref_value":process_json(ref_coordinates)})
                
        except Exception as e:
            print(f"Error in process_canvas_image: {str(e)}")
            traceback.print_exc()
            return ()
    @classmethod
    def IS_CHANGED(s, markimage_image,**kwargs):
        image_path = folder_paths.get_annotated_filepath(markimage_image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

def process_json(file_path,):
        if file_path:
            if os.path.exists(file_path) and file_path.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)

                # 按 number 字段排序
                sorted_data = sorted(data, key=lambda item: item['number'])

                # 转换为 [[x的int值, y的int值], ...] 格式
                result = [[int(item['x']), int(item['y'])] for item in sorted_data]
                return result
            else:
                return None
        else:
            return None

WEB_DIRECTORY = "./js"

NODE_CLASS_MAPPINGS = {
    "MangaNinjiaLoader":MangaNinjiaLoader,
    "MangaNinjiaSampler":MangaNinjiaSampler,
    "MarkImageNode":MarkImageNode,
    }

NODE_DISPLAY_NAME_MAPPINGS = {
    "MangaNinjiaLoader":"MangaNinjiaLoader",
    "MangaNinjiaSampler":"MangaNinjiaSampler",
    "MarkImageNode":"MarkImageNode",
}
