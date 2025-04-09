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
from aiohttp import web
from aiohttp import web
from server import PromptServer

import uuid
from .node_utils import  load_images,equalize_lists,tensor2pil_list
import folder_paths
from .infer import infer_main,nijia_loader

server_instance = PromptServer.instance

MAX_SEED = np.iinfo(np.int32).max
current_node_path = os.path.dirname(os.path.abspath(__file__))
device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

node_instances = {}
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
                "xy_data_ref": ("MINJIA_DATA",),
                "xy_data_lineart": ("MINJIA_DATA",),
            }

        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "lineart")
    FUNCTION = "sampler_main"
    CATEGORY = "MangaNinjia"
    
    def sampler_main(self, model,image,lineart_image,seed,width,height,guidance_scale_ref,guidance_scale_point,steps,is_lineart,**kwargs):
        
        xy_data_ref=kwargs.get("xy_data_ref")
        xy_data_lineart=kwargs.get("xy_data_lineart")
       
        print(xy_data_ref,xy_data_lineart)
        ref_image_list=tensor2pil_list(image,width,height)
        lineart_image_list=tensor2pil_list(lineart_image,width,height)
        ref_image_list,lineart_image_list=equalize_lists(ref_image_list,lineart_image_list)
        ref_value,lineart_value=None,None
        if xy_data_ref is not None and xy_data_lineart is not None:
            lineart_value=xy_data_lineart
            ref_value=xy_data_ref
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

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "markimage_image": ("STRING", {"default": "canvas_image.png"}),
                "trigger": ("INT", {"default": 0, "min": 0, "max": 99999999, "step": 1, "hidden": True}),
            },
        }
    
    RETURN_TYPES = ("IMAGE", "MINJIA_DATA")
    RETURN_NAMES = ("image", "xy_data")
    FUNCTION = "process_canvas_image"
    CATEGORY = "MangaNinjia"

    def process_canvas_image(self, markimage_image,trigger):
        
        try:
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
                node_id = os.path.basename(path_image)
            except Exception as e:
                # 如果读取失败，创建白色画布
                processed_image = torch.ones((1, 512, 512, 3), dtype=torch.float32)
                node_id=''

            # 使用 clickedPoints 数据
            clicked_points = node_instances.get(node_id)
            #print(f"Using clickedPoints in process_canvas_image: {clicked_points}")
         
        
            # 示例：将 clickedPoints 数据与画布图像结合处理
            if clicked_points:
                return (processed_image,process_json(clicked_points),)
            else:
                return (processed_image, None,)  
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

def process_json(json_data):
    if not json_data:
        return None

    # 类型检查和数据验证
    if not isinstance(json_data, list):
        print("Error: Expected a list of dictionaries.")
        return None

    for item in json_data:
        if not isinstance(item, dict) or 'number' not in item:
            print(f"Skipping invalid item: {item}")
            continue

    try:
        # 对数据进行排序
        sorted_data = sorted(
            [item for item in json_data if isinstance(item, dict) and 'number' in item],
            key=lambda item: item['number']
        )

        # 转换为 [[x的int值, y的int值], ...] 格式
        result = [[int(item['x']), int(item['y'])] for item in sorted_data if 'x' in item and 'y' in item]
        print(f"Processed JSON data: {result}")
        return result
    except Exception as e:
        print(f"Error processing JSON data: {str(e)}")
        return None
    


# 动态添加路由到服务器
@PromptServer.instance.routes.post("/upload/clickedPoints")
async def upload_clicked_points(request):

    #print("Custom route '/api//upload/clickedPoints' added successfully.")
    # 打印当前路由
    # for route in server_instance.app.router.routes():
    #     print(f"Route: {route.method} {route.resource}")
   
    try:
        # 获取 JSON 数据
        json_data = await request.json()
        clicked_points = json_data.get("clickedPoints", [])
        node_id = json_data.get("node_id")  # 前端需要传递节点的唯一标识
        #print(f"Received clickedPoints for node {node_id}: {clicked_points}")

        # 根据 node_id 存储数据
        if node_id:
            node_instances[node_id] = clicked_points
        else:
            print(f"Node with ID {node_id} not found.")
        
            
        
        return web.json_response({"status": "success", "message": "Clicked points received."})
    except Exception as e:
        print(f"Error processing clickedPoints: {e}")
        return web.json_response({"status": "error", "message": str(e)}, status=500)


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
