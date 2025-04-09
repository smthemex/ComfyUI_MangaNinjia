# ComfyUI_MangaNinjia
ComfyUI_MangaNinjia is a ComfyUI node of [MangaNinja](https://github.com/ali-vilab/MangaNinjia) which‌ is a " Line Art Colorization with Precise Reference Following " method。

# update
* 改进参考点的前端方法，只需要在参考和线稿图上点选对应位置的点即可，唯一要注意的是，2个节点的markimage_image栏里的canvas_image.png名称必须不一样，比如有个是canvas_image.png，另一个需要改成canvas_image1.png或则其他，请使用json工作流
* To improve the front-end method of reference points, simply click on the corresponding points on the reference and line draft images. The only thing to note is that the name of the canvas_image.png in the markimage_image column of the two nodes must be different. For example, one node needs to be named canvas_image.png, while the other needs to be changed to canvas_image1. png or something else. Please use the JSON workflow



# 1. Installation

In the ./ComfyUI /custom_node directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_MangaNinjia.git
```
---

# 2. Requirements  
* no need, because it's base in sd1.5 and controlnet,Perhaps someone may be missing the library.没什么特殊的库,懒得删了
```
pip install -r requirements.txt
```

# 3. Models
* sd1.5 [address](https://modelscope.cn/models/AI-ModelScope/stable-diffusion-v1-5/files) v1-5-pruned-emaonly.safetensors #example
* controlnet lineart [address](https://huggingface.co/lllyasviel/control_v11p_sd15_lineart/tree/main)   control_v11p_sd15_lineart_fp16.safetensors  #example
* Annotators [address](https://huggingface.co/lllyasviel/Annotators/blob/main/sk_model.pth)   sk_model.pth #example
* MangaNinjia [address](https://huggingface.co/Johanan0528/MangaNinjia/tree/main)  #example
* clipvison and text   [address](https://huggingface.co/openai/clip-vit-large-patch14/tree/main) #clip-vit-large-patch14 
 
```
--  ComfyUI/models/checkpoints
    |-- any sd1.5 safetensors #任意sd1.5模型，注意要带vae的
--  ComfyUI/models/controlnet
    |-- control_v11p_sd15_lineart_fp16.safetensors or control_v11p_sd15s2_lineart_anime_fp16.safetensors
--  ComfyUI/models/clip
    |-- clip_l.safetensors #clip-vit-large-patch14 
--  ComfyUI/models/MangaNinjia
        |-- denoising_unet.pth
        |-- reference_unet.pth
        |-- point_net.pthnaz
        |-- controlnet.pth
        |-- sk_model.pth
```
# 4.Tips
* is_lineart :  if False you can link a normal image at lineart_image. 关闭is_lineart，会自动预处理图片为线稿

  
# 5.Example
* old
![](https://github.com/smthemex/ComfyUI_MangaNinjia/blob/main/example.png)
* NEW
![](https://github.com/smthemex/ComfyUI_MangaNinjia/blob/main/example_new.png)


# 6.Citation
```
@article{liu2025manganinja,
  title={MangaNinja: Line Art Colorization with Precise Reference Following},
  author={Liu, Zhiheng and Cheng, Ka Leong and Chen, Xi and Xiao, Jie and Ouyang, Hao and Zhu, Kai and Liu, Yu and Shen, Yujun and Chen, Qifeng and Luo, Ping},
  journal={arXiv preprint arXiv:2501.08332},
  year={2025}
}
```
