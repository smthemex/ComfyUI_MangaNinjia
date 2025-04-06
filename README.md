# ComfyUI_MangaNinjia
ComfyUI_MangaNinjia is a ComfyUI node of [MangaNinja](https://github.com/ali-vilab/MangaNinjia) which‌ is a " Line Art Colorization with Precise Reference Following " method。

# update
* use single clip now 改成使用单体clip_l模型

* 坐标参考在comfy里实现麻烦，主要是要前端实现gradio的蒙版多图层，我就偷懒用一个蒙版吧。 如果使用外接蒙版（在参考图和线稿同一物理位置，用蒙版编辑点一个点）作为位置参考，guidance_scale_point>15,guidance_scale_ref>9,如果不使用，则guidance_scale_point>9,guidance_scale_ref>15，参数自己调节到合适的。guidance_scale_point是倾向于用线稿和参考图的位置点来上色，guidance_scale_ref是主要用参考图来上色;
* Coordinate reference is troublesome to implement in comfy, mainly to implement the multi-layer mask of gradio on the front end, so I'm lazy to use a mask. If you use an external mask (at the same physical location as the reference diagram and line artwork, edit a point with the mask) as the position reference, guidance_scale_point> 15,guidance_scale_ref>9, and if you don't, guidance_scale_point>9,guidance_scale_ref>15. The parameters are adjusted to the appropriate one. guidance_scale_point tend to use line drawings and reference drawings for coloring, guidance_scale_ref mainly use reference drawings for coloring;

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
![](https://github.com/smthemex/ComfyUI_MangaNinjia/blob/main/exampleA.png)
* NEW
![](https://github.com/smthemex/ComfyUI_MangaNinjia/blob/main/exampleB.png)


# 6.Citation
```
@article{liu2025manganinja,
  title={MangaNinja: Line Art Colorization with Precise Reference Following},
  author={Liu, Zhiheng and Cheng, Ka Leong and Chen, Xi and Xiao, Jie and Ouyang, Hao and Zhu, Kai and Liu, Yu and Shen, Yujun and Chen, Qifeng and Luo, Ping},
  journal={arXiv preprint arXiv:2501.08332},
  year={2025}
}
```
