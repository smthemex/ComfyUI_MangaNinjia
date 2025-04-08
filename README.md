# ComfyUI_MangaNinjia
ComfyUI_MangaNinjia is a ComfyUI node of [MangaNinja](https://github.com/ali-vilab/MangaNinjia) which‌ is a " Line Art Colorization with Precise Reference Following " method。

# update
* Adding a manual reference point node, my front-end skills are almost zeroBy using this node, the coordinate reference value can be adjusted to around 20
* 添加一个手动的参考点节点，因为我的前端技术狗屎一样,使用该节点可以将坐标参考的数值调到20左右,欢迎提交PR修改该节点;
* 使用方法:拉两个参考点节点,然后点击按钮加载参考图和线稿,选中其中的画布,鼠标左键是标记坐标点,右键取消上一次,中键全部取消,下载按钮则是下载json文件用.假设你2张图都处理了,获得了2个json文件,将两个json文件的绝对路径填入地址栏,即可使用参考点模式.
* Usage: Pull two reference point nodes, then click the button to load the reference image and line draft, select the canvas, left click to mark the coordinate points, right-click to cancel the previous one, middle click to cancel all, and the download button is for downloading JSON files Assuming you have processed both images and obtained two JSON files, fill in the absolute paths of the two JSON files in the address bar to use the reference point mode 


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
* NEW
![](https://github.com/smthemex/ComfyUI_MangaNinjia/blob/main/example.png)
* old
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
