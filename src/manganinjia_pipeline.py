
from typing import Any, Dict, Union
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm.auto import tqdm
from PIL import Image
from diffusers import (
    DiffusionPipeline,
    ControlNetModel,
    DDIMScheduler,
    AutoencoderKL,
)
from diffusers.utils import BaseOutput
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import CLIPImageProcessor
from transformers import CLIPVisionModelWithProjection
import gc
from .image_util import resize_max_res,chw2hwc
from .point_network import PointNet
from .models.mutual_self_attention_multi_scale import ReferenceAttentionControl
from .models.unet_2d_condition import UNet2DConditionModel
from .models.refunet_2d_condition import RefUNet2DConditionModel


class MangaNinjiaPipelineOutput(BaseOutput):
    latent: torch.Tensor
    to_save_dict: dict


class MangaNinjiaPipeline(DiffusionPipeline):
    rgb_latent_scale_factor = 0.18215
    
    def __init__(self,
        reference_unet: RefUNet2DConditionModel,
        controlnet: ControlNetModel,
        denoising_unet: UNet2DConditionModel,
        #vae: AutoencoderKL,
        # refnet_tokenizer: CLIPTokenizer,
        # refnet_text_encoder: CLIPTextModel,
        # refnet_image_encoder: CLIPVisionModelWithProjection,
        # controlnet_tokenizer: CLIPTokenizer,
        # controlnet_text_encoder: CLIPTextModel,
        # controlnet_image_encoder: CLIPVisionModelWithProjection,
        scheduler: DDIMScheduler,
        point_net: PointNet
    ):
        super().__init__()
            
        self.register_modules(
            reference_unet=reference_unet,
            controlnet=controlnet,
            denoising_unet=denoising_unet,       
            #vae=vae,
            # refnet_tokenizer=refnet_tokenizer,
            # refnet_text_encoder=refnet_text_encoder,
            # refnet_image_encoder=refnet_image_encoder,
            # controlnet_tokenizer=controlnet_tokenizer,
            # controlnet_text_encoder=controlnet_text_encoder,
            # controlnet_image_encoder=controlnet_image_encoder,
            point_net=point_net,
            scheduler=scheduler,
        )
        self.empty_text_embed = None
        self.clip_image_processor = CLIPImageProcessor()
        
    @torch.no_grad()
    def __call__(
        self,
        is_lineart: bool,
        ref1: Image.Image,
        raw2: Image.Image,
        edit2: Image.Image,
        denosing_steps: int = 20,
        processing_res: int = 512,
        match_input_res: bool = True,
        batch_size: int = 0,
        show_progress_bar: bool = True,
        guidance_scale_ref: float = 7,
        guidance_scale_point: float = 12,
        preprocessor=None,
        generator=None,
        point_ref=None,
        point_main=None,
        controlnet_encoder_hidden_states=None,
        controlnet_uncond_encoder_hidden_states=None,
        refnet_encoder_hidden_states=None,
        refnet_uncond_encoder_hidden_states=None,
        ref1_latents=None,
    ) -> MangaNinjiaPipelineOutput:
        controlnet_encoder_hidden_states=controlnet_encoder_hidden_states
        controlnet_uncond_encoder_hidden_states=controlnet_uncond_encoder_hidden_states
        refnet_encoder_hidden_states=refnet_encoder_hidden_states
        refnet_uncond_encoder_hidden_states=refnet_uncond_encoder_hidden_states

        device = self.device
        self.ref1_latents=ref1_latents
        input_size = raw2.size
        point_ref=point_ref.float().to(device)
        point_main=point_main.float().to(device)
        # def img2embeds(img, image_enc):
        #     clip_image = self.clip_image_processor.preprocess(
        #         img, return_tensors="pt"
        #     ).pixel_values
        #     clip_image_embeds = image_enc(
        #         clip_image.to(device, dtype=image_enc.dtype)
        #     ).image_embeds
        #     encoder_hidden_states = clip_image_embeds.unsqueeze(1)
        #     return encoder_hidden_states
        # if self.reference_unet:
        #     refnet_encoder_hidden_states = img2embeds(ref1, self.refnet_image_encoder)
        # else:
        #     refnet_encoder_hidden_states = None
        # if self.controlnet:
        #     controlnet_encoder_hidden_states = img2embeds(ref1, self.controlnet_image_encoder)
        # else:
        #     controlnet_encoder_hidden_states = None

        # prompt = ""
        # def prompt2embeds(prompt, tokenizer, text_encoder):
        #     text_inputs = tokenizer(
        #         prompt,
        #         padding="do_not_pad",
        #         max_length=tokenizer.model_max_length,
        #         truncation=True,
        #         return_tensors="pt",
        #     )
        #     text_input_ids = text_inputs.input_ids.to(device) #[1,2]
        #     empty_text_embed = text_encoder(text_input_ids)[0].to(self.dtype)
        #     uncond_encoder_hidden_states = empty_text_embed.repeat((1, 1, 1))[:,0,:].unsqueeze(0)
        #     return uncond_encoder_hidden_states
        # if self.reference_unet:
        #     refnet_uncond_encoder_hidden_states = prompt2embeds(prompt, self.refnet_tokenizer, self.refnet_text_encoder)
        # else:
        #     refnet_uncond_encoder_hidden_states = None
        # if self.controlnet:
        #     controlnet_uncond_encoder_hidden_states = prompt2embeds(prompt, self.controlnet_tokenizer, self.controlnet_text_encoder)
        # else:
        #     controlnet_uncond_encoder_hidden_states = None

        do_classifier_free_guidance = guidance_scale_ref > 1.0
        
        # adjust the input resolution.
        if not match_input_res:
            assert (
                processing_res is not None                
            )," Value Error: `resize_output_back` is only valid with "
        
        assert processing_res >= 0
        assert denosing_steps >= 1
        
        # --------------- Image Processing ------------------------
        # Resize image
        if processing_res > 0:
            def resize_img(img):
                img = resize_max_res(img, max_edge_resolution=processing_res)
                return img
            ref1 = resize_img(ref1)
            raw2 = resize_img(raw2)
            edit2 = resize_img(edit2)
        
        # Normalize image
        def normalize_img(img):
            img = img.convert("RGB")
            img = np.array(img)

            # Normalize RGB Values.
            rgb = np.transpose(img,(2,0,1))
            rgb_norm = rgb / 255.0 * 2.0 - 1.0
            rgb_norm = torch.from_numpy(rgb_norm).to(self.dtype)
            rgb_norm = rgb_norm.to(device)
            img = rgb_norm
            assert img.min() >= -1.0 and img.max() <= 1.0
            return img
        raw2_real = raw2.convert('L')
        ref1 = normalize_img(ref1)
        raw2 = normalize_img(raw2)
        edit2 = normalize_img(edit2)
        single_rgb_dataset = TensorDataset(ref1[None], raw2[None], edit2[None])

        
        # find the batch size
        if batch_size>0:
            _bs = batch_size
        else:
            _bs = 1
        point_ref=self.point_net(point_ref)
        point_main=self.point_net(point_main)
        single_rgb_loader = DataLoader(single_rgb_dataset,batch_size=_bs,shuffle=False)
        
        # load the model to cpu
        self.point_net.to("cpu")
        # self.refnet_image_encoder.to("cpu")
        # self.refnet_text_encoder.to("cpu")
        # self.controlnet_image_encoder.to("cpu")
        # self.controlnet_text_encoder.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
        # classifier guidance
        if do_classifier_free_guidance:
            if self.reference_unet:
                refnet_encoder_hidden_states = torch.cat(
                    [refnet_uncond_encoder_hidden_states, refnet_encoder_hidden_states,refnet_encoder_hidden_states], dim=0
                )
            else:
                refnet_encoder_hidden_states = None

            if self.controlnet:
                controlnet_encoder_hidden_states = torch.cat(
                    [controlnet_uncond_encoder_hidden_states, controlnet_encoder_hidden_states,controlnet_encoder_hidden_states], dim=0
                )
            else:
                controlnet_encoder_hidden_states = None

        if self.reference_unet:
            reference_control_writer = ReferenceAttentionControl(
                self.reference_unet,
                do_classifier_free_guidance=do_classifier_free_guidance,
                mode="write",
                batch_size=batch_size,
                fusion_blocks="full",
            )
            reference_control_reader = ReferenceAttentionControl(
                self.denoising_unet,
                do_classifier_free_guidance=do_classifier_free_guidance,
                mode="read",
                batch_size=batch_size,
                fusion_blocks="full",
            )
        else:
            reference_control_writer = None
            reference_control_reader = None
            
        if show_progress_bar:
            iterable_bar = tqdm(
                single_rgb_loader, desc=" " * 2 + "Inference batches", leave=False
            )
        else:
            iterable_bar = single_rgb_loader
        
        assert len(iterable_bar) == 1
        for batch in iterable_bar:
            (ref1, raw2, edit2) = batch  # here the image is still around 0-1
            if is_lineart:
                raw2 = raw2_real
            img_pred, to_save_dict = self.single_infer(
                is_lineart=is_lineart,
                ref1=ref1,
                raw2=raw2,
                edit2=edit2,
                num_inference_steps=denosing_steps,
                show_pbar=show_progress_bar,
                guidance_scale_ref=guidance_scale_ref,
                guidance_scale_point=guidance_scale_point,
                refnet_encoder_hidden_states=refnet_encoder_hidden_states,
                controlnet_encoder_hidden_states=controlnet_encoder_hidden_states,
                reference_control_writer=reference_control_writer,
                reference_control_reader=reference_control_reader,
                preprocessor=preprocessor,
                generator=generator,
                point_ref=point_ref,
                point_main=point_main,
                ref1_latents=self.ref1_latents,
            )
            for k, v in to_save_dict.items():
                if k =='edge2_black':
                    to_save_dict[k] = Image.fromarray(
                   ((to_save_dict['edge2_black'][:,0].squeeze().detach().cpu().numpy() + 1.) / 2 * 255).astype(np.uint8)
                    )
                else:
                    try:
                        to_save_dict[k] = Image.fromarray(
                        chw2hwc(((v.squeeze().detach().cpu().numpy() + 1.) / 2 * 255).astype(np.uint8))
                        )
                    except:
                        import ipdb;ipdb.set_trace()
        
        torch.cuda.empty_cache()  # clear vram cache for ensembling
        
        # ----------------- Post processing -----------------        
        # Convert to numpy
        # img_pred = img_pred.squeeze().cpu().numpy().astype(np.float32)
        # img_pred_np = (((img_pred + 1.) / 2.) * 255).astype(np.uint8)
        # img_pred_np = chw2hwc(img_pred_np)
        # img_pred_pil = Image.fromarray(img_pred_np)

        # Resize back to original resolution
        # if match_input_res:
        #     img_pred_pil = img_pred_pil.resize(input_size)
        #     #img_pred_np = np.asarray(img_pred_pil)        

        return MangaNinjiaPipelineOutput(
            latent=img_pred,
            to_save_dict=to_save_dict
        )

    
    def __encode_empty_text(self):
        """
        Encode text embedding for empty prompt
        """
        prompt = ""
        text_inputs = self.tokenizer(
            prompt,
            padding="do_not_pad",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device) #[1,2]
        # print(text_input_ids.shape)
        self.empty_text_embed = self.text_encoder(text_input_ids)[0].to(self.dtype) #[1,2,1024]

    def get_timesteps(self, num_inference_steps, strength, device, denoising_start=None):
        # get the original timestep using init_timestep
        if denoising_start is None:
            init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
            t_start = max(num_inference_steps - init_timestep, 0)
        else:
            t_start = 0

        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]

        # Strength is irrelevant if we directly request a timestep to start at;
        # that is, strength is determined by the denoising_start instead.
        if denoising_start is not None:
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (denoising_start * self.scheduler.config.num_train_timesteps)
                )
            )
            timesteps = list(filter(lambda ts: ts < discrete_timestep_cutoff, timesteps))
            return torch.tensor(timesteps), len(timesteps)

        return timesteps, num_inference_steps - t_start
        
    @torch.no_grad()
    def single_infer(
        self,
        is_lineart: bool,
        ref1: torch.Tensor,
        raw2: torch.Tensor,
        edit2: torch.Tensor,
        num_inference_steps: int,
        show_pbar: bool,
        guidance_scale_ref: float,
        guidance_scale_point: float,
        refnet_encoder_hidden_states: torch.Tensor,
        controlnet_encoder_hidden_states: torch.Tensor,
        reference_control_writer: ReferenceAttentionControl,
        reference_control_reader: ReferenceAttentionControl,
        preprocessor,
        generator,
        point_ref,
        point_main,
        ref1_latents
    ):
        do_classifier_free_guidance = guidance_scale_ref > 1.0
        device = ref1.device
        to_save_dict = {
            'ref1': ref1,
        }
        
        # Set timesteps: inherit from the diffuison pipeline
        self.scheduler.set_timesteps(num_inference_steps, device=device) # here the numbers of the steps is only 10.
        timesteps = self.scheduler.timesteps  # [T]
        
        # encode image
        #ref1_latents = self.encode_RGB(ref1, generator=generator) # 1/8 Resolution with a channel nums of 4. 
        edge2_src = raw2

        timesteps_add,_=self.get_timesteps(num_inference_steps, 1.0, device, denoising_start=None)
        if is_lineart is not True:
            edge2 = preprocessor(edge2_src)
        else:
            gray_image_np = np.array(edge2_src)
            gray_image_np = gray_image_np / 255.0
            edge2 = torch.from_numpy(gray_image_np.astype(np.float32)).unsqueeze(0).unsqueeze(0).cuda()
        edge2[edge2<=0.24]=0
        edge2_black = edge2.repeat(1, 3, 1, 1) * 2 - 1.
        to_save_dict['edge2_black']=edge2_black

        edge2 = edge2.repeat(1, 3, 1, 1) * 2 - 1.
        to_save_dict['edge2'] = (1-((edge2+1.)/2))*2-1
        
        noisy_edit2_latents = torch.randn(
            ref1_latents.shape, device=device, dtype=self.dtype
        )  # [B, 4, H/8, W/8]
            

        # Denoising loop
        if show_pbar:
            iterable = tqdm(
                enumerate(timesteps),
                total=len(timesteps),
                leave=False,
                desc=" " * 4 + "Diffusion denoising",
            )
        else:
            iterable = enumerate(timesteps)

        for i, t in iterable:
            
            refnet_input = ref1_latents
            controlnet_inputs = (noisy_edit2_latents, edge2)
            unet_input = torch.cat([noisy_edit2_latents], dim=1)

            if i == 0:
                if self.reference_unet:
                    # self.reference_unet.to(self.device)
                    self.reference_unet(
                        refnet_input.repeat(
                            (3 if do_classifier_free_guidance else 1), 1, 1, 1
                        ),
                        torch.zeros_like(t),
                        
                        encoder_hidden_states=refnet_encoder_hidden_states,
                        return_dict=False,
                    )
                    reference_control_reader.update(reference_control_writer,point_embedding_ref=point_ref,point_embedding_main=point_main)#size不对
                    # self.reference_unet.to("cpu") #只计算第一步
                    # torch.cuda.empty_cache()

            if self.controlnet:
                noisy_latents, controlnet_cond = controlnet_inputs
                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    noisy_latents.repeat(
                        (3 if do_classifier_free_guidance else 1), 1, 1, 1
                    ),
                    t,
                    encoder_hidden_states=controlnet_encoder_hidden_states,
                    controlnet_cond=controlnet_cond.repeat(
                        (3 if do_classifier_free_guidance else 1), 1, 1, 1
                    ),
                    return_dict=False,
                )
            else:
                down_block_res_samples, mid_block_res_sample = None, None
            
            # predict the noise residual
            noise_pred = self.denoising_unet(
                unet_input.repeat(
                    (3 if do_classifier_free_guidance else 1), 1, 1, 1
                ).to(dtype=self.denoising_unet.dtype), 
                t, 
                encoder_hidden_states=refnet_encoder_hidden_states,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            ).sample  # [B, 4, h, w]
            noise_pred_uncond, noise_pred_ref, noise_pred_point = noise_pred.chunk(3)
            noise_pred_1 = noise_pred_uncond + guidance_scale_ref * (
                noise_pred_ref - noise_pred_uncond
            )
            noise_pred_2 = noise_pred_ref + guidance_scale_point * (
                noise_pred_point - noise_pred_ref
            )
            noise_pred=(noise_pred_1+noise_pred_2)/2
            noisy_edit2_latents = self.scheduler.step(noise_pred, t, noisy_edit2_latents).prev_sample

        self.controlnet.to("cpu")
        self.reference_unet.to("cpu")
        reference_control_reader.clear()
        reference_control_writer.clear()
        gc.collect()
        torch.cuda.empty_cache()

        # clip prediction
        # self.vae.to("cuda")
        # edit2 = self.decode_RGB(noisy_edit2_latents)
        # edit2 = torch.clip(edit2, -1.0, 1.0)

        return noisy_edit2_latents, to_save_dict
        
    
    def encode_RGB(self,rgb_in: torch.Tensor, generator) -> torch.Tensor:
        """
        Encode RGB image into latent.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image to be encoded.

        Returns:
            `torch.Tensor`: Image latent.
        """
        
        # generator = None
        rgb_latent =self.encode(rgb_in).latent_dist.sample(generator)
        rgb_latent = rgb_latent * self.rgb_latent_scale_factor
        self.vae.to("cpu")
        return rgb_latent
               
   
    
    def decode_RGB(self, rgb_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode depth latent into depth map.

        Args:
            rgb_latent (`torch.Tensor`):
                Depth latent to be decoded.

        Returns:
            `torch.Tensor`: Decoded depth map.
        """
       

        rgb_latent = rgb_latent / self.rgb_latent_scale_factor
        rgb_out = self.vae.decode(rgb_latent, return_dict=False)[0]
        return rgb_out


