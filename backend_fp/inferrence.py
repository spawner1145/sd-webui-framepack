from diffusers_helper.hf_login import login
import os
import gradio as gr
import torch
import traceback
import einops
import safetensors.torch as sf
import numpy as np
import math
import json
import time
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, generate_timestamp
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete
from diffusers_helper.thread_utils import AsyncStream, async_run
from diffusers_helper.gradio.progress_bar import make_progress_bar_html
from transformers import SiglipImageProcessor, SiglipVisionModel
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket
import random
from diffusers_helper import lora_utils
from backend_fp.prompt_handler import parse_timestamped_prompt

# 设置输出和缓存目录
outputs_folder = 'outputs/hunyuan'
try:
    os.makedirs(outputs_folder, exist_ok=True)
except Exception as e:
    print(f"Error creating output directory {outputs_folder}: {e}")
    outputs_folder = 'outputs_fallback'
    os.makedirs(outputs_folder, exist_ok=True)

cache_dir = 'models/hunyuan'
try:
    os.makedirs(cache_dir, exist_ok=True)
except Exception as e:
    print(f"Error creating model cache directory {cache_dir}: {e}")
    cache_dir = os.path.join('models', 'hunyuan_fallback')
    os.makedirs(cache_dir, exist_ok=True)

# 检查 GPU 内存
free_mem_gb = get_cuda_free_memory_gb(gpu)
high_vram = free_mem_gb > 60
print(f'Free VRAM {free_mem_gb} GB')
print(f'High-VRAM Mode: {high_vram}')

# 全局模型变量
text_encoder = None
text_encoder_2 = None
tokenizer = None
tokenizer_2 = None
vae = None
feature_extractor = None
image_encoder = None
transformer = None
stream = AsyncStream()

# LoRA 目录和文件
lora_dir = 'models/hunyuan/lora'
os.makedirs(lora_dir, exist_ok=True)
lora_names = [f.split('.')[0] for f in os.listdir(lora_dir) if f.endswith('.safetensors') or f.endswith('.pt')]

models_loaded = False

def move_lora_adapters_to_device(model, target_device):
    """将 LoRA 适配器移动到指定设备"""
    print(f"Moving all LoRA adapters to {target_device}")
    lora_modules = []
    for name, module in model.named_modules():
        if hasattr(module, 'active_adapter') and hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            lora_modules.append((name, module))
    for name, module in lora_modules:
        active_adapter = module.active_adapter
        if active_adapter is not None:
            if isinstance(module.lora_A, torch.nn.ModuleDict):
                for adapter_name in list(module.lora_A.keys()):
                    if adapter_name in module.lora_A:
                        module.lora_A[adapter_name] = module.lora_A[adapter_name].to(target_device)
                    if adapter_name in module.lora_B:
                        module.lora_B[adapter_name] = module.lora_B[adapter_name].to(target_device)
                    if hasattr(module, 'scaling') and isinstance(module.scaling, dict) and adapter_name in module.scaling:
                        if isinstance(module.scaling[adapter_name], torch.Tensor):
                            module.scaling[adapter_name] = module.scaling[adapter_name].to(target_device)
            else:
                if hasattr(module, 'lora_A') and module.lora_A is not None:
                    module.lora_A = module.lora_A.to(target_device)
                if hasattr(module, 'lora_B') and module.lora_B is not None:
                    module.lora_B = module.lora_B.to(target_device)
                if hasattr(module, 'scaling') and module.scaling is not None:
                    if isinstance(module.scaling, torch.Tensor):
                        module.scaling = module.scaling.to(target_device)
    print(f"Moved all LoRA adapters to {target_device}")
    return model

@torch.no_grad()
def worker(input_image, end_image, prompt_text, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf, resolution, save_metadata, blend_sections, latent_type, clean_up_videos, selected_loras, lora_values=None, job_stream=None):
    """处理视频生成的核心工作函数"""
    global text_encoder, text_encoder_2, tokenizer, tokenizer_2, vae, feature_extractor, image_encoder, transformer
    stream_to_use = job_stream if job_stream is not None else stream
    total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))
    prompt_sections = parse_timestamped_prompt(prompt_text, total_second_length, latent_window_size)
    job_id = generate_timestamp()
    stream_to_use.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Starting ...'))))
    if seed == -1:
        seed = random.randint(0, 2**32 - 1)
    try:
        if not high_vram:
            unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)
        stream_to_use.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Text encoding ...'))))
        if not high_vram:
            fake_diffusers_current_device(text_encoder, gpu)
            load_model_as_complete(text_encoder_2, target_device=gpu)
        unique_prompts = []
        for section in prompt_sections:
            if section.prompt not in unique_prompts:
                unique_prompts.append(section.prompt)
        encoded_prompts = {}
        for prompt in unique_prompts:
            llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
            llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
            encoded_prompts[prompt] = (llama_vec, llama_attention_mask, clip_l_pooler)
        prompt_change_indices = []
        last_prompt = None
        for idx, section in enumerate(prompt_sections):
            if section.prompt != last_prompt:
                prompt_change_indices.append((idx, section.prompt))
                last_prompt = section.prompt
        if cfg == 1:
            llama_vec_n, clip_l_pooler_n = torch.zeros_like(encoded_prompts[prompt_sections[0].prompt][0]), torch.zeros_like(encoded_prompts[prompt_sections[0].prompt][2])
            llama_attention_mask_n = torch.zeros_like(encoded_prompts[prompt_sections[0].prompt][1])
        else:
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
            llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)
        stream_to_use.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Processing start frame ...'))))
        if input_image is None:
            default_height, default_width = resolution, resolution
            if latent_type == "White":
                input_image = np.ones((default_height, default_width, 3), dtype=np.uint8) * 255
            elif latent_type == "Noise":
                input_image = np.random.randint(0, 256, (default_height, default_width, 3), dtype=np.uint8)
            elif latent_type == "Green Screen":
                input_image = np.zeros((default_height, default_width, 3), dtype=np.uint8)
                input_image[:, :, 1] = 177
                input_image[:, :, 2] = 64
            else:
                input_image = np.zeros((default_height, default_width, 3), dtype=np.uint8)
        H, W, C = input_image.shape
        height, width = find_nearest_bucket(H, W, resolution=resolution)
        input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)
        if save_metadata:
            metadata = PngInfo()
            metadata.add_text("prompt", prompt_text)
            metadata.add_text("seed", str(seed))
            Image.fromarray(input_image_np).save(os.path.join(outputs_folder, f'{job_id}.png'), pnginfo=metadata)
            metadata_dict = {
                "prompt": prompt_text,
                "seed": seed,
                "total_second_length": total_second_length,
                "steps": steps,
                "cfg": cfg,
                "gs": gs,
                "rs": rs,
                "latent_type": latent_type,
                "blend_sections": blend_sections,
                "latent_window_size": latent_window_size,
                "mp4_crf": mp4_crf,
                "timestamp": time.time()
            }
            if selected_loras:
                lora_data = {lora_name: float(lora_values[i]) if i < len(lora_values) else 1.0 for i, lora_name in enumerate(selected_loras)}
                metadata_dict["loras"] = lora_data
            with open(os.path.join(outputs_folder, f'{job_id}.json'), 'w') as f:
                json.dump(metadata_dict, f, indent=2)
        else:
            Image.fromarray(input_image_np).save(os.path.join(outputs_folder, f'{job_id}.png'))
        input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]
        has_end_image = end_image is not None
        if has_end_image:
            stream_to_use.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Processing end frame ...'))))
            H_end, W_end, C_end = end_image.shape
            end_image_np = resize_and_center_crop(end_image, target_width=width, target_height=height)
            Image.fromarray(end_image_np).save(os.path.join(outputs_folder, f'{job_id}_end.png'))
            end_image_pt = torch.from_numpy(end_image_np).float() / 127.5 - 1
            end_image_pt = end_image_pt.permute(2, 0, 1)[None, :, None]
        stream_to_use.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'VAE encoding ...'))))
        if not high_vram:
            load_model_as_complete(vae, target_device=gpu)
        start_latent = vae_encode(input_image_pt, vae)
        if has_end_image:
            end_latent = vae_encode(end_image_pt, vae)
        stream_to_use.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'CLIP Vision encoding ...'))))
        if not high_vram:
            load_model_as_complete(image_encoder, target_device=gpu)
        image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state
        if has_end_image:
            end_image_encoder_output = hf_clip_vision_encode(end_image_np, feature_extractor, image_encoder)
            end_image_encoder_last_hidden_state = end_image_encoder_output.last_hidden_state
            image_encoder_last_hidden_state = (image_encoder_last_hidden_state + end_image_encoder_last_hidden_state) / 2
        for prompt_key in encoded_prompts:
            llama_vec, llama_attention_mask, clip_l_pooler = encoded_prompts[prompt_key]
            llama_vec = llama_vec.to(transformer.dtype)
            clip_l_pooler = clip_l_pooler.to(transformer.dtype)
            encoded_prompts[prompt_key] = (llama_vec, llama_attention_mask, clip_l_pooler)
        llama_vec_n = llama_vec_n.to(transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
        llama_attention_mask_n = llama_attention_mask_n.to(transformer.dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)
        stream_to_use.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Start sampling ...'))))
        rnd = torch.Generator("cpu").manual_seed(seed)
        num_frames = latent_window_size * 4 - 3
        history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
        history_pixels = None
        total_generated_latent_frames = 0
        latent_paddings = list(reversed(range(total_latent_sections)))
        if total_latent_sections > 4:
            latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]
        section_idx = 0
        transformer = lora_utils.unload_all_loras(transformer)
        if selected_loras:
            for idx, lora_name in enumerate(selected_loras):
                lora_file = None
                for ext in [".safetensors", ".pt"]:
                    candidate = os.path.join(lora_dir, lora_name + ext)
                    if os.path.exists(candidate):
                        lora_file = lora_name + ext
                        break
                if lora_file:
                    print(f"Loading LoRA {lora_file}")
                    transformer = lora_utils.load_lora(transformer, lora_dir, lora_file)
                    if lora_values and idx < len(lora_values):
                        lora_strength = float(lora_values[idx])
                        for name, module in transformer.named_modules():
                            if hasattr(module, 'scaling'):
                                if isinstance(module.scaling, dict):
                                    if lora_name in module.scaling:
                                        if isinstance(module.scaling[lora_name], torch.Tensor):
                                            module.scaling[lora_name] = torch.tensor(lora_strength, device=module.scaling[lora_name].device)
                                        else:
                                            module.scaling[lora_name] = lora_strength
                                else:
                                    if isinstance(module.scaling, torch.Tensor):
                                        module.scaling = torch.tensor(lora_strength, device=module.scaling.device)
                                    else:
                                        module.scaling = lora_strength
        def callback(d):
            preview = d['denoised']
            preview = vae_decode_fake(preview)
            preview = (preview * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
            preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')
            if stream_to_use.input_queue.top() == 'end':
                stream_to_use.output_queue.push(('end', None))
                raise KeyboardInterrupt('User ends the task.')
            current_step = d['i'] + 1
            percentage = int(100.0 * current_step / steps)
            current_pos = (total_generated_latent_frames * 4 - 3) / 30
            original_pos = total_second_length - current_pos
            if current_pos < 0: current_pos = 0
            if original_pos < 0: original_pos = 0
            hint = f'Sampling {current_step}/{steps}'
            desc = f'Total generated frames: {int(max(0, total_generated_latent_frames * 4 - 3))}, Video length: {max(0, (total_generated_latent_frames * 4 - 3) / 30):.2f} seconds (FPS-30). Current position: {current_pos:.2f}s (original: {original_pos:.2f}s).'
            stream_to_use.output_queue.push(('progress', (preview, desc, make_progress_bar_html(percentage, hint))))
            return preview, desc, make_progress_bar_html(percentage, hint)
        video_path = None
        for latent_padding in latent_paddings:
            is_last_section = latent_padding == 0
            is_first_section = latent_padding == latent_paddings[0]
            latent_padding_size = latent_padding * latent_window_size
            if stream_to_use.input_queue.top() == 'end':
                stream_to_use.output_queue.push(('end', None))
                return
            current_time_position = (total_generated_latent_frames * 4 - 3) / 30
            if current_time_position < 0:
                current_time_position = 0.01
            current_prompt = prompt_sections[0].prompt
            for section in prompt_sections:
                if section.start_time <= current_time_position and (section.end_time is None or current_time_position < section.end_time):
                    current_prompt = section.prompt
                    break
            blend_alpha = None
            prev_prompt = current_prompt
            next_prompt = current_prompt
            for i, (change_idx, prompt) in enumerate(prompt_change_indices):
                if section_idx < change_idx:
                    prev_prompt = prompt_change_indices[i - 1][1] if i > 0 else prompt
                    next_prompt = prompt
                    blend_start = change_idx
                    blend_end = change_idx + blend_sections
                    if section_idx >= change_idx and section_idx < blend_end:
                        blend_alpha = (section_idx - change_idx + 1) / blend_sections
                    break
                elif section_idx == change_idx:
                    if i > 0:
                        prev_prompt = prompt_change_indices[i - 1][1]
                        next_prompt = prompt
                        blend_alpha = 1.0 / blend_sections
                    else:
                        prev_prompt = prompt
                        next_prompt = prompt
                        blend_alpha = None
                    break
            else:
                prev_prompt = current_prompt
                next_prompt = current_prompt
                blend_alpha = None
            if blend_alpha is not None and prev_prompt != next_prompt:
                prev_llama_vec, prev_llama_attention_mask, prev_clip_l_pooler = encoded_prompts[prev_prompt]
                next_llama_vec, next_llama_attention_mask, next_clip_l_pooler = encoded_prompts[next_prompt]
                llama_vec = (1 - blend_alpha) * prev_llama_vec + blend_alpha * next_llama_vec
                llama_attention_mask = prev_llama_attention_mask
                clip_l_pooler = (1 - blend_alpha) * prev_clip_l_pooler + blend_alpha * next_clip_l_pooler
            else:
                llama_vec, llama_attention_mask, clip_l_pooler = encoded_prompts[current_prompt]
            print(f'latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}, time position: {current_time_position:.2f}s')
            indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
            clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)
            clean_latents_pre = start_latent.to(history_latents)
            clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
            clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)
            if has_end_image and is_first_section:
                clean_latents_post = end_latent.to(history_latents)
                clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)
            if not high_vram:
                unload_complete_models()
                move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)
                if lora_names:
                    move_lora_adapters_to_device(transformer, gpu)
            if use_teacache:
                transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
            else:
                transformer.initialize_teacache(enable_teacache=False)
            generated_latents = sample_hunyuan(
                transformer=transformer,
                sampler='unipc',
                width=width,
                height=height,
                frames=num_frames,
                real_guidance_scale=cfg,
                distilled_guidance_scale=gs,
                guidance_rescale=rs,
                num_inference_steps=steps,
                generator=rnd,
                prompt_embeds=llama_vec,
                prompt_embeds_mask=llama_attention_mask,
                prompt_poolers=clip_l_pooler,
                negative_prompt_embeds=llama_vec_n,
                negative_prompt_embeds_mask=llama_attention_mask_n,
                negative_prompt_poolers=clip_l_pooler_n,
                device=gpu,
                dtype=torch.bfloat16,
                image_embeddings=image_encoder_last_hidden_state,
                latent_indices=latent_indices,
                clean_latents=clean_latents,
                clean_latent_indices=clean_latent_indices,
                clean_latents_2x=clean_latents_2x,
                clean_latent_2x_indices=clean_latent_2x_indices,
                clean_latents_4x=clean_latents_4x,
                clean_latent_4x_indices=clean_latent_4x_indices,
                callback=callback,
            )
            if is_last_section:
                generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)
            total_generated_latent_frames += int(generated_latents.shape[2])
            history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)
            if not high_vram:
                if lora_names:
                    move_lora_adapters_to_device(transformer, cpu)
                offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=8)
                load_model_as_complete(vae, target_device=gpu)
            real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]
            if history_pixels is None:
                history_pixels = vae_decode(real_history_latents, vae).cpu()
            else:
                section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
                overlapped_frames = latent_window_size * 4 - 3
                current_pixels = vae_decode(real_history_latents[:, :, :section_latent_frames], vae).cpu()
                history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)
            if not high_vram:
                unload_complete_models()
            output_filename = os.path.join(outputs_folder, f'framepack_{job_id}_{total_generated_latent_frames}.mp4')
            save_bcthw_as_mp4(history_pixels, output_filename, fps=30, crf=mp4_crf)
            print(f'Decoded. Current latent shape {real_history_latents.shape}; pixel shape {history_pixels.shape}')
            stream_to_use.output_queue.push(('file', output_filename))
            video_path = output_filename
            if is_last_section:
                break
            section_idx += 1
        if clean_up_videos:
            video_files = [f for f in os.listdir(outputs_folder) if f.startswith(f"{job_id}_") and f.endswith(".mp4")]
            if video_files:
                def get_frame_count(filename):
                    try:
                        return int(filename.replace(f"{job_id}_", "").replace(".mp4", ""))
                    except Exception:
                        return -1
                video_files_sorted = sorted(video_files, key=get_frame_count)
                final_video = video_files_sorted[-1]
                for vf in video_files_sorted[:-1]:
                    try:
                        os.remove(os.path.join(outputs_folder, vf))
                    except Exception as e:
                        print(f"Failed to delete {vf}: {e}")
                video_path = os.path.join(outputs_folder, final_video)
        stream_to_use.output_queue.push(('end', None))
        return video_path
    except:
        traceback.print_exc()
        if not high_vram:
            unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)
        stream_to_use.output_queue.push(('end', None))
        return

def process(input_image, end_image, latent_type, prompt_text, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf, resolution, save_metadata, blend_sections, clean_up_videos, selected_loras, lora_values, randomize_seed=False):
    """处理视频生成请求的主函数，流式返回8个值以匹配UI输出"""
    global stream, text_encoder, text_encoder_2, tokenizer, tokenizer_2, vae, feature_extractor, image_encoder, transformer, models_loaded
    if not models_loaded:
        print("Loading models...")
        text_encoder = LlamaModel.from_pretrained(
            "hunyuanvideo-community/HunyuanVideo",
            subfolder='text_encoder',
            torch_dtype=torch.float16,
            cache_dir=cache_dir
        ).cpu()
        text_encoder_2 = CLIPTextModel.from_pretrained(
            "hunyuanvideo-community/HunyuanVideo",
            subfolder='text_encoder_2',
            torch_dtype=torch.float16,
            cache_dir=cache_dir
        ).cpu()
        tokenizer = LlamaTokenizerFast.from_pretrained(
            "hunyuanvideo-community/HunyuanVideo",
            subfolder='tokenizer',
            cache_dir=cache_dir
        )
        tokenizer_2 = CLIPTokenizer.from_pretrained(
            "hunyuanvideo-community/HunyuanVideo",
            subfolder='tokenizer_2',
            cache_dir=cache_dir
        )
        vae = AutoencoderKLHunyuanVideo.from_pretrained(
            "hunyuanvideo-community/HunyuanVideo",
            subfolder='vae',
            torch_dtype=torch.float16,
            cache_dir=cache_dir
        ).cpu()
        feature_extractor = SiglipImageProcessor.from_pretrained(
            "lllyasviel/flux_redux_bfl",
            subfolder='feature_extractor',
            cache_dir=cache_dir
        )
        image_encoder = SiglipVisionModel.from_pretrained(
            "lllyasviel/flux_redux_bfl",
            subfolder='image_encoder',
            torch_dtype=torch.float16,
            cache_dir=cache_dir
        ).cpu()
        transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(
            'lllyasviel/FramePackI2V_HY',
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir
        ).cpu()
        vae.eval()
        text_encoder.eval()
        text_encoder_2.eval()
        image_encoder.eval()
        transformer.eval()
        if not high_vram:
            vae.enable_slicing()
            vae.enable_tiling()
        transformer.high_quality_fp32_output_for_inference = True
        transformer.to(dtype=torch.bfloat16)
        vae.to(dtype=torch.float16)
        image_encoder.to(dtype=torch.float16)
        text_encoder.to(dtype=torch.float16)
        text_encoder_2.to(dtype=torch.float16)
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        text_encoder_2.requires_grad_(False)
        image_encoder.requires_grad_(False)
        transformer.requires_grad_(False)
        if not high_vram:
            DynamicSwapInstaller.install_model(transformer, device=gpu)
            DynamicSwapInstaller.install_model(text_encoder, device=gpu)
        else:
            text_encoder.to(gpu)
            text_encoder_2.to(gpu)
            image_encoder.to(gpu)
            vae.to(gpu)
            transformer.to(gpu)
        models_loaded = True
        print("Models loaded successfully.")
    video_path = None
    preview = None
    desc = ''
    html = ''
    error_message = ''
    new_seed = seed if not randomize_seed else random.randint(0, 2**32 - 1)
    try:
        # 初始状态：禁用开始按钮，启用取消按钮
        yield None, None, '', '', gr.update(interactive=False), gr.update(interactive=True), gr.update(value=new_seed), gr.update()
        stream = AsyncStream()
        print(f"Starting new task with stream: {stream}")
        async_run(
            worker,
            input_image=input_image.copy() if input_image is not None else None,
            end_image=end_image,
            prompt_text=prompt_text,
            n_prompt=n_prompt,
            seed=new_seed,
            total_second_length=total_second_length,
            latent_window_size=latent_window_size,
            steps=steps,
            cfg=cfg,
            gs=gs,
            rs=rs,
            gpu_memory_preservation=gpu_memory_preservation,
            use_teacache=use_teacache,
            mp4_crf=mp4_crf,
            resolution=resolution,
            save_metadata=save_metadata,
            blend_sections=blend_sections,
            latent_type=latent_type,
            clean_up_videos=clean_up_videos,
            selected_loras=selected_loras,
            lora_values=lora_values
        )
        while True:
            try:
                flag, data = stream.output_queue.next()
                if flag == 'progress':
                    preview, desc, html = data
                    yield video_path, preview, desc, html, gr.update(interactive=False), gr.update(interactive=True), gr.update(value=new_seed), gr.update()
                elif flag == 'file':
                    video_path = data
                    yield video_path, preview, desc, html, gr.update(interactive=False), gr.update(interactive=True), gr.update(value=new_seed), gr.update()
                elif flag == 'end':
                    print("Task ended")
                    yield video_path, None, '', '', gr.update(interactive=True), gr.update(interactive=False), gr.update(value=new_seed), gr.update()
                    break
            except IndexError:
                time.sleep(0.1)
                continue
    except Exception as e:
        desc = f"Error: {str(e)}"
        html = make_progress_bar_html(0, desc)
        error_message = desc
        print(f"Error occurred: {desc}")
        yield video_path, preview, desc, html, gr.update(interactive=True), gr.update(interactive=False), gr.update(value=new_seed), gr.update(value=error_message, visible=True)
    # 最终返回（仅在异常情况下使用，通常通过 yield 退出）
    return video_path, preview, desc, html, gr.update(interactive=True), gr.update(interactive=False), gr.update(value=new_seed), gr.update(value=error_message, visible=True)

def end_process():
    """取消生成过程"""
    global stream
    print("Cancelling task...")
    stream.input_queue.push('end')
