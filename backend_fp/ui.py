from diffusers_helper.hf_login import login
import gradio as gr
from diffusers_helper.gradio.progress_bar import make_progress_bar_css, make_progress_bar_html
from backend_fp.inferrence import process, end_process
from backend_fp.prompt_handler import get_section_boundaries, get_quick_prompts
import os
import random
from typing import List

# LoRA目录
lora_dir = 'models/hunyuan/lora'
os.makedirs(lora_dir, exist_ok=True)

# 加载LoRA文件列表
lora_names = []
lora_sliders = {}
if os.path.isdir(lora_dir):
    lora_files = [f for f in os.listdir(lora_dir) if f.endswith('.safetensors') or f.endswith('.pt')]
    for lora_file in lora_files:
        lora_names.append(lora_file.split('.')[0])

# Gradio界面构建函数
def create_ui():
    global lora_names, lora_sliders
    css = make_progress_bar_css()
    css += """
    .contain-image img {
        object-fit: contain !important;
        width: 100% !important;
        height: 100% !important;
        background: #222;
    }
    """
    section_boundaries = get_section_boundaries()
    quick_prompts = get_quick_prompts()
    
    block = gr.Blocks(css=css).queue()
    with block:
        gr.Markdown('# FramePack')
        with gr.Tabs():
            with gr.TabItem("Generate"):
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            with gr.Column():
                                input_image = gr.Image(sources='upload', type="numpy", label="Start Frame (Optional)", height=320, elem_classes="contain-image")
                            with gr.Column():
                                end_image = gr.Image(sources='upload', type="numpy", label="End Frame (Optional)", height=320, elem_classes="contain-image")
                        with gr.Accordion("Latent Image Options", open=False):
                            latent_type = gr.Dropdown(
                                ["Black", "White", "Noise", "Green Screen"], label="Latent Image", value="Black",
                                info="Used as a starting point if no start frame is provided"
                            )
                        prompt = gr.Textbox(label="Prompt", value='The girl dances gracefully, with clear movements, full of charm.')
                        resolution = gr.Slider(label="Resolution", minimum=240, maximum=720, value=640, step=16)
                        example_quick_prompts = gr.Dataset(samples=quick_prompts, label='Quick List', samples_per_page=1000, components=[prompt])
                        example_quick_prompts.click(lambda x: x[0], inputs=[example_quick_prompts], outputs=prompt, show_progress=False, queue=False)
                        with gr.Accordion("Prompt Parameters", open=False):
                            blend_sections = gr.Slider(minimum=0, maximum=10, value=4, step=1, label="Number of sections to blend between prompts")
                        with gr.Accordion("Generation Parameters", open=True):
                            with gr.Row():
                                steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=25, step=1)
                                total_second_length = gr.Slider(label="Video Length (Seconds)", minimum=1, maximum=120, value=5, step=0.1)
                            with gr.Row():
                                lora_selector = gr.Dropdown(choices=lora_names, label="Select LoRAs", multiselect=True, value=[])
                                gr.Markdown("No LoRA models found. Please upload .safetensors files in the LoRA Management tab.", visible=len(lora_names) == 0)
                                for lora in lora_names:
                                    lora_sliders[lora] = gr.Slider(minimum=0.0, maximum=2.0, value=1.0, step=0.01, label=f"{lora} Weight", visible=False)
                            with gr.Row():
                                json_upload = gr.File(label="Upload Metadata JSON", file_types=[".json"], type="filepath")
                                save_metadata = gr.Checkbox(label="Save Metadata", value=True)
                            with gr.Row():
                                use_teacache = gr.Checkbox(label='Use TeaCache', value=True)
                                n_prompt = gr.Textbox(label="Negative Prompt", value="", visible=False)
                            with gr.Row():
                                seed = gr.Number(label="Seed", value=-1, precision=0)
                                randomize_seed = gr.Checkbox(label="Randomize", value=False)
                        with gr.Accordion("Advanced Parameters", open=False):
                            latent_window_size = gr.Slider(label="Latent Window Size", minimum=1, maximum=33, value=9, step=1)
                            cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=1.0, step=0.01)
                            gs = gr.Slider(label="Distilled CFG Scale", minimum=1.0, maximum=32.0, value=10.0, step=0.01)
                            rs = gr.Slider(label="CFG Re-Scale", minimum=0.0, maximum=1.0, value=0.0, step=0.01)
                            gpu_memory_preservation = gr.Slider(label="GPU Inference Preserved Memory (GB)", minimum=0, maximum=128, value=6, step=0.1)
                        with gr.Accordion("Output Parameters", open=False):
                            mp4_crf = gr.Slider(label="MP4 Compression", minimum=0, maximum=100, value=16, step=1)
                            clean_up_videos = gr.Checkbox(label="Clean up video files", value=True)
                        with gr.Row():
                            start_button = gr.Button(value="Generate")
                            end_button = gr.Button(value="Cancel", interactive=True)
                        error_message = gr.Markdown("", visible=False)
                    with gr.Column():
                        preview_image = gr.Image(label="Next Latents", height=200, visible=True)
                        result_video = gr.Video(label="Finished Frames", autoplay=True, show_share_button=False, height=512, loop=True)
                        gr.Markdown(f'When using only a start frame, the ending actions will be generated before the starting actions due to inverted sampling. Section boundaries: {section_boundaries}')
                        progress_desc = gr.Markdown('', elem_classes='no-generating-animation')
                        progress_bar = gr.HTML('', elem_classes='no-generating-animation')
            with gr.TabItem("LoRA Management"):
                gr.Markdown("## Manage LoRA Models")
                with gr.Row():
                    lora_upload = gr.File(label="Upload LoRA File (.safetensors)", file_types=[".safetensors"], type="filepath")
                    lora_upload_status = gr.Markdown("")
                with gr.Row():
                    lora_list = gr.DataFrame(headers=["Name", "Path", "Status"], datatype=["str", "str", "str"], label="Installed LoRAs")
                    refresh_lora_btn = gr.Button(value="Refresh List")
        
        # LoRA列表刷新
        def list_loras():
            global lora_names
            loras = []
            lora_names = []
            for file in os.listdir(lora_dir):
                if file.endswith('.safetensors') or file.endswith('.pt'):
                    lora_base_name = file.split('.')[0]
                    loras.append([lora_base_name, os.path.join(lora_dir, file), "Loaded"])
                    if lora_base_name not in lora_names:
                        lora_names.append(lora_base_name)
            return loras
        
        # LoRA滑块更新
        def update_lora_sliders(selected_loras):
            return [gr.update(visible=(lora in selected_loras)) for lora in lora_names]
        
        # LoRA上传处理
        def handle_lora_upload(file):
            global lora_names, lora_sliders
            if not file:
                return gr.update(), "No file uploaded", gr.update()
            try:
                _, lora_name = os.path.split(file.name)
                lora_dest = os.path.join(lora_dir, lora_name)
                import shutil
                shutil.copy(file.name, lora_dest)
                lora_base_name = lora_name.split('.')[0]
                if lora_base_name not in lora_names:
                    lora_names.append(lora_base_name)
                    lora_sliders[lora_base_name] = gr.Slider(minimum=0.0, maximum=2.0, value=1.0, step=0.01, label=f"{lora_base_name} Weight", visible=False)
                return gr.update(choices=lora_names), f"Successfully uploaded {lora_name}", gr.update()
            except Exception as e:
                return gr.update(), f"Error uploading LoRA: {e}", gr.update()
        
        # JSON元数据加载
        def load_metadata_from_json(json_path):
            if not json_path:
                return [gr.update(), gr.update(), gr.update()]
            try:
                import json
                with open(json_path, 'r') as f:
                    metadata = json.load(f)
                return [
                    gr.update(value=metadata.get('prompt')),
                    gr.update(value=metadata.get('seed')),
                    gr.update()
                ]
            except Exception as e:
                return [gr.update(), gr.update(), gr.update(value=f"Error loading JSON: {e}")]
        
        # 处理生成请求
        def process_directly(*args):
            input_image, end_image, latent_type, prompt_text, _, n_prompt, blend_sections, steps, total_second_length, lora_selector, json_upload, save_metadata, use_teacache, seed_value, randomize_seed_checked, latent_window_size, cfg, gs, rs, gpu_memory_preservation, mp4_crf, clean_up_videos, resolution = args[:23]
            lora_values = args[23:]
            if randomize_seed_checked:
                seed_value = random.randint(0, 2**32 - 1)
            try:
                for outputs in process(
                    input_image=input_image,
                    end_image=end_image,
                    latent_type=latent_type,
                    prompt_text=prompt_text,
                    n_prompt=n_prompt,
                    seed=seed_value,
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
                    clean_up_videos=clean_up_videos,
                    selected_loras=lora_selector,
                    lora_values=lora_values,
                    randomize_seed=randomize_seed_checked
                ):
                    video_path, preview, desc, html, start_button_state, end_button_state, seed_state, error_state = outputs
                    yield [video_path, preview, desc, html, start_button_state, end_button_state, seed_state, error_state]
            except Exception as e:
                desc = f"Error: Failed to generate video: {str(e)}"
                html = make_progress_bar_html(0, desc)
                yield [None, None, desc, html, gr.update(interactive=True), gr.update(interactive=False), gr.update(value=seed_value), gr.update(value=desc, visible=True)]
        
        # 输入参数
        ips = [
            input_image, end_image, latent_type, prompt, example_quick_prompts, n_prompt, blend_sections,
            steps, total_second_length, lora_selector, json_upload, save_metadata, use_teacache, seed, randomize_seed,
            latent_window_size, cfg, gs, rs, gpu_memory_preservation, mp4_crf, clean_up_videos, resolution
        ] + [lora_sliders[lora] for lora in lora_names]
        
        # 连接事件
        start_button.click(
            fn=process_directly,
            inputs=ips,
            outputs=[result_video, preview_image, progress_desc, progress_bar, start_button, end_button, seed, error_message]
        )
        end_button.click(fn=end_process, outputs=[])
        lora_selector.change(fn=update_lora_sliders, inputs=[lora_selector], outputs=[lora_sliders[lora] for lora in lora_names])
        lora_upload.change(fn=handle_lora_upload, inputs=[lora_upload], outputs=[lora_selector, lora_upload_status, error_message])
        json_upload.change(fn=load_metadata_from_json, inputs=[json_upload], outputs=[prompt, seed, error_message])
        refresh_lora_btn.click(fn=list_loras, outputs=[lora_list])
    
    return block
