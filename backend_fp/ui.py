from diffusers_helper.hf_login import login
import gradio as gr
from diffusers_helper.gradio.progress_bar import make_progress_bar_css, make_progress_bar_html
from backend_fp.inferrence import process, end_process

quick_prompts = [
    'The girl dances gracefully, with clear movements, full of charm.',
    'A character doing some simple body movements.',
]
quick_prompts = [[x] for x in quick_prompts]

# Gradio 界面构建函数
def create_ui():
    css = make_progress_bar_css()
    block = gr.Blocks(css=css).queue()
    with block:
        gr.Markdown('# FramePack')
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        input_image = gr.Image(sources='upload', type="numpy", label="Start Frame", height=320)
                    with gr.Column():
                        end_image = gr.Image(sources='upload', type="numpy", label="End Frame (Optional)", height=320)
                prompt = gr.Textbox(label="Prompt", value='')
                resolution = gr.Slider(label="Resolution", minimum=240, maximum=720, value=640, step=16)
                example_quick_prompts = gr.Dataset(samples=quick_prompts, label='Quick List', samples_per_page=1000, components=[prompt])
                example_quick_prompts.click(lambda x: x[0], inputs=[example_quick_prompts], outputs=prompt, show_progress=False, queue=False)
                with gr.Row():
                    start_button = gr.Button(value="Start Generation")
                    end_button = gr.Button(value="End Generation", interactive=False)
                with gr.Group():
                    use_teacache = gr.Checkbox(label='Use TeaCache', value=True, info='Faster speed, but often makes hands and fingers slightly worse.')
                    n_prompt = gr.Textbox(label="Negative Prompt", value="", visible=False)
                    seed = gr.Number(label="Seed", value=-1, precision=0)
                    total_second_length = gr.Slider(label="Total Video Length (Seconds)", minimum=1, maximum=120, value=5, step=0.1)
                    latent_window_size = gr.Slider(label="Latent Window Size", minimum=1, maximum=33, value=9, step=1, visible=False)
                    steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=25, step=1, info='Changing this value is not recommended.')
                    cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=1.0, step=0.01, visible=False)
                    gs = gr.Slider(label="Distilled CFG Scale", minimum=1.0, maximum=32.0, value=10.0, step=0.01, info='Changing this value is not recommended.')
                    rs = gr.Slider(label="CFG Re-Scale", minimum=0.0, maximum=1.0, value=0.0, step=0.01, visible=False)
                    gpu_memory_preservation = gr.Slider(label="GPU Inference Preserved Memory (GB) (larger means slower)", minimum=0, maximum=128, value=6, step=0.1, info="Set this number to a larger value if you encounter OOM. Larger value causes slower speed.")
                    mp4_crf = gr.Slider(label="MP4 Compression", minimum=0, maximum=100, value=16, step=1, info="Lower means better quality. 0 is uncompressed. Change to 16 if you get black outputs. ")
            with gr.Column():
                preview_image = gr.Image(label="Next Latents", height=200, visible=False)
                result_video = gr.Video(label="Finished Frames", autoplay=True, show_share_button=False, height=512, loop=True)
                gr.Markdown('When using only a start frame, the ending actions will be generated before the starting actions due to the inverted sampling. If using both start and end frames, the model will try to create a smooth transition between them.')
                progress_desc = gr.Markdown('', elem_classes='no-generating-animation')
                progress_bar = gr.HTML('', elem_classes='no-generating-animation')
        gr.HTML('<div style="text-align:center; margin-top:20px;">Share your results and find ideas at the <a href="https://x.com/search?q=framepack&f=live" target="_blank">FramePack Twitter (X) thread</a></div>')
        ips = [input_image, end_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf, resolution]
        start_button.click(fn=process, inputs=ips, outputs=[result_video, preview_image, progress_desc, progress_bar, start_button, end_button])
        end_button.click(fn=end_process)
    return block