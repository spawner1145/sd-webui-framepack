import base64
import os
import re
import logging
from typing import Callable
from threading import Lock
from secrets import compare_digest
from io import BytesIO

from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field
from PIL import Image
import numpy as np
from backend_fp.inferrence import worker, process, AsyncStream, async_run
import asyncio

try:
    from modules import shared
    from modules.call_queue import queue_lock as webui_queue_lock
    IN_WEBUI = True
except ImportError:
    IN_WEBUI = False
    shared = type('Shared', (), {'cmd_opts': type('CmdOpts', (), {'api_auth': None})()})()
    webui_queue_lock = None

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Image2VideoRequest(BaseModel):
    input_image: str | None = Field(None, description="首帧图片的 Base64 编码字符串（可选）")
    end_image: str | None = Field(None, description="尾帧图片的 Base64 编码字符串（可选）")
    latent_type: str = Field("Black", pattern="^(Black|White|Noise|Green Screen)$", description="无首帧时使用的潜在图像类型")
    prompt: str = Field(..., min_length=1, description="提示词，支持时间戳格式如[0s-2s: 描述]，支持LoRA语法如<lora:模型名:权重>")
    negative_prompt: str = Field("", description="负向提示词")
    seed: int = Field(-1, ge=-1, le=2**32-1, description="随机种子")
    total_second_length: float = Field(5.0, ge=1.0, le=120.0, description="视频总时长（秒）")
    latent_window_size: int = Field(9, ge=1, le=33, description="潜在窗口大小")
    steps: int = Field(25, ge=1, le=100, description="推理步数")
    cfg: float = Field(1.0, ge=1.0, le=32.0, description="CFG Scale")
    distilled_guidance_scale: float = Field(10.0, ge=1.0, le=32.0, description="Distilled CFG Scale")
    guidance_rescale: float = Field(0.0, ge=0.0, le=1.0, description="CFG Re-Scale")
    gpu_memory_preservation: float = Field(6.0, ge=0.0, le=128.0, description="GPU 推理保留内存（GB）")
    use_teacache: bool = Field(True, description="是否使用 TeaCache")
    mp4_crf: int = Field(16, ge=0, le=100, description="MP4 压缩率，0 为无压缩")
    resolution: int = Field(640, ge=240, le=720, description="视频分辨率")
    save_metadata: bool = Field(False, description="是否保存元数据")
    blend_sections: int = Field(4, ge=0, le=10, description="提示词混合的段数")
    clean_up_videos: bool = Field(True, description="是否清理中间视频文件")

class VideoResponse(BaseModel):
    video: str | None = Field(None, description="生成的视频，Base64 编码的 MP4 文件")
    info: str = Field(..., description="生成信息或错误信息")

class Api:
    def __init__(self, app: FastAPI, queue_lock: Lock = None, prefix: str = "/framepack/v1"):
        self.app = app
        self.queue_lock = queue_lock or Lock()
        self.prefix = prefix
        self.credentials = {}

        if IN_WEBUI and shared.cmd_opts.api_auth:
            for auth in shared.cmd_opts.api_auth.split(","):
                user, password = auth.split(":")
                self.credentials[user] = password

        self.add_api_route(
            "i2v",
            self.endpoint_image2video,
            methods=["POST"],
            response_model=VideoResponse,
            summary="Generate video from an initial image",
            description="Creates a video starting from an initial image (and optional end image) with a text prompt."
        )

    def auth(self, creds: HTTPBasicCredentials = Depends(HTTPBasic())):
        if not self.credentials:
            return True
        if creds.username in self.credentials:
            if compare_digest(creds.password, self.credentials[creds.username]):
                return True
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"}
        )

    def add_api_route(self, path: str, endpoint: Callable, **kwargs):
        path = f"{self.prefix}/{path}" if self.prefix else path
        if self.credentials:
            return self.app.add_api_route(path, endpoint, dependencies=[Depends(self.auth)], **kwargs)
        return self.app.add_api_route(path, endpoint, **kwargs)

    def decode_base64_image(self, base64_str: str) -> np.ndarray:
        try:
            img_data = base64.b64decode(base64_str, validate=True)
            img = Image.open(BytesIO(img_data)).convert("RGB")
            return np.array(img)
        except base64.binascii.Error:
            raise HTTPException(400, "Invalid Base64 string format")
        except Exception as e:
            raise HTTPException(400, f"Failed to decode image: {str(e)}")

    def encode_video_to_base64(self, video_path: str) -> str:
        try:
            with open(video_path, "rb") as f:
                video_data = f.read()
            return base64.b64encode(video_data).decode("utf-8")
        except Exception as e:
            raise HTTPException(500, f"Failed to encode video to Base64: {str(e)}")

    def extract_lora_from_prompt(self, prompt: str) -> tuple[str, list[str], list[float]]:
        """从prompt中提取<lora:模型名:权重>语法，返回清理后的prompt和LoRA信息"""
        lora_pattern = r"<lora:([^:>]+):([^>]+)>"
        lora_names = []
        lora_weights = []
        clean_prompt = prompt

        matches = re.findall(lora_pattern, prompt)
        for lora_name, weight in matches:
            try:
                weight = float(weight)
                if weight < 0 or weight > 2:
                    raise ValueError(f"LoRA weight {weight} must be between 0 and 2")
                lora_names.append(lora_name)
                lora_weights.append(weight)
                clean_prompt = clean_prompt.replace(f"<lora:{lora_name}:{weight}>", "")
            except ValueError as e:
                raise HTTPException(400, f"Invalid LoRA weight for {lora_name}: {str(e)}")

        # 验证 LoRA 文件是否存在
        lora_dir = "models/hunyuan/lora"
        for lora_name in lora_names:
            lora_file = None
            for ext in [".safetensors", ".pt"]:
                if os.path.exists(os.path.join(lora_dir, lora_name + ext)):
                    lora_file = lora_name + ext
                    break
            if not lora_file:
                raise HTTPException(400, f"LoRA model {lora_name} not found in {lora_dir}")

        return clean_prompt.strip(), lora_names, lora_weights

    async def endpoint_image2video(self, req: Image2VideoRequest):
        logger.info(f"Received request: prompt={req.prompt}, seed={req.seed}")
        try:
            with self.queue_lock:
                input_image = self.decode_base64_image(req.input_image) if req.input_image else None
                end_image = self.decode_base64_image(req.end_image) if req.end_image else None

                # 提取 LoRA 信息
                clean_prompt, lora_names, lora_weights = self.extract_lora_from_prompt(req.prompt)

                # 创建 AsyncStream 用于处理 worker 的流式输出
                stream = AsyncStream()
                task = async_run(
                    worker,
                    input_image=input_image.copy() if input_image is not None else None,
                    end_image=end_image,
                    prompt_text=clean_prompt,
                    n_prompt=req.negative_prompt,
                    seed=req.seed,
                    total_second_length=req.total_second_length,
                    latent_window_size=req.latent_window_size,
                    steps=req.steps,
                    cfg=req.cfg,
                    gs=req.distilled_guidance_scale,
                    rs=req.guidance_rescale,
                    gpu_memory_preservation=req.gpu_memory_preservation,
                    use_teacache=req.use_teacache,
                    mp4_crf=req.mp4_crf,
                    resolution=req.resolution,
                    save_metadata=req.save_metadata,
                    blend_sections=req.blend_sections,
                    latent_type=req.latent_type,
                    clean_up_videos=req.clean_up_videos,
                    selected_loras=lora_names,
                    lora_values=lora_weights,
                    job_stream=stream
                )

                video_path = None
                while True:
                    try:
                        flag, data = stream.output_queue.next()
                        if flag == 'file':
                            video_path = data
                        elif flag == 'end':
                            break
                    except IndexError:
                        await asyncio.sleep(0.1)
                        continue

                await task  # 确保任务完成

                if video_path is None:
                    raise HTTPException(500, "Failed to generate video: No video path returned")

                video_base64 = self.encode_video_to_base64(video_path)
                logger.info(f"Video generated successfully: {video_path}")
                return VideoResponse(video=video_base64, info="Video generated successfully")
        except ValueError as ve:
            logger.error(f"Invalid input: {str(ve)}")
            raise HTTPException(400, f"Invalid input: {str(ve)}")
        except RuntimeError as re:
            logger.error(f"Runtime error, possibly GPU memory issue: {str(re)}")
            raise HTTPException(500, f"Runtime error, possibly GPU memory issue: {str(re)}")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise HTTPException(500, f"Failed to generate video: {str(e)}")

def on_app_started(_: None, app: FastAPI):
    queue_lock = webui_queue_lock if IN_WEBUI else Lock()
    # 初始化模型（模拟 process 函数的模型加载逻辑）
    async def init_models():
        async for _ in process(
            input_image=None,
            end_image=None,
            latent_type="Black",
            prompt_text="",
            n_prompt="",
            seed=-1,
            total_second_length=5.0,
            latent_window_size=9,
            steps=25,
            cfg=1.0,
            gs=10.0,
            rs=0.0,
            gpu_memory_preservation=6.0,
            use_teacache=True,
            mp4_crf=16,
            resolution=640,
            save_metadata=False,
            blend_sections=4,
            clean_up_videos=True,
            selected_loras=[],
            lora_values=[],
            randomize_seed=False
        ):
            break  # 只需要触发模型加载，不需要处理输出
    asyncio.run(init_models())
    Api(app, queue_lock, "/framepack/v1")
