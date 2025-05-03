import base64
import os
from typing import Callable
from threading import Lock
from secrets import compare_digest
from io import BytesIO

from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field
from PIL import Image
import numpy as np

try:
    from modules import shared
    from modules.call_queue import queue_lock as webui_queue_lock
    IN_WEBUI = True
except ImportError:
    IN_WEBUI = False
    shared = type('Shared', (), {'cmd_opts': type('CmdOpts', (), {'api_auth': None})()})()
    webui_queue_lock = None

from backend_fp.inferrence import worker

class Image2VideoRequest(BaseModel):
    input_image: str | None = Field(None, description="首帧图片的 Base64 编码字符串（可选）")
    end_image: str | None = Field(None, description="尾帧图片的 Base64 编码字符串（可选）")
    latent_type: str = Field("Black", description="无首帧时使用的潜在图像类型：Black, White, Noise, Green Screen")
    prompt: str = Field("", description="提示词，支持时间戳格式如[0s-2s: 描述]")
    negative_prompt: str = Field("", description="负向提示词")
    seed: int = Field(-1, description="随机种子")
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
    lora_names: list[str] = Field([], description="使用的LoRA模型名称")
    lora_weights: list[float] = Field([], description="LoRA模型权重")

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
            img_data = base64.b64decode(base64_str)
            img = Image.open(BytesIO(img_data)).convert("RGB")
            return np.array(img)
        except Exception as e:
            raise HTTPException(400, f"Invalid image Base64 data: {str(e)}")

    def encode_video_to_base64(self, video_path: str) -> str:
        try:
            with open(video_path, "rb") as f:
                video_data = f.read()
            return base64.b64encode(video_data).decode("utf-8")
        except Exception as e:
            raise HTTPException(500, f"Failed to encode video to Base64: {str(e)}")

    def endpoint_image2video(self, req: Image2VideoRequest):
        input_image = self.decode_base64_image(req.input_image) if req.input_image else None
        end_image = self.decode_base64_image(req.end_image) if req.end_image else None
        try:
            video_path = worker(
                input_image=input_image,
                end_image=end_image,
                latent_type=req.latent_type,
                prompt_text=req.prompt,
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
                clean_up_videos=req.clean_up_videos,
                selected_loras=req.lora_names,
                lora_values=req.lora_weights
            )
            video_base64 = self.encode_video_to_base64(video_path) if video_path else None
            return VideoResponse(video=video_base64, info="Video generated successfully")
        except Exception as e:
            raise HTTPException(500, f"Failed to generate video: {str(e)}")

def on_app_started(_: None, app: FastAPI):
    queue_lock = webui_queue_lock if IN_WEBUI else Lock()
    Api(app, queue_lock, "/framepack/v1")