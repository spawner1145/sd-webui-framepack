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
import tempfile
import numpy as np

try:
    from modules import shared
    from modules.call_queue import queue_lock as webui_queue_lock
    IN_WEBUI = True
except ImportError:
    IN_WEBUI = False
    shared = type('Shared', (), {'cmd_opts': type('CmdOpts', (), {'api_auth': None})()})()
    webui_queue_lock = None

from backend_fp.inferrence import process

class Image2VideoRequest(BaseModel):
    input_image: str = Field(..., description="首帧图片的 Base64 编码字符串")
    end_image: str | None = Field(None, description="尾帧图片的 Base64 编码字符串（可选）")
    prompt: str = Field("", description="正向提示词，描述视频内容")
    negative_prompt: str = Field("", description="负向提示词，用于排除不需要的元素")
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
    if_save: bool = Field(False, description="是否将视频保存到 outputs 文件夹")

class VideoResponse(BaseModel):
    video: str = Field(..., description="生成的视频，Base64 编码的 MP4 文件")
    info: str = Field(..., description="生成信息，包括耗时、分辨率等")

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
        """将 Base64 图片解码为 numpy 数组"""
        try:
            img_data = base64.b64decode(base64_str)
            img = Image.open(BytesIO(img_data)).convert("RGB")
            return np.array(img)
        except Exception as e:
            raise HTTPException(400, f"Invalid image Base64 data: {str(e)}")

    def encode_video_to_base64(self, video_path: str) -> str:
        """将视频文件编码为 Base64 字符串"""
        try:
            with open(video_path, "rb") as f:
                video_data = f.read()
            return base64.b64encode(video_data).decode("utf-8")
        except Exception as e:
            raise HTTPException(500, f"Failed to encode video to Base64: {str(e)}")

    def endpoint_image2video(self, req: Image2VideoRequest):
        input_image = self.decode_base64_image(req.input_image)
        end_image = self.decode_base64_image(req.end_image) if req.end_image else None
        with self.queue_lock:
            outputs = process(
                input_image=input_image,
                end_image=end_image,
                prompt=req.prompt,
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
                resolution=req.resolution
            )
            # process 返回的是一个生成器，获取最终输出
            output_path = None
            for output in outputs:
                if output[0] is not None:
                    output_path = output[0]
            if output_path is None:
                raise HTTPException(500, "Video generation failed")
            video_base64 = self.encode_video_to_base64(output_path)
            # 如果 if_save 为 True，则不删除输出文件
            if not req.if_save:
                os.remove(output_path)
            else:
                # 确保 outputs 文件夹存在
                outputs_folder = "outputs"
                os.makedirs(outputs_folder, exist_ok=True)
            info = f"Video generated with resolution {req.resolution}, duration {req.total_second_length}s, seed {req.seed}, saved: {req.if_save}"
            return VideoResponse(video=video_base64, info=info)
        
def on_app_started(_: None, app: FastAPI):
    # 在 WebUI 环境下使用 webui_queue_lock
    queue_lock = webui_queue_lock if IN_WEBUI else Lock()
    Api(app, queue_lock, "/framepack/v1")