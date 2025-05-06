import requests
import base64
import json
import os
from PIL import Image
from io import BytesIO
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API 配置
API_URL = "http://127.0.0.1:7871/framepack/v1/i2v"
USERNAME = "user"  # 替换为你的用户名，如果未启用认证可留空
PASSWORD = "password"  # 替换为你的密码，如果未启用认证可留空
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def encode_image_to_base64(image_path: str) -> str:
    """将图片编码为 Base64 字符串"""
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        logger.error(f"Failed to encode image {image_path}: {str(e)}")
        raise

def save_video_from_base64(base64_str: str, output_path: str):
    """将 Base64 编码的视频保存为 MP4 文件"""
    try:
        video_data = base64.b64decode(base64_str)
        with open(output_path, "wb") as f:
            f.write(video_data)
        logger.info(f"Video saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save video: {str(e)}")
        raise

def generate_video():
    """调用 /i2v 端点生成视频"""
    # 准备请求数据
    #input_image_path = "test_image.png"  # 替换为你的测试图片路径
    try:
        #input_image_base64 = encode_image_to_base64(input_image_path)
        pass
    except Exception as e:
        logger.error(f"Error encoding image: {str(e)}")
        return

    payload = {
        "input_image": None, #input_image_base64,
        "end_image": None,
        "latent_type": "Black",
        "prompt": "A serene lake with mountains in the background, sunset glow,<lora:sex_helper_0.4_pov.safetensors:1.0>",#可以使用 <lora:sex_helper_0.4_pov.safetensors:1.0>语法
        "negative_prompt": "blurry, low quality",
        "seed": -1,
        "total_second_length": 5.0,
        "latent_window_size": 9,
        "steps": 25,
        "cfg": 1.0,
        "distilled_guidance_scale": 10.0,
        "guidance_rescale": 0.0,
        "gpu_memory_preservation": 6.0,
        "use_teacache": True,
        "mp4_crf": 16,
        "resolution": 640,
        "save_metadata": True,
        "blend_sections": 4,
        "clean_up_videos": True
    }

    # 设置认证（如果需要）
    auth = (USERNAME, PASSWORD) if USERNAME and PASSWORD else None

    try:
        logger.info("Sending video generation request...")
        response = requests.post(API_URL, json=payload, auth=auth, timeout=None)

        # 检查响应状态
        response.raise_for_status()
        result = response.json()

        # 处理响应
        if "video" in result and result["video"]:
            output_path = os.path.join(OUTPUT_DIR, "generated_video.mp4")
            save_video_from_base64(result["video"], output_path)
            logger.info(f"Video generation succeeded: {result['info']}")
        else:
            logger.error(f"Video generation failed: {result.get('info', 'No video returned')}")
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error: {str(e)}")
        if response.text:
            try:
                error_detail = response.json().get("detail", str(e))
                logger.error(f"Error detail: {error_detail}")
            except json.JSONDecodeError:
                logger.error(f"Raw response: {response.text}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    generate_video()
