import requests
import base64
import json
from io import BytesIO
from PIL import Image
import logging
import os

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def encode_image_to_base64(image_path: str) -> str:
    """
    将图像文件编码为 Base64 字符串。
    
    Args:
        image_path (str): 图像文件的路径。
    
    Returns:
        str: Base64 编码的字符串。
    
    Raises:
        FileNotFoundError: 如果图像文件不存在。
        Exception: 如果图像编码失败。
    """
    try:
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
        return base64.b64encode(image_data).decode("utf-8")
    except FileNotFoundError:
        logger.error(f"Image file not found: {image_path}")
        raise
    except Exception as e:
        logger.error(f"Failed to encode image {image_path}: {str(e)}")
        raise

def save_video_from_base64(base64_video: str, output_path: str) -> None:
    """
    将 Base64 编码的视频数据保存为 MP4 文件。
    
    Args:
        base64_video (str): Base64 编码的视频数据。
        output_path (str): 保存视频的路径。
    
    Raises:
        Exception: 如果保存视频失败。
    """
    try:
        video_data = base64.b64decode(base64_video)
        with open(output_path, "wb") as video_file:
            video_file.write(video_data)
        logger.info(f"Video saved successfully to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save video to {output_path}: {str(e)}")
        raise

def call_image2video_api(
    api_url: str,
    prompt: str,
    input_image_path: str = None,
    end_image_path: str = None,
    negative_prompt: str = "",
    seed: int = -1,
    total_second_length: float = 5.0,
    latent_window_size: int = 9,
    steps: int = 25,
    cfg: float = 1.0,
    distilled_guidance_scale: float = 10.0,
    guidance_rescale: float = 0.0,
    gpu_memory_preservation: float = 6.0,
    use_teacache: bool = True,
    mp4_crf: int = 16,
    resolution: int = 640,
    save_metadata: bool = False,
    blend_sections: int = 4,
    clean_up_videos: bool = True,
    latent_type: str = "Black",
    username: str = None,
    password: str = None
) -> dict:
    """
    调用 Image-to-Video API 生成视频。
    
    Args:
        api_url (str): API 端点 URL，例如 "http://localhost:8000/framepack/v1/i2v"。
        prompt (str): 提示词，支持时间戳和 LoRA 语法，例如 "A sunset <lora:sunset_lora:1.2> [0s-2s: calm]".
        input_image_path (str, optional): 首帧图像路径。
        end_image_path (str, optional): 尾帧图像路径。
        negative_prompt (str): 负向提示词。
        seed (int): 随机种子，-1 表示随机。
        total_second_length (float): 视频总时长（秒）。
        latent_window_size (int): 潜在窗口大小。
        steps (int): 推理步数。
        cfg (float): CFG Scale。
        distilled_guidance_scale (float): Distilled CFG Scale。
        guidance_rescale (float): CFG Re-Scale。
        gpu_memory_preservation (float): GPU 推理保留内存（GB）。
        use_teacache (bool): 是否使用 TeaCache。
        mp4_crf (int): MP4 压缩率，0 为无压缩。
        resolution (int): 视频分辨率。
        save_metadata (bool): 是否保存元数据。
        blend_sections (int): 提示词混合段数。
        clean_up_videos (bool): 是否清理中间视频文件。
        latent_type (str): 无首帧时的潜在图像类型（Black, White, Noise, Green Screen）。
        username (str, optional): HTTP 基本认证的用户名。
        password (str, optional): HTTP 基本认证的密码。
    
    Returns:
        dict: API 响应，包含生成的视频（Base64 编码）和信息。
    
    Raises:
        HTTPException: 如果 API 请求失败。
        Exception: 如果处理请求或响应时发生错误。
    """
    # 准备请求数据
    request_data = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "total_second_length": total_second_length,
        "latent_window_size": latent_window_size,
        "steps": steps,
        "cfg": cfg,
        "distilled_guidance_scale": distilled_guidance_scale,
        "guidance_rescale": guidance_rescale,
        "gpu_memory_preservation": gpu_memory_preservation,
        "use_teacache": use_teacache,
        "mp4_crf": mp4_crf,
        "resolution": resolution,
        "save_metadata": save_metadata,
        "blend_sections": blend_sections,
        "clean_up_videos": clean_up_videos,
        "latent_type": latent_type,
        "input_image": None,
        "end_image": None
    }

    # 处理首帧图像
    if input_image_path:
        try:
            request_data["input_image"] = encode_image_to_base64(input_image_path)
            logger.info(f"Encoded input image: {input_image_path}")
        except Exception as e:
            logger.error(f"Failed to encode input image: {str(e)}")
            raise

    # 处理尾帧图像
    if end_image_path:
        try:
            request_data["end_image"] = encode_image_to_base64(end_image_path)
            logger.info(f"Encoded end image: {end_image_path}")
        except Exception as e:
            logger.error(f"Failed to encode end image: {str(e)}")
            raise

    # 准备认证（如果需要）
    auth = None
    if username and password:
        auth = (username, password)
        logger.info("Using HTTP basic authentication")

    # 发送 API 请求
    try:
        logger.info(f"Sending POST request to {api_url}")
        response = requests.post(
            api_url,
            json=request_data,
            auth=auth,
            timeout=None
        )
        response.raise_for_status()  # 抛出 HTTP 错误（如果有）
        logger.info("API request successful")
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error: {str(e)}")
        raise
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {str(e)}")
        raise

    # 解析响应
    try:
        response_data = response.json()
        if "video" not in response_data or "info" not in response_data:
            logger.error("Invalid response format: missing 'video' or 'info'")
            raise ValueError("Invalid response format")
        return response_data
    except ValueError as e:
        logger.error(f"Failed to parse response: {str(e)}")
        raise

def main():
    """
    主函数，演示如何调用 Image-to-Video API 并保存生成的视频。
    """
    # API 配置
    api_url = "http://127.0.0.1:7871/framepack/v1/i2v"  # 替换为你的 API 地址
    username = None  # 如果需要认证，设置用户名
    password = None  # 如果需要认证，设置密码
    output_dir = "outputs"  # 输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 请求参数
    prompt = (
        "A beautiful sunset over the ocean <lora:sunset_lora:1.2> "
        "[0s-2s: calm waves] [2s-5s: vibrant colors] <lora:vivid_lora:0.8>"
    )
    input_image_path = "input_image.png"  # 可选，替换为实际图像路径
    end_image_path = None  # 可选，替换为实际图像路径
    negative_prompt = "blurry, low quality"
    seed = 42
    total_second_length = 5.0
    latent_window_size = 9
    steps = 25
    cfg = 1.0
    distilled_guidance_scale = 10.0
    guidance_rescale = 0.0
    gpu_memory_preservation = 6.0
    use_teacache = True
    mp4_crf = 16
    resolution = 640
    save_metadata = False
    blend_sections = 4
    clean_up_videos = True
    latent_type = "Black"

    try:
        # 调用 API
        response = call_image2video_api(
            api_url=api_url,
            prompt=prompt,
            input_image_path=input_image_path,
            end_image_path=end_image_path,
            negative_prompt=negative_prompt,
            seed=seed,
            total_second_length=total_second_length,
            latent_window_size=latent_window_size,
            steps=steps,
            cfg=cfg,
            distilled_guidance_scale=distilled_guidance_scale,
            guidance_rescale=guidance_rescale,
            gpu_memory_preservation=gpu_memory_preservation,
            use_teacache=use_teacache,
            mp4_crf=mp4_crf,
            resolution=resolution,
            save_metadata=save_metadata,
            blend_sections=blend_sections,
            clean_up_videos=clean_up_videos,
            latent_type=latent_type,
            username=username,
            password=password
        )

        # 处理响应
        if response["video"]:
            output_path = os.path.join(output_dir, f"generated_video_{seed}.mp4")
            save_video_from_base64(response["video"], output_path)
            logger.info(f"Video generation successful: {response['info']}")
        else:
            logger.warning(f"No video generated: {response['info']}")

    except Exception as e:
        logger.error(f"Failed to generate video: {str(e)}")
        raise

if __name__ == "__main__":
    main()
