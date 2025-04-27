import base64
import requests
from io import BytesIO
from PIL import Image
import os
import json

def image_to_base64(image_path):
    """将图片文件转换为Base64编码字符串"""
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_base64
    except Exception as e:
        raise Exception(f"Failed to encode image {image_path}: {str(e)}")

def call_framepack_api(
    input_image_path,
    end_image_path=None,
    prompt="A character doing some simple body movements.",
    negative_prompt="",
    seed=-1,
    total_second_length=5.0,
    latent_window_size=9,
    steps=25,
    cfg=1.0,
    distilled_guidance_scale=10.0,
    guidance_rescale=0.0,
    gpu_memory_preservation=6.0,
    use_teacache=True,
    mp4_crf=16,
    resolution=640,
    if_save=False,
    api_url="http://127.0.0.1:7871/framepack/v1/i2v",
    auth=None
):
    """
    调用FramePack API生成视频
    参数：
        input_image_path: 首帧图片路径
        end_image_path: 尾帧图片路径（可选）
        prompt: 正向提示词
        negative_prompt: 负向提示词
        seed: 随机种子
        total_second_length: 视频时长（秒）
        latent_window_size: 潜在窗口大小
        steps: 推理步数
        cfg: CFG Scale
        distilled_guidance_scale: Distilled CFG Scale
        guidance_rescale: CFG Re-Scale
        gpu_memory_preservation: GPU保留内存（GB）
        use_teacache: 是否使用TeaCache
        mp4_crf: MP4压缩率
        resolution: 视频分辨率
        if_save: 是否保存视频到服务器的outputs文件夹
        api_url: API端点URL
        auth: 元组 (username, password)，用于HTTP基本认证（可选）
    返回：
        video_path: 本地保存的视频路径（如果保存）
        info: API返回的生成信息
    """
    # 编码输入图片
    input_image_base64 = image_to_base64(input_image_path)
    end_image_base64 = image_to_base64(end_image_path) if end_image_path else None

    # 构造请求数据
    request_data = {
        "input_image": input_image_base64,
        "end_image": end_image_base64,
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
        "if_save": if_save
    }

    # 发送POST请求
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(api_url, json=request_data, headers=headers, auth=auth, timeout=None)
        response.raise_for_status()  # 抛出HTTP错误
    except requests.exceptions.RequestException as e:
        raise Exception(f"API request failed: {str(e)}")

    # 解析响应
    response_data = response.json()
    video_base64 = response_data.get("video")
    info = response_data.get("info")

    if not video_base64:
        raise Exception("No video returned in response")

    # 解码视频并保存到本地（可选）
    video_path = None
    if video_base64:
        video_data = base64.b64decode(video_base64)
        output_dir = "output_videos"
        os.makedirs(output_dir, exist_ok=True)
        video_path = os.path.join(output_dir, f"framepack_video_{int(os.times().elapsed)}.mp4")
        with open(video_path, "wb") as f:
            f.write(video_data)
        print(f"Video saved to {video_path}")

    return video_path, info

if __name__ == "__main__":
    # 示例用法
    input_image = "00105-1649913665.png"  # 替换为实际首帧图片路径
    end_image = None  # 替换为实际尾帧图片路径（可选）
    
    try:
        print("Starting video generation...")
        video_path, info = call_framepack_api(
            input_image_path=input_image,
            end_image_path=end_image,
            prompt="A girl dances gracefully, with clear movements, full of charm.",
            if_save=True,  # 将视频保存到服务器的outputs文件夹
            auth=None  # 如果API需要认证，设为 (username,password)
        )
        print(f"Generated video: {video_path}")
        print(f"Generation info: {info}")
    except Exception as e:
        print(f"Error: {str(e)}")