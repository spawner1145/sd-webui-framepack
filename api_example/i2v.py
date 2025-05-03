import requests
import base64
import json
from datetime import datetime

# API 端点 URL
API_URL = "http://127.0.0.1:7871/framepack/v1/i2v"

# 如果你的 API 配置了认证（在 api.py 中通过 shared.cmd_opts.api_auth 设置）
# 请提供用户名和密码；如果没有认证，可以将此部分注释掉
AUTH = ("username", "password")  # 替换为你的实际用户名和密码

# 准备请求数据
payload = {
    # 首帧图片（Base64 编码），这里设为 null 表示不提供首帧
    "input_image": None,
    
    # 尾帧图片（Base64 编码），可选，这里设为 null
    "end_image": None,
    
    # 无首帧时的潜在图像类型，可选 "Black", "White", "Noise", "Green Screen"
    "latent_type": "Black",
    
    # 提示词，支持时间戳格式，如 "[0s-2s: 描述]"
    "prompt": "[0s-2s: A girl dancing gracefully] [2s-5s: A girl spinning with joy]",
    
    # 负向提示词，用于排除不需要的元素
    "negative_prompt": "blurry, low quality",
    
    # 随机种子，-1 表示随机生成
    "seed": -1,
    
    # 视频总时长（秒），范围 1.0 到 120.0
    "total_second_length": 5.0,
    
    # 潜在窗口大小，影响视频生成的分段，范围 1 到 33
    "latent_window_size": 9,
    
    # 推理步数，影响生成质量，范围 1 到 100
    "steps": 25,
    
    # CFG Scale，控制提示词的遵循程度，范围 1.0 到 32.0
    "cfg": 7.0,
    
    # Distilled CFG Scale，控制蒸馏引导强度，范围 1.0 到 32.0
    "distilled_guidance_scale": 10.0,
    
    # CFG Re-Scale，调整引导尺度，范围 0.0 到 1.0
    "guidance_rescale": 0.0,
    
    # GPU 推理保留内存（GB），范围 0.0 到 128.0
    "gpu_memory_preservation": 6.0,
    
    # 是否使用 TeaCache 优化内存
    "use_teacache": True,
    
    # MP4 压缩率，0 表示无压缩，范围 0 到 100
    "mp4_crf": 16,
    
    # 视频分辨率，范围 240 到 720
    "resolution": 640,
    
    # 是否保存元数据（如提示词、种子等）
    "save_metadata": True,
    
    # 提示词混合的段数，范围 0 到 10
    "blend_sections": 4,
    
    # 是否清理中间视频文件
    "clean_up_videos": True,
    
    # 使用的 LoRA 模型名称列表
    "lora_names": ["lora_model_1.safetensors"],
    
    # LoRA 模型权重列表，与 lora_names 对应
    "lora_weights": [0.8]
}

def call_api():
    """
    调用 FramePack Studio 的 API 生成视频
    """
    try:
        # 发送 POST 请求
        # 如果没有认证，移除 auth 参数
        response = requests.post(
            API_URL,
            json=payload,
            auth=AUTH if AUTH else None,
            timeout=None
        )

        # 检查响应状态码
        if response.status_code == 200:
            # 解析 JSON 响应
            result = response.json()
            
            # 检查是否成功生成视频
            if result.get("video"):
                # 解码 Base64 编码的视频
                video_data = base64.b64decode(result["video"])
                
                # 生成输出文件名（基于时间戳）
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"output_video_{timestamp}.mp4"
                
                # 保存视频到文件
                with open(output_filename, "wb") as f:
                    f.write(video_data)
                
                print(f"视频已保存到 {output_filename}")
                print(f"生成信息: {result['info']}")
            else:
                print(f"生成失败: {result['info']}")
        else:
            # 处理 HTTP 错误
            print(f"请求失败，状态码: {response.status_code}")
            print(f"错误信息: {response.text}")
    
    except requests.exceptions.RequestException as e:
        print(f"请求发生错误: {e}")
    
    except ValueError as e:
        print(f"响应解析错误: {e}")

if __name__ == "__main__":
    print("开始调用 API...")
    call_api()