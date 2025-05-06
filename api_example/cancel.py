import requests
import logging
import json

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API 配置
CANCEL_URL = "http://127.0.0.1:7871/framepack/v1/cancel"
USERNAME = "user"  # 替换为你的用户名，如果未启用认证可留空
PASSWORD = "password"  # 替换为你的密码，如果未启用认证可留空

def cancel_task():
    """调用 /cancel 端点取消视频生成任务"""
    # 设置认证（如果需要）
    auth = (USERNAME, PASSWORD) if USERNAME and PASSWORD else None

    try:
        logger.info("Sending cancel request...")
        response = requests.post(CANCEL_URL, auth=auth, timeout=None)

        # 检查响应状态
        response.raise_for_status()
        result = response.json()

        # 处理响应
        logger.info(f"Cancel request succeeded: {result['info']}")
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
    cancel_task()
