import logging
import threading
from threading import Lock
from fastapi import FastAPI
from backend_fp.inferrence import *
from backend_fp.ui import *
from backend_fp.api import Api
import uvicorn

logging.basicConfig(level=logging.INFO)

try:
    from modules import script_callbacks, shared
    IN_WEBUI = True
except ImportError:
    IN_WEBUI = False
    shared = type('Shared', (), {'opts': type('Opts', (), {
        'outdir_samples': '',
        'outdir_txt2img_samples': '',
        'outdir_img2img_samples': ''
    })})()

HOST = "127.0.0.1"
PORT_GRADIO = 7870
PORT_API = 7871
SHARE = True

if IN_WEBUI:
    from backend_fp.api import on_app_started
    def on_ui_tabs():
        block = create_ui()
        return [(block, "FramePack Studio", "framepack_tab")]
    script_callbacks.on_ui_tabs(on_ui_tabs)
    script_callbacks.on_app_started(on_app_started)
else:
    if __name__ == "__main__":
        block = create_ui()
        logging.info("Gradio 界面已创建")

        app = FastAPI(docs_url="/docs", openapi_url="/openapi.json")
        queue_lock = Lock()
        api = Api(app, queue_lock, prefix="/framepack/v1")
        logging.info("API 路由已挂载到独立的 FastAPI 实例")

        print(f"API 文档可用：http://{HOST}:{PORT_API}/docs")
        def run_gradio():
            try:
                block.launch(
                    server_name=HOST,
                    server_port=PORT_GRADIO,
                    share=SHARE,
                    prevent_thread_lock=True
                )
            except Exception as e:
                logging.error(f"Gradio 启动失败: {str(e)}")

        gradio_thread = threading.Thread(target=run_gradio)
        gradio_thread.start()

        try:
            uvicorn.run(
                app,
                host=HOST,
                port=PORT_API,
                log_level="info"
            )
        except Exception as e:
            logging.error(f"FastAPI 启动失败: {str(e)}")
        finally:
            block.close()
            gradio_thread.join()
