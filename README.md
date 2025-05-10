Okay, here is the complete bilingual version of the README again:

# sd-webui-framepack (中英双语版 / Chinese & English Version)

已支持30系以下显卡和首尾帧
已支持api调用
已支持lora和文生视频

Support for graphics cards below the 30-series and for first/last frames has been added.
API calls are now supported.
Lora and text-to-video generation are now supported.

## 两种运行方法：
## Two Ways to Run:

### 1.直接运行，在根目录下运行
### 1. Direct Execution: Run in the root directory

```
python -m scripts.app
```

或者双击 `单独启动.bat`
Or double-click `单独启动.bat` (Standalone Startup.bat)

#### 注意，直接运行不要运行 `install.py` ，而是先装 `torch`和 `torchvision` ，然后 `pip install -r requirements.txt` 把必要依赖装好，然后建议 `xformers` (配套装triton，windows安装方法上网查)，`flash attention` 或者 `sage attention` 装上，否则会很慢
#### Note: For direct execution, do not run `install.py`. Instead, first install `torch` and `torchvision`. Then, run `pip install -r requirements.txt` to install the necessary dependencies. It is also recommended to install `xformers` (along with triton; search online for Windows installation methods), `flash attention`, or `sage attention`, otherwise it will be very slow.

### 2.扔进webui的extensions文件夹里，模型将会自动下载在 `models/hunyuan`,如果你有lora，放到 `models/hunyuan/lora` 文件夹内
### 2. Place it in the webui's extensions folder. The model will be automatically downloaded to `models/hunyuan`. If you have lora models, place them in the `models/hunyuan/lora` folder.

#### 重要：关于作为插件时的环境冲突
#### Important: Regarding environment conflicts when running as an extension

根据测试，冲突的环境有这些：
According to tests, these are the conflicting environments:

```
Installed version of accelerate (0.31.0) does not satisfy the requirement (==1.6.0).
Installing accelerate==1.6.0
Installed version of diffusers (0.31.0) does not satisfy the requirement (>=0.33.1).
Installing diffusers from source
Installed version of pillow (10.4.0) does not satisfy the requirement (==11.1.0).
Installing pillow==11.1.0
```

解决方法是去你的webui根目录下，找到requirements_versions.txt，把里面相关的冲突依赖删掉，有效避免每次启动都安装依赖
The solution is to go to your webui root directory, find `requirements_versions.txt`, and delete the conflicting dependencies listed within. This will effectively prevent the installation of dependencies every time you start.

不过如果要省事你可以在启动成功后加上启动参数 `--skip-prepare-environment` ，这样下次启动就不用依赖安装浪费一堆时间了
However, if you want to save trouble, you can add the launch parameter `--skip-prepare-environment` after a successful startup. This way, you won't waste a lot of time on dependency installation the next time you start.
