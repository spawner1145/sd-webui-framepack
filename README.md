# sd-webui-framepack
两种运行方法

1.直接运行，在根目录下运行
```
python -m scripts.demo_gradio
```
2.扔进webui的extensions文件夹里，模型将会自动下载在`models/hunyuan`

## 重要：关于环境冲突
根据测试，冲突的环境有这些：
```
Installed version of accelerate (0.31.0) does not satisfy the requirement (==1.6.0).
Installing accelerate==1.6.0
Installed version of diffusers (0.31.0) does not satisfy the requirement (>=0.33.1).
Installing diffusers from source
Installed version of transformers (4.48.1) does not satisfy the requirement (==4.46.2).
Installing transformers==4.46.2
Installed version of pillow (10.4.0) does not satisfy the requirement (==11.1.0).
Installing pillow==11.1.0
Installed version of numpy (1.26.4) does not satisfy the requirement (==1.26.2).
Installing numpy==1.26.2
```
解决方法是去你的webui根目录下，找到requirements_versions.txt，把里面相关的冲突依赖删掉，有效避免每次启动都安装依赖

顺便有些插件，例如supermerger，你可以启动过一次以后直接把插件目录的install.py删了
