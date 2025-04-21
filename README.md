# sd-webui-framepack
已支持30系以下显卡和首尾帧

两种运行方法

1.直接运行，在根目录下运行
```
python -m scripts.demo_gradio
```
#### 注意，直接运行不要运行`install.py`，自己去`requirements.txt`把依赖装好，别忘了装`torch`和`torchvision`，然后建议`xformers`(配套装Triton，windows安装上网查)，`flash attention`或者`sage attention`装上，否则会很慢

2.扔进webui的extensions文件夹里，模型将会自动下载在`models/hunyuan`

## 重要：关于环境冲突
根据测试，冲突的环境有这些：
```
Installed version of accelerate (0.31.0) does not satisfy the requirement (==1.6.0).
Installing accelerate==1.6.0
Installed version of diffusers (0.31.0) does not satisfy the requirement (>=0.33.1).
Installing diffusers from source
Installed version of pillow (10.4.0) does not satisfy the requirement (==11.1.0).
Installing pillow==11.1.0
```
解决方法是去你的webui根目录下，找到requirements_versions.txt，把里面相关的冲突依赖删掉，有效避免每次启动都安装依赖

顺便有些插件，例如supermerger，你可以启动过一次以后直接把插件目录的install.py删了
