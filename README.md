# DANet_PyTorch
A Pytorch implementation of Dual Attention Network for Scene Segmentation

## 实验环境简介
- 环境: Python3.6, Pytorch1.0, OpenCV, Numpy等必备环境
- DANet_ResNet网络代码: danet.py, attention.py, danet_res152.py

## 实验数据介绍
- 一副无人机拍摄的高分辨率矿区影像图
- 实验室进行标注的对应label
- v0219版本：进行裁剪后的640 x 640的图像与label数据
- v0225&v0301版本及之后：进行裁剪后的320 x 320的图像与label数据，并更换测试集

