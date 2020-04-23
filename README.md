# DANet_PyTorch
A Pytorch implementation of Dual Attention Network for Scene Segmentation

## 实验环境简介
- 环境: Python3.6, Pytorch1.0, OpenCV, Numpy等必备环境
- DANet_ResNet网络代码: danet.py, attention.py, danet_res152.py

## 实验数据介绍
- 一副无人机拍摄的高分辨率矿区影像图
- 实验室进行标注的对应label
- 进行裁剪后的320 x 320的图像与label数据

## 实验代码介绍
- [danet.py](https://github.com/yearing1017/DANet_PyTorch/blob/master/DAN_ResNet/danet.py): DANet网络代码
- [attention.py](https://github.com/yearing1017/DANet_PyTorch/blob/master/DAN_ResNet/attention.py): 注意力模块代码，pam和cam模块代码
- [danet_res152.py](https://github.com/yearing1017/DANet_PyTorch/blob/master/danet_res152.py): 基于resnet152的danet代码，替换aspp模块
- [MyData.py](https://github.com/yearing1017/DANet_PyTorch/blob/master/MyData.py): 数据载入的代码
- [train_danet_res.py](https://github.com/yearing1017/DANet_PyTorch/blob/master/train_danet_res.py): 训练代码
- [predict_gray.py](https://github.com/yearing1017/DANet_PyTorch/blob/master/predict_gray.py): 预测灰度结果的代码
- [MIoU.py](https://github.com/yearing1017/DANet_PyTorch/blob/master/MIoU.py): 根据灰度预测结果计算相关指标
- [predict.py](https://github.com/yearing1017/DANet_PyTorch/blob/master/predict.py): 预测结果并进行涂色

## 实验结果

### 实验数据指标

|     版本&指标    |  Acc   |  MIoU  | Kappa  |  地面  |  房屋  |  道路  |  车辆  |
| :-----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| Danet0408 | 0.9294 | 0.6762 | 0.7347 | 0.9818 | 0.5991 | 0.8409 | 0.4458 |


### 实验初步效果截图对比

- 基于ResNet152的DANet网络实验（danet替换deeplabv3的aspp模块）
![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/yearing1017/danet0408.jpg)
