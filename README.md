# Stylized-Avator

## 请大家使用git/VScode/GitHub Desktop等相关软件进行上传

## 如果不清楚git怎么用可以看这个视频：[*Click here*](https://www.bilibili.com/video/BV1u94y1n73L/ "7分钟的简单介绍")

## 2024-10-14更新

程序主要流程如下：

1.加载UI界面
当 (用户没有选择退出)
    2.程序读取用户选取的image和输入的prompts
    3.将image和prompts传入SD模型
    4.SD模型根据image和prompts生成风格化的new_image
    5.将new_image传给程序
    6.程序输出生成是否成功(flag)，若成功则显示new_image
    7.程序允许用户选择保存图片或继续生成新的图片
8.退出程序

## 1.总体框架

主要分成三部分，算法设计，UI设计和美工。

## 2.算法

图片输入 图片处理 图片输出

处理：训练StableDiffution大模型来输出结果

## 3.UI

使用PyQt5模块和Qtdesigner开发

## 4.美工

制作答辩用PPT和海报
