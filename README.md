# HourGlass-mxnet
### 使用gluon复现Stacked Hourglass姿态检测网络

    文件夹结构说明:
        logs        ---   训练日志 可用tensorboard查看
        model       ---   gluon制作的模型
        tools       ---   工具类
        train_data  ---   存放训练数据


## 环境依赖

    1. mxnet
    

## 报错提示

    1. module 'scipy.misc' has no attribute 'imresize'
        安装 scipy==1.2.1 Pillow==6.0.0