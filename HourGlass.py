import mxnet as mx
from mxnet.gluon import nn

class Residual(nn.HybridBlock):
    def __init__(self, in_channels, out_channels, use_bn=True, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = self.in_channels // 2
        self.use_bn = use_bn

        if self.use_bn:
            self.bn = nn.BatchNorm()

        with self.name_scope():
            self.conv1 = nn.Conv2D(self.mid_channels, 1)
            self.conv2 = nn.Conv2D(self.mid_channels, 3)
            self.conv3 = nn.Conv2D(self.out_channels, 1)
            self.conv4 = nn.Conv2D(self.out_channels, 1, in_channels=self.in_channels)
        
    def hybrid_forward(self, F, x):    
        # 上行支路三个连续的Conv
        if self.use_bn:
            x = self.bn(x)
        x = F.relu(x)
        x = self.conv1(x)
        if self.use_bn:
            x = self.bn(x)
        x = F.relu(x)
        x = self.conv2(x)
        if self.use_bn:
            x = self.bn(x)
        x = F.relu(x)
        x = self.conv3(x)

        # 加上下行支路的Conv
        x = x + self.conv4(x)

        return x


class HourGlass(nn.HybridBlock):
    def __init__(self, n, in_channels, input_model, **kwargs):
        '''
        args:
            n:              当前HourGlass所在的阶数
            in_channels:    当前HourGlass输入的channels
            input_model:    连接当前HourGlass的上一级模块
        '''
        super(HourGlass, self).__init__(**kwargs)
        self.n = n
        self.in_channels = in_channels
        self.input_model = input_model

        # Upper branch
        self.up1 = Residual(self.in_channels, self.in_channels)

        # Lower branch
        self.low1_MaxPool = nn.MaxPool2D((2, 2), (2, 2))
        self.low1 = Residual(self.in_channels, self.in_channels)

        if self.n > 1:
            self.low2 = HourGlass(self.n - 1, self.in_channels, self.low1)
        else:
            self.low2 = Residual(self.in_channels, self.in_channels)

        self.low3 = Residual(self.in_channels, self.in_channels)

    def hybrid_forward(self, F, x):
        up1 = self.up1(x)

        x = self.low1_MaxPool(x)
        x = self.low1(x)
        x = self.low2(x)
        x = self.low3(x)

        up2 = F.UpSampling(x, scale=2, sample_type="nearest")

        return up1 + up2


if __name__ == "__main__":
    input = mx.nd.random.uniform(-1, 1, shape=(1,3,256,256))

    conv = nn.Conv2D(64, 3)
    model = HourGlass(4, 64, conv)
    model.initialize()

    model(input)


