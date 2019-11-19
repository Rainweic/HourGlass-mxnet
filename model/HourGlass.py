import mxnet as mx
from mxboard import SummaryWriter
from mxnet.gluon import nn

class Residual(nn.HybridBlock):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.residual_conv = nn.HybridSequential()
        self.residual_skip = nn.HybridSequential()

        with self.residual_conv.name_scope():
            # 卷积路
            self.residual_conv.add(nn.BatchNorm())
            self.residual_conv.add(nn.Activation('relu'))
            self.residual_conv.add(nn.Conv2D(self.out_channels // 2, (1, 1)))
            self.residual_conv.add(nn.BatchNorm())
            self.residual_conv.add(nn.Activation('relu'))
            self.residual_conv.add(nn.Conv2D(self.out_channels // 2, (3, 3), (1, 1), (1, 1)))
            self.residual_conv.add(nn.BatchNorm())
            self.residual_conv.add(nn.Activation('relu'))
            self.residual_conv.add(nn.Conv2D(self.out_channels, (1, 1)))

        # 连接路
        if not self.in_channels == self.out_channels:
            with self.residual_skip.name_scope():
                self.residual_skip.add(nn.Conv2D(self.out_channels, (1, 1)))
        
    def hybrid_forward(self, F, in_data):    
        x = in_data
        x = self.residual_conv(x)
        if not self.in_channels == self.out_channels:
            x = x + self.residual_skip(in_data)
        return x

class HourGlassBlock(nn.HybridBlock):

    def __init__(self, n, in_channels, **kwargs):
        '''
        args:
            n:              当前HourGlass所在的阶数
            in_channels:    当前HourGlass输入的channels
        '''
        super(HourGlassBlock, self).__init__(**kwargs)
        self.n = n
        self.in_channels = in_channels

        with self.name_scope():
            # Upper branch
            self.up1 = Residual(self.in_channels, self.in_channels)

            # Lower branch
            self.low1_MaxPool = nn.MaxPool2D((2, 2), (2, 2))
            self.low1 = Residual(self.in_channels, self.in_channels)

            if self.n > 1:
                self.low2 = HourGlassBlock(self.n - 1, self.in_channels)
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


class Lin(nn.HybridBlock):

    def __init__(self, numOut, **kwargs):
        super(Lin, self).__init__(**kwargs)
        self.numOut = numOut
        self.lin = nn.HybridSequential()

        with self.lin.name_scope():
            self.lin.add(nn.Conv2D(numOut, 1))
            self.lin.add(nn.BatchNorm())
            self.lin.add(nn.Activation('relu'))

    def hybrid_forward(self, F, x):
        return self.lin(x)


class Hourglass(nn.HybridBlock):

    def __init__(self, nStack=8, **kwargs):
        super(Hourglass, self).__init__(**kwargs)
        self.nStack = nStack

        # HourglassBlock模块之前的图片处理模块
        self.preprocess = nn.HybridSequential()
        with self.preprocess.name_scope():
            self.preprocess.add(nn.Conv2D(64, 7, (2, 2), (3, 3)))
            self.preprocess.add(nn.BatchNorm())
            self.preprocess.add(nn.Activation("relu"))
            self.preprocess.add(Residual(64, 128))
            self.preprocess.add(nn.MaxPool2D((2, 2), (2, 2)))
            self.preprocess.add(Residual(128, 128))
            self.preprocess.add(Residual(128, 256))

        
        self.out = nn.HybridSequential()
        # HourglassBlock模块
        self.hourglass_block = nn.HybridSequential()
        with self.hourglass_block.name_scope():
            for i in range(self.nStack):
                self.hourglass_block.add(HourGlassBlock(4, 256))
                self.hourglass_block.add(Residual(256, 256))
                self.hourglass_block.add(Lin(256))
                self.conv_temp_out = nn.Conv2D(16, (1, 1), (1, 1))
                self.hourglass_block.add(self.conv_temp_out)
                # TODO 添加conv_temp_out到self.out
                
                if i < self.nStack:
                    self.hourglass_block.add(nn.Conv2D(256, (1, 1), (1, 1)))
                    # TODO 这里还有一个操作

    def hybrid_forward(self, F, x):
        x = self.preprocess(x)
        x = self.hourglass_block(x)
        # TODO return 的不是 x 肯定还有其他操作
        return x

if __name__ == "__main__":
    model = Hourglass(nStack=1)
    model.initialize()
    model.hybridize()
    print(model)

    in_data = mx.nd.random.uniform(-1, 1, shape=[1,3,256,256])
    out = model(in_data)
    sw = SummaryWriter(logdir='./logs', flush_secs=5)
    sw.add_graph(model)
    sw.close()