import mindspore
import mindspore.nn as nn
from mindspore.common.initializer import TruncatedNormal
from mindspore.ops import operations as P
from mindspore.nn.loss.loss import _Loss
import mindspore.ops as ops
from mindspore import Tensor
from mindspore import dtype as mstype
def weight_variable():
    """Weight variable."""
    return TruncatedNormal(0.02)


class Conv2dBlock(nn.Cell):
    """
     Basic convolutional block
     Args:
         in_channles (int): Input channel.
         out_channels (int): Output channel.
         kernel_size (int): Input kernel size. Default: 1
         stride (int): Stride size for the first convolutional layer. Default: 1.
         padding (int): Implicit paddings on both sides of the input. Default: 0.
         pad_mode (str): Padding mode. Optional values are "same", "valid", "pad". Default: "same".
      Returns:
          Tensor, output tensor.
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, pad_mode="same"):
        super(Conv2dBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, pad_mode=pad_mode, weight_init=weight_variable())
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Inception(nn.Cell):
    """
    Inception Block
    """

    def __init__(self, in_channels, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()
        self.b1 = Conv2dBlock(in_channels, n1x1, kernel_size=1)
        self.b2 = nn.SequentialCell([Conv2dBlock(in_channels, n3x3red, kernel_size=1),
                                     Conv2dBlock(n3x3red, n3x3, kernel_size=3, padding=0)])
        self.b3 = nn.SequentialCell([Conv2dBlock(in_channels, n5x5red, kernel_size=1),
                                     Conv2dBlock(n5x5red, n5x5, kernel_size=5, padding=0)])
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, pad_mode="same")
        self.b4 = Conv2dBlock(in_channels, pool_planes, kernel_size=1)
        self.concat = P.Concat(axis=1)

    def construct(self, x):
        branch1 = self.b1(x)
        branch2 = self.b2(x)
        branch3 = self.b3(x)
        cell = self.maxpool(x)
        branch4 = self.b4(cell)
        return self.concat((branch1, branch2, branch3, branch4))

class InceptionAux(nn.Cell):
    #辅助分类器
    def __init__(self,in_channels,num_classes):
        super(InceptionAux, self).__init__()
        #self.averagePool=nn.AvgPool2d(kernel_size=3,stride=2,pad_mode='same')#平均池化
        self.conv=Conv2dBlock(in_channels,128,kernel_size=1)#output=[batch,128,2,2]
        self.fc1=nn.Dense(512,256)
        self.fc2=nn.Dense(256,num_classes)
        self.flatten=nn.Flatten()
        self.dropout=nn.Dropout(keep_prob=0.7)
        self.relu=nn.ReLU()
        #self.pad = nn.Pad(paddings=((2, 2), (2, 2)), mode="CONSTANT")
    def construct(self, x):
        #x=self.averagePool(x)
        #x=nn.Pad(x,paddings=((2,1),(2,1)))
        x=self.conv(x)
        #nn.Pad(x, paddings=((2, 1), (2, 1)))
        x=self.flatten(x)
        x=self.dropout(x)
        x=self.relu(self.fc1(x))
        x=self.dropout(x)
        x=self.fc2(x)
        return x

class LossAux(_Loss):
    def __init__(self,reduction="mean"):
        super(LossAux, self).__init__(reduction)
        self.loss=nn.SoftmaxCrossEntropyWithLogits(sparse=True)
    def construct(self,y,aux2,aux1,labels):
        x1=self.loss(y,labels)
        x2=self.loss(aux2,labels)
        x3=self.loss(aux1,labels)
        return self.get_loss(x1)+self.get_loss(x2)*0.3+self.get_loss(x3)*0.3

class CustomWithLossCell(nn.Cell):
    def __init__(self,backbone,loss_fn):
        super(CustomWithLossCell, self).__init__(auto_prefix=False)
        self._backbone=backbone
        self._loss_fn=loss_fn
    def construct(self,data,label):
        output,aux2,aux1=self._backbone(data)
        return self._loss_fn(output,aux2,aux1,label)

class GoogleNetAux(nn.Cell):
    """
    Googlenet architecture
    """
    def __init__(self, num_classes, aux_logits=True,include_top=True,train=True):
        super(GoogleNetAux, self).__init__()
        self.train=train
        self.aux_logits=aux_logits
        #self.conv1 = Conv2dBlock(3, 64, kernel_size=7, stride=2, padding=0)
        self.conv1 = Conv2dBlock(3, 64, kernel_size=7, stride=2, padding=0)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")
        self.conv2 = Conv2dBlock(64, 64, kernel_size=1)
        self.conv3 = Conv2dBlock(64, 192, kernel_size=3, padding=0)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")
        self.block3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.block3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")
        self.block4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.block4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.block4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.block4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.block4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode="same")
        self.block5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.block5b = Inception(832, 384, 192, 384, 48, 128, 128)

        if self.aux_logits:
            self.aux1=InceptionAux(512,num_classes)
            self.aux2=InceptionAux(528,num_classes)
        self.dropout = nn.Dropout(keep_prob=0.7)
        self.include_top = include_top
        if self.include_top:
            self.mean = P.ReduceMean(keep_dims=True)
            self.flatten = nn.Flatten()
            self.classifier = nn.Dense(1024, num_classes, weight_init=weight_variable(),
                                       bias_init=weight_variable())
    def construct(self, x):
        """construct"""
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        x = self.block3a(x)
        x = self.block3b(x)
        x = self.maxpool3(x)
        x = self.block4a(x)
        if self.train and self.aux_logits:
            aux1=self.aux1(x)
        x = self.block4b(x)
        x = self.block4c(x)
        x = self.block4d(x)
        if self.train and self.aux_logits:
            aux2=self.aux2(x)
        x = self.block4e(x)
        x = self.maxpool4(x)
        x = self.block5a(x)
        x = self.block5b(x)
        if not self.include_top:
            return x
        x = self.mean(x, (2, 3))
        x = self.flatten(x)
        x=self.dropout(x)
        x = self.classifier(x)
        if self.training and self.aux_logits:
            return x,aux2,aux1
        return x