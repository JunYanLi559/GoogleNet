from easydict import EasyDict as ed
import os
import mindspore.common.dtype as mstype
import mindspore.dataset.engine as de

import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset.transforms.c_transforms as C2
import numpy as np
import mindspore.nn as nn
from matplotlib import pyplot as plt
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
import os
import random
import argparse
import mindspore.nn as nn
import mindspore.common.dtype as mstype
import mindspore.ops.functional as F
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset.transforms.c_transforms as C2
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.communication.management import init
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.nn.optim.momentum import Momentum
from mindspore.train.model import Model, ParallelMode
from mindspore import context
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor, Callback # 回调函数
from mindspore import Model # 承载网络结构
from mindspore import save_checkpoint, load_checkpoint # 保存与读取最佳参数
from mindspore import context # 设置mindspore运行的环境
import numpy as np # numpy
import matplotlib.pyplot as plt # 可视化用
import copy # 保存网络参数用
# 数据路径处理
import os, stat
config = ed({
    "class_num": 10,  # 分组10个。
    "batch_size": 32,  # 分组大小 32.
    "loss_scale": 1024,  # loss_scale是分
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "epoch_size": 50,
    "buffer_size": 100,
    "image_height": 224,
    "image_width": 224,
    "save_checkpoint": True,
    "save_checkpoint_steps": 195,
    "keep_checkpoint_max": 10,
    "save_checkpoint_path": "./data/",
    "lr_init": 0.01,
    "lr_end": 0.00001,
    "lr_max": 0.1,
    "warmup_epochs": 5,
    "lr_decay_mode": "poly"
})


def get_lr(global_step, lr_init, lr_end, lr_max, warmup_epochs, total_epochs, steps_per_epoch, lr_decay_mode):
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    warmup_steps = steps_per_epoch * warmup_epochs
    if lr_decay_mode == 'steps':
        decay_epoch_index = [0.3 * total_steps, 0.6 * total_steps, 0.8 * total_steps]
        for i in range(total_steps):
            if i < decay_epoch_index[0]:
                lr = lr_max
            elif i < decay_epoch_index[1]:
                lr = lr_max * 0.1
            elif i < decay_epoch_index[2]:
                lr = lr_max * 0.01
            else:
                lr = lr_max * 0.001
            lr_each_step.append(lr)
    elif lr_decay_mode == 'poly':
        if warmup_steps != 0:
            inc_each_step = (float(lr_max) - float(lr_init)) / float(warmup_steps)
        else:
            inc_each_step = 0
        for i in range(total_steps):
            if i < warmup_steps:
                lr = float(lr_init) + inc_each_step * float(i)
            else:
                base = (1.0 - (float(i) - float(warmup_steps)) / (float(total_steps) - float(warmup_steps)))
                lr = float(lr_max) * base * base
                if lr < 0.0:
                    lr = 0.0
            lr_each_step.append(lr)
    else:
        for i in range(total_steps):
            if i < warmup_steps:
                lr = lr_init + (lr_max - lr_init) * i / warmup_steps
            else:
                lr = lr_max - (lr_max - lr_end) * (i - warmup_steps) / (total_steps - warmup_steps)
            lr_each_step.append(lr)

    current_step = global_step
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    learning_rate = lr_each_step[current_step:]

    return learning_rate


def _weight_variable(shape, factor=0.01):
    init_value = np.random.randn(*shape).astype(np.float32) * factor
    return Tensor(init_value)


def _conv3x3(in_channel, out_channel, stride=1):
    weight_shape = (out_channel, in_channel, 3, 3)
    weight = _weight_variable(weight_shape)
    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=3, stride=stride, padding=0, pad_mode='same', weight_init=weight)


def _conv1x1(in_channel, out_channel, stride=1):
    weight_shape = (out_channel, in_channel, 1, 1)
    weight = _weight_variable(weight_shape)
    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=1, stride=stride, padding=0, pad_mode='same', weight_init=weight)


def _conv7x7(in_channel, out_channel, stride=1):
    weight_shape = (out_channel, in_channel, 7, 7)
    weight = _weight_variable(weight_shape)
    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=7, stride=stride, padding=0, pad_mode='same', weight_init=weight)


def _bn(channel):
    return nn.BatchNorm2d(channel, eps=1e-4, momentum=0.9,
                          gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1)


def _bn_last(channel):
    return nn.BatchNorm2d(channel, eps=1e-4, momentum=0.9,
                          gamma_init=0, beta_init=0, moving_mean_init=0, moving_var_init=1)


def _fc(in_channel, out_channel):
    weight_shape = (out_channel, in_channel)
    weight = _weight_variable(weight_shape)
    return nn.Dense(in_channel, out_channel, has_bias=True, weight_init=weight, bias_init=0)


class ResidualBlock(nn.Cell):
    expansion = 4

    def __init__(self,
                 in_channel,
                 out_channel,
                 stride=1):
        super(ResidualBlock, self).__init__()

        channel = out_channel // self.expansion
        self.conv1 = _conv1x1(in_channel, channel, stride=1)
        self.bn1 = _bn(channel)

        self.conv2 = _conv3x3(channel, channel, stride=stride)
        self.bn2 = _bn(channel)

        self.conv3 = _conv1x1(channel, out_channel, stride=1)
        self.bn3 = _bn_last(out_channel)

        self.relu = nn.ReLU()

        self.down_sample = False

        if stride != 1 or in_channel != out_channel:
            self.down_sample = True
        self.down_sample_layer = None

        if self.down_sample:
            self.down_sample_layer = nn.SequentialCell([_conv1x1(in_channel, out_channel, stride),
                                                        _bn(out_channel)])
        self.add = P.Add()

    def construct(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.down_sample:
            identity = self.down_sample_layer(identity)
        out = self.add(out, identity)
        out = self.relu(out)
        return out


class ResNet(nn.Cell):
    def __init__(self,
                 block,
                 layer_nums,
                 in_channels,
                 out_channels,
                 strides,
                 num_classes):
        super(ResNet, self).__init__()

        if not len(layer_nums) == len(in_channels) == len(out_channels) == 4:
            raise ValueError("the length of layer_num, in_channels, out_channels list must be 4!")

        self.conv1 = _conv7x7(3, 64, stride=2)
        self.bn1 = _bn(64)
        self.relu = P.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")

        self.layer1 = self._make_layer(block,
                                       layer_nums[0],
                                       in_channel=in_channels[0],
                                       out_channel=out_channels[0],
                                       stride=strides[0])
        self.layer2 = self._make_layer(block,
                                       layer_nums[1],
                                       in_channel=in_channels[1],
                                       out_channel=out_channels[1],
                                       stride=strides[1])
        self.layer3 = self._make_layer(block,
                                       layer_nums[2],
                                       in_channel=in_channels[2],
                                       out_channel=out_channels[2],
                                       stride=strides[2])
        self.layer4 = self._make_layer(block,
                                       layer_nums[3],
                                       in_channel=in_channels[3],
                                       out_channel=out_channels[3],
                                       stride=strides[3])

        self.mean = P.ReduceMean(keep_dims=True)
        self.flatten = nn.Flatten()
        self.end_point = _fc(out_channels[3], num_classes)

    def _make_layer(self, block, layer_num, in_channel, out_channel, stride):
        """
        Make stage network of ResNet.

        Args:
            block (Cell): Resnet block.
            layer_num (int): Layer number.
            in_channel (int): Input channel.
            out_channel (int): Output channel.
            stride (int): Stride size for the first convolutional layer.

        Returns:
            SequentialCell, the output layer.

        """
        layers = []

        resnet_block = block(in_channel, out_channel, stride=stride)
        layers.append(resnet_block)

        for _ in range(1, layer_num):
            resnet_block = block(out_channel, out_channel, stride=1)
            layers.append(resnet_block)

        return nn.SequentialCell(layers)

    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        c1 = self.maxpool(x)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        out = self.mean(c5, (2, 3))
        out = self.flatten(out)
        out = self.end_point(out)
        return out


def resnet50(class_num=10):
    return ResNet(ResidualBlock,
                  [3, 4, 6, 3],
                  [64, 256, 512, 1024],
                  [256, 512, 1024, 2048],
                  [1, 2, 2, 2],
                  class_num)


def resnet101(class_num=1001):
    return ResNet(ResidualBlock,
                  [3, 4, 23, 3],
                  [64, 256, 512, 1024],
                  [256, 512, 1024, 2048],
                  [1, 2, 2, 2],
                  class_num)


random.seed(1)
parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--do_train', type=bool, default=True, help='Do train or not.')
parser.add_argument('--do_eval', type=bool, default=False, help='Do eval or not.')
parser.add_argument('--epoch_size', type=int, default=50, help='Epoch size.')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
parser.add_argument('--num_classes', type=int, default=10, help='Num classes.')
parser.add_argument('--data_urt', type=str, default='../cifar-10-batches-bin', help='Dataset path')
parser.add_argument('--checkpoint_path', type=str, default='ckpt', help='CheckPoint file path.')
parser.add_argument('--dataset_path', type=str, default='../cifar-10-batches-bin', help='Dataset path.')
args_opt = parser.parse_known_args()[0]
device_target = context.get_context('device_target')  # 获取运行装置（CPU，GPU，Ascend）

dataset_sink_mode = True if device_target in ['Ascend', 'GPU'] else False  # 是否将数据通过pipeline下发到装置上

context.set_context(mode=context.GRAPH_MODE, device_target=device_target)


def create_dataset(data_path, repeat_num=1, training=True, batch_size=args_opt.batch_size):
    """
    create data for next use such as training or infering
    """
    cifar_ds = ds.Cifar10Dataset(data_path)

    cifar_ds = ds.Cifar10Dataset(data_path)

    resize_height = 224
    resize_width = 224
    rescale = 1.0 / 255.0
    shift = 0.0

    # 数据增强
    random_crop_op = C.RandomCrop((32, 32), (4, 4, 4, 4))  # 随机裁剪
    random_horizontal_op = C.RandomHorizontalFlip()  # 随机翻转
    resize_op = C.Resize((resize_height, resize_width))  # 重定义大小
    rescale_op = C.Rescale(rescale, shift)
    normalize_op = C.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    changeswap_op = C.HWC2CHW()
    type_cast_op = C2.TypeCast(mstype.int32)

    c_trans = []
    if training:
        c_trans = [random_crop_op, random_horizontal_op]
    c_trans += [resize_op, rescale_op, normalize_op,
                changeswap_op]

    cifar_ds = cifar_ds.map(input_columns="label", operations=type_cast_op)
    cifar_ds = cifar_ds.map(input_columns="image", operations=c_trans)

    # shuffle
    cifar_ds = cifar_ds.shuffle(buffer_size=10)

    # batch
    cifar_ds = cifar_ds.batch(batch_size=batch_size, drop_remainder=True)

    # repeat
    cifar_ds = cifar_ds.repeat(repeat_num)

    return cifar_ds


class TrainHistroy(Callback):
    def __init__(self, history):
        super(TrainHistroy, self).__init__()
        self.history = history

    # 每个epoch结束时执行
    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs.asnumpy()
        self.history.append(loss)


# 测试并记录模型在试集的loss和accuracy，每个epoch结束时进行模型测试并记录结果，跟踪并保存准确率最高的模型网络参数
class EvalHistory(Callback):
    # 保存accuracy最高的网络参数
    best_param = None

    def __init__(self, model, loss_history, acc_history, eval_data):
        super(EvalHistory, self).__init__()
        self.loss_history = loss_history
        self.acc_history = acc_history
        self.eval_data = eval_data
        self.model = model

    # 每个epoch结束时执行
    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        res = self.model.eval(self.eval_data, dataset_sink_mode=False)
        if len(self.acc_history) == 0 or res['accuracy'] >= max(self.acc_history):
            self.best_param = copy.deepcopy(cb_params.network)
        self.loss_history.append(res['loss'])
        self.acc_history.append(res['accuracy'])
        print('acc_eval: ', res['accuracy'])

    # 训练结束后执行
    def end(self, run_context):
        # 保存最优网络参数
        best_param_path = os.path.join(ckpt_path, 'best_param.ckpt')
        if os.path.exists(best_param_path):
            # best_param.ckpt已存在时MindSpore会覆盖旧的文件，这里修改文件读写权限防止报错
            os.chmod(best_param_path, stat.S_IWRITE)
        save_checkpoint(self.best_param, best_param_path)


if __name__ == '__main__':
    epoch_size = args_opt.epoch_size
    batch_size = args_opt.batch_size
    net = resnet50(class_num=args_opt.num_classes)
    ls = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 0.01, 0.9)

    model = Model(net, loss_fn=ls, optimizer=opt, metrics={'acc'})
    # 模型训练
    dataset = create_dataset('../cifar-10-batches-bin', epoch_size)
    test_data = create_dataset('../cifar-10-verify-bin', training=False)
    ckpt_path = os.path.join('..', 'results')  # 网络参数保存路径
    hist = {'loss': [], 'loss_eval': [], 'acc_eval': []}  # 训练过程记录
    # 网络参数自动保存，这里设定每2000个step保存一次，最多保存10次
    config_ck = CheckpointConfig(save_checkpoint_steps=2000, keep_checkpoint_max=10)
    ckpoint_cb = ModelCheckpoint(prefix='checkpoint_Inceptionv3googlenet', directory=ckpt_path, config=config_ck)
    # 监控每次迭代的时间
    time_cb = TimeMonitor(data_size=dataset.get_dataset_size())
    # 监控loss值
    loss_cb = LossMonitor(per_print_times=1)
    # 记录每次迭代的模型损失值
    train_hist_cb = TrainHistroy(hist['loss'])
    # 测试并记录模型在验证集的loss和accuracy，并保存最优网络参数
    eval_hist_cb = EvalHistory(model=model,
                               loss_history=hist['loss_eval'],
                               acc_history=hist['acc_eval'],
                               eval_data=test_data)
    epoch = 50  # 迭代次数
    model.train(epoch, dataset, callbacks=[train_hist_cb, eval_hist_cb, time_cb, ckpoint_cb, loss_cb],
                dataset_sink_mode=dataset_sink_mode)


def plot_loss(hist):
    plt.plot(hist['loss'], marker='.')
    plt.plot(hist['loss_eval'], marker='.')
    plt.title('loss record')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss_train', 'loss_eval'], loc='upper right')
    plt.show()
    plt.close()


plot_loss(hist)


def plot_accuracy(hist):
    plt.plot(hist['acc_eval'], marker='.')
    plt.title('accuracy history')
    plt.xlabel('epoch')
    plt.ylabel('acc_eval')
    plt.grid()
    plt.show()
    plt.close()


plot_accuracy(hist)