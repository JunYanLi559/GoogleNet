import matplotlib.pyplot as plt
import mindspore.dataset as ds # 数据集的载入
import mindspore.dataset.transforms.c_transforms as C # 常用转化算子
import mindspore.dataset.vision.c_transforms as CV # 图像转化算子
from mindspore.common import dtype as mstype # 数据形态转换
import mindspore.nn as nn # 各类网络层都在nn里面
from GoogleNetv1 import LossAux
import numpy as np
from mindspore import Model # 承载网络结构
from mindspore import save_checkpoint, load_checkpoint,load_param_into_net# 保存与读取最佳参数
from mindspore import context # 设置mindspore运行的环境
from GoogleNetv1 import GoogleNetAux
from GoogleNetv1 import CustomWithLossCell
from mindspore.nn.metrics import Accuracy
# 数据路径处理
import os
device_target = context.get_context('device_target')
# 获取运行装置（CPU，GPU，Ascend）
dataset_sink_mode = True if device_target in ['Ascend','GPU'] else False
# 是否将数据通过pipeline下发到装置上
context.set_context(mode = context.PYNATIVE_MODE, device_target = device_target)
# 设置运行环境，静态图context.GRAPH_MODE指向静态图模型，即在运行之前会把全部图建立编译完毕
print(f'device_target: {device_target}')
print(f'dataset_sink_mode: {dataset_sink_mode}')
# 数据路径
train_path = os.path.join('../cifar-10-batches-bin') # 训练集路径
test_path = os.path.join('../cifar-10-verify-bin') # 测试集路径
print(f'训练集路径：{train_path}')
print(f'测试集路径：{test_path}')
# 创建图像标签列表
category_dict = {0:'airplane',1:'automobile',2:'bird',3:'cat',4:'deer',5:'dog',
                 6:'frog',7:'horse',8:'ship',9:'truck'}
ds_train = ds.Cifar10Dataset(train_path)
#计算数据集平均数和标准差，数据标准化时使用
tmp = np.asarray( [x['image'] for x in ds_train.create_dict_iterator(output_numpy=True)] )
RGB_mean = tuple(np.mean(tmp, axis=(0, 1, 2)))
RGB_std = tuple(np.std(tmp, axis=(0, 1, 2)))
def create_dataset(data_path, batch_size=32, repeat_num=1, usage='train'):
    # 载入数据集
    data = ds.Cifar10Dataset(data_path)
    # 打乱数据集
    data = data.shuffle(buffer_size=10000)
    # 定义算子
    if usage == 'train':
        trans = [
            CV.Normalize(RGB_mean, RGB_std),  # 数据标准化
            CV.HWC2CHW()  # 通道前移（为配适网络，CHW的格式可最佳发挥昇腾芯片算力）
        ]
    else:
        trans = [
            CV.Normalize(RGB_mean, RGB_std),  # 数据标准化
            CV.HWC2CHW()  # 通道前移（为配适网络，CHW的格式可最佳发挥昇腾芯片算力）
        ]
    typecast_op = C.TypeCast(mstype.int32)  # 原始数据的标签是unint，计算损失需要int
    # 算子运算
    data = data.map(input_columns='label', operations=typecast_op)
    data = data.map(input_columns='image', operations=trans)
    # 批处理
    data = data.batch(batch_size, drop_remainder=True)
    return data
train_data=create_dataset(train_path,batch_size=100,usage='train')
test_data=create_dataset(test_path,batch_size=50,usage='test')
net=GoogleNetAux(10,aux_logits=True,train=False)
net.set_train(False)
# 模型
ckpt_path = os.path.join('.','results') # 网络参数保存路径
accuracy_list=[]
for i in range(50):
    filename='checkpoint_googlenet_v1-%d_500.ckpt'%(i+1)
    param_dict=load_checkpoint(os.path.join(ckpt_path, filename))
    load_param_into_net(net,param_dict)
    loss=LossAux()
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    model=Model(net,net_loss,metrics={'accuracy': Accuracy()})
    res = model.eval(test_data, dataset_sink_mode=False)
    print(filename+':Accuracy')
    accuracy_list.append(res['accuracy'])
    print(res)
x=np.linspace(1,50,50)
y=np.array(accuracy_list)
plt.plot(x,y)
plt.xlabel('epoch')
plt.ylabel('acc_eval')
plt.show()