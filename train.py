import mindspore
# mindspore.dataset
import mindspore.dataset as ds # 数据集的载入
import mindspore.dataset.transforms.c_transforms as C # 常用转化算子
import mindspore.dataset.vision.c_transforms as CV # 图像转化算子
# mindspore.common
from mindspore.common import dtype as mstype # 数据形态转换
from mindspore.common.initializer import Normal # 参数初始化
# mindspore.nn
import mindspore.nn as nn # 各类网络层都在nn里面
from mindspore.nn.metrics import Accuracy, Loss # 测试模型用
from GoogleNetv1 import LossAux
# mindspore.train.callback
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor, Callback # 回调函数
from mindspore import Model # 承载网络结构
from mindspore import save_checkpoint, load_checkpoint # 保存与读取最佳参数
from mindspore import context # 设置mindspore运行的环境
from GoogleNetv1 import GoogleNetAux
from GoogleNetv1 import CustomWithLossCell
import numpy as np # numpy
import matplotlib.pyplot as plt # 可视化用
import copy # 保存网络参数用
# 数据路径处理
import os, stat
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
            # 数据增强
            CV.RandomCrop([32, 32], [4, 4, 4, 4]),  # 随机裁剪
            CV.RandomHorizontalFlip(),  # 随机翻转
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
net=GoogleNetAux(10,aux_logits=True)
loss=LossAux()
net_loss = CustomWithLossCell(net,loss)
# 优化器
net_opt=nn.Adam(params=net.trainable_params(),learning_rate=0.0006)
# 模型
model=Model(network=net_loss,optimizer=net_opt)
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
ckpt_path = os.path.join('.','results') # 网络参数保存路径
hist = {'loss':[], 'loss_eval':[], 'acc_eval':[]} # 训练过程记录
# 网络参数自动保存，这里设定每2000个step保存一次，最多保存10次
config_ck = CheckpointConfig(save_checkpoint_steps=2000, keep_checkpoint_max=10)
ckpoint_cb = ModelCheckpoint(prefix='checkpoint_googlenet_v1', directory=ckpt_path, config=config_ck)
# 监控每次迭代的时间
time_cb = TimeMonitor(data_size=ds_train.get_dataset_size())
# 监控loss值
loss_cb = LossMonitor(per_print_times=100)
# 记录每次迭代的模型损失值
train_hist_cb = TrainHistroy(hist['loss'])
# 测试并记录模型在验证集的loss和accuracy，并保存最优网络参数
eval_hist_cb = EvalHistory(model = model,
                           loss_history = hist['loss_eval'],
                           acc_history = hist['acc_eval'],
                           eval_data = test_data)
epoch = 50 # 迭代次数
model.train(epoch, train_data, callbacks=[LossMonitor(),TimeMonitor()], dataset_sink_mode=dataset_sink_mode)
# 定义loss记录绘制函数
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
# 使用准确率最高的参数组合建立模型，并测试其在验证集上的效果
load_checkpoint(os.path.join(ckpt_path, 'best_param.ckpt'), net=net)
res = model.eval(test_data, dataset_sink_mode=dataset_sink_mode)
print(res)
# 创建图像标签列表
category_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog',
                 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}

data_path = os.path.join('data', '10-verify-bin')
demo_data = create_dataset(test_path, batch_size=1, usage='test')


# 将数据标准化至0~1区间
def normalize(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range
