import logging
import os
import socket
import time
import yaml

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn


class Dotdict(dict):
    """
    Dotdict represents an internal node in the configuration tree. It's a simple
    dict-like container that allows for attribute-based access to keys.
    """
    def __init__(self, init_dict=None):
        # Recursively convert nested dictionaries in init_dict into Dotdicts
        init_dict = {} if init_dict is None else init_dict
        
        for k, v in init_dict.items():
            if type(v) is dict:
                init_dict[k] = Dotdict(v)
        super(Dotdict, self).__init__(init_dict)

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __str__(self):
        def _indent(s_, num_spaces):
            s = s_.split("\n")
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * " ") + line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s
            return s
        r = ""
        s = []
        for k, v in sorted(self.items()):
            seperator = "\n" if isinstance(v, Dotdict) else " "
            attr_str = "{}:{}{}".format(str(k), seperator, str(v))
            attr_str = _indent(attr_str, 2)
            s.append(attr_str)
        r += "\n".join(s)
        return r

def read_yaml(file):
    cfg = {}
    assert os.path.isfile(file) and file.endswith('.yaml'), '{} is not a yaml file'.format(file)
    with open(file, 'r') as f:
        # yaml_file = yaml.safe_load(f)
        yaml_file = yaml.unsafe_load(f)
    for key in yaml_file:
        for k, v in yaml_file[key].items():
            cfg[k] = v
    cfg = Dotdict(cfg)
    return cfg


def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''
    gold = gold.contiguous().view(-1)
    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')
    return loss

def is_main_process():
    return get_rank() == 0  

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_dist_avail_and_initialized():
    """检查是否支持分布式环境"""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def reduce_value(value, average=True):
    world_size = get_world_size()
    if world_size < 2:  # 单GPU的情况
        return value
    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value = value/world_size
        return value

def find_free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port

def get_logger(filename='main.log'):
    logger = logging.getLogger(__name__)
    logger.setLevel(level = logging.INFO)

    # 屏幕输出
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)

    fmt = "==>  %(message)s"
    # fmt = "%(asctime)s [%(levelname)s line:%(lineno)d] %(message)s"
    formatter = logging.Formatter(fmt)
    sh.setFormatter(formatter)

    logger.addHandler(sh)

    # 文件输出
    fh = logging.FileHandler(filename)    #创建一个文件记录日志的handler,设置级别为info
    fh.setLevel(logging.INFO)

    fmt = "%(asctime)s [%(levelname)s %(filename)s line:%(lineno)d process:%(process)d] %(message)s"
    formatter = logging.Formatter(fmt)
    fh.setFormatter(formatter)

    logger.addHandler(fh) #把对象加到logger里
    return logger

# >>> print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
class Configs():
    """
    ## Configurations
    """
    gpu: list = [0]

    # Model
    # model: GAT
    # Number of nodes to train on
    training_samples: int = 500
    # Number of features per node in the input
    in_features: int
    # Number of features in the first graph attention layer
    n_hidden: int = 64
    # Number of heads
    
    # Number of classes for classification
    n_classes: int
    # Dropout probability
    dropout: float = 0.6
    # Whether to include the citation network
    include_edges: bool = True
    # Dataset
    # dataset: CoraDataset
    # Number of training iterations
    epochs = 1_000
    # Loss function
    loss_func = nn.CrossEntropyLoss()
    # Device to train on
    #
    # This creates configs for device, so that
    # we can change the device by passing a config value
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # Optimizer
    optimizer: torch.optim.Adam

    t_time1 = time.strftime("%c", time.localtime()) # Sun Jan 23 21:01:03 2022
    t_time2 = time.strftime("%B %A %w ", time.localtime()) # January Sunday 0  "月 周几 第几周"
    t_time3 = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()) # 2022-01-23_21:03:07
    t_time4 = time.strftime("%Z", time.localtime()) # CST "时区"
    t_time5 = time.strftime("%x", time.localtime()) # 01/23/22 "日期"
    t_time6 = time.strftime("%X", time.localtime()) # 21:03:07 "时间"
    t_time7 = time.strftime("%j", time.localtime()) # 023 "一年内的第几天(001-366)"

    torch_version = torch.__version__
    torch_usecuda = torch.cuda.is_available()
    cudnn_version = torch.backends.cudnn.version()

    
    exp_name: str = 'ppp'

if __name__ == '__main__':
    conf = Configs()
    print(conf.t_time1)
    print(conf.t_time2)
    print(conf.t_time3)
    print(conf.t_time4)
    print(conf.t_time5)
    print(conf.t_time6)
    print(conf.t_time7)

    print(conf.gpu)
