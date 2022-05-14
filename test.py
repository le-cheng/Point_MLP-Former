import argparse
import gc
import importlib
import os
import sys
import time

import numpy as np
import torch
from tensorboardX import SummaryWriter
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.step_lr import StepLRScheduler
from timm.utils import AverageMeter, accuracy
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import dataset_get
from utils import cal_loss, get_logger, read_yaml

# sys.path.append("./") 
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('-c', '--config', default='configs/config.yaml', type=str, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--exp_name', default=None, type=str, metavar='PATH',
                    help='exp_name')
parser.add_argument('--model', type=str, help='model_name')          
parser.add_argument('--seed', type=int, help='random seed')    
parser.add_argument('--token', type=int, help='random seed')   

def main():
    args = parser.parse_args()
    assert args.config is not None
    cfg = read_yaml(args.config)
    if args.exp_name is not None:
        cfg.exp_name = args.exp_name
    if args.model is not None:
        cfg.model_name = args.model
    if cfg.data_name == 'scanobjectnn':
        cfg.num_classes = 15
    if args.token is not None:
        cfg.token = args.token

    # prepare file structures
    time_str = time.strftime("%Y-%m-%d_%H:%M_", time.localtime())
    root_dir = cfg.log_dir+time_str+cfg.exp_name
    backup_dir = root_dir + '/backups'

    if not os.path.exists(cfg.log_dir):
        os.makedirs(cfg.log_dir)
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
  

    logger = get_logger(os.path.join(root_dir, cfg.logger_filename))
    # logger.info("start code ---------------")
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
   
    # logger.info("use Model: [ {} ]".format(cfg.model_name))
    # logger.info("use Data: [ {} ]".format(cfg.data_name))
    # logger.info("data class number: [ {} ]".format(cfg.num_classes))
    # logger.info("epochs: [ {} ]".format(cfg.epochs))
    # logger.info("lr: [ {} ]".format(cfg.learning_rate))
    if args.seed is not None:
        seed = args.seed
    else:
        seed = torch.randint(1, 10000,(1,))
    # seed = 6666
    logger.info("Seed: [ {} ]".format(seed))
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.set_printoptions(10)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)
    model = getattr(importlib.import_module('models.{}'.format('model')), 'Module')(cfg).cuda()
    # for param in model.parameters():
    #     print(param)
    # logger.info(model.state_dict())
    a = model.token
    
   
    # Optionally resume from a checkpoint
    if args.resume is not None:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            # print(checkpoint['model_state_dict'].keys())
            # start_epoch = checkpoint['epoch']
            # best_prec1 = checkpoint['test_acc']
            # class_acc = checkpoint['class_acc']
            model.load_state_dict(checkpoint['model_state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    '''DATA LOADING'''
    logger.info('Load dataset ---------------')
    Test_DataLoader = DataLoader(dataset_get(cfg.data_dir, model_name = cfg.data_name, num_points=cfg.num_point)[1], 
                                 num_workers=cfg.workers, batch_size=cfg.test_batch_size, pin_memory=True)
 
    t1 = time.time()
   
    test_acc1, test_acc5, class_acc= test(model, Test_DataLoader, cfg.num_classes)
    logger.info('Test Acc1: %f, Class Acc: %f'% (test_acc1, class_acc)) 
       
    logger.info('End of TESTing...')
    logger.info('trian and eval model time is %.4f S'%((time.time()-t1)))

@torch.no_grad()
def test(model, Test_DataLoader, num_class=15):
    model.eval()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    
    class_acc = torch.zeros((num_class,3)).cuda()
    # with torch.no_grad():
    Test_DataLoader = tqdm(Test_DataLoader, ncols=100)
    for points, label in Test_DataLoader:# label [b,1]
        points, label = points.cuda(non_blocking=True), label.squeeze(-1).cuda(non_blocking=True)
        # print(label)
        pred = model(points)
        acc1, acc5 = accuracy(pred, label, topk=(1, 5))
        acc1_meter.update(acc1.item(), label.size(0))
        acc5_meter.update(acc5.item(), label.size(0))

       
        pred = pred.argmax(dim=1, keepdim=True)
        for cat in torch.unique(label):
            cat_idex = (label==cat)
            classacc = pred[cat_idex].eq(label[cat_idex].view_as(pred[cat_idex])).sum()
            class_acc[cat,0] += classacc
            class_acc[cat,1] += cat_idex.sum()
        # break
    
    class_acc[:,2] =  class_acc[:,0] / class_acc[:,1]
    class_acc = torch.mean(class_acc[:,2])
    return acc1_meter.avg, acc5_meter.avg, class_acc

@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()
    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


if __name__ == '__main__':
    # 垃圾回收gc.collect() 返回处理这些循环引用一共释放掉的对象个数
    gc.collect()
    main()
