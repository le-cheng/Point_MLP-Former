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
# parser.add_argument('--token', type=int, help='random seed')   

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
    # if args.token is not None:
    #     cfg.token = args.token
    
    # prepare file structures
    time_str = time.strftime("%Y-%m-%d_%H:%M_", time.localtime())
    root_dir = cfg.log_dir+time_str+cfg.exp_name
    backup_dir = root_dir + '/backups'

    if not os.path.exists(cfg.log_dir):
        os.makedirs(cfg.log_dir)
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    else:
        a = input("File location already exists, overwrite or not:  y / n  ?")
        # if a == 'n':
        #     # os._exit(0)
        #     try:
        #         sys.exit(0)
        #         os._exit(0)
        #     except:
        #         print ('exit')
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    
    os.system('cp '+ __file__ + ' ' + backup_dir + '/train.py')
    os.system('cp models/{}.py '.format(cfg.model_name) + backup_dir + '/model.py')
    os.system('cp models/{}.py models/{}.py'.format(cfg.model_name, cfg.model_copy_name))
    os.system('cp '+ args.config + ' ' + backup_dir + '/config.yaml')


    logger = get_logger(os.path.join(root_dir, cfg.logger_filename))
    logger.info("start code ---------------")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu)
    assert torch.cuda.is_available(), "Please ensure codes are executed in cuda."
    logger.info(f'Using NO.{str(cfg.gpu)} device')
    # logger.info("There are {} gpus in total".format(torch.cuda.device_count()))
    # logger.info("Torch Version: {}".format(torch.__version__))
    # logger.info("Cuda Version: {}".format(torch.version.cuda))
    # assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."
    # logger.info("Cudnn Version: {}".format(torch.backends.cudnn.version()))
    logger.info("use Model: [ {} ]".format(cfg.model_name))
    logger.info("use Data: [ {} ]".format(cfg.data_name))
    logger.info("data class number: [ {} ]".format(cfg.num_classes))
    logger.info("epochs: [ {} ]".format(cfg.epochs))
    logger.info("lr: [ {} ]".format(cfg.learning_rate))
    
    logger.info("token: [ {} ]".format(cfg.token))
    
    '''init'''
    if args.seed is not None:
        seed = args.seed
    else:
        seed = torch.randint(1, 10000,(1,))
    logger.info("Seed: [ {} ]".format(seed))
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.set_printoptions(10)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    '''MODEL LOADING'''
    logger.info('Load MODEL ---------------')
    model = getattr(importlib.import_module('models.{}'.format(cfg.model_copy_name)), 'Module')(cfg).cuda()
    if cfg.print_model:
        logger.info(model)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters/1e6} M")

    optimizer = torch.optim.SGD(model.parameters(),lr=cfg.learning_rate,momentum=0.9,weight_decay=1e-4)
    lossfn = cal_loss

    # scheduler = CosineAnnealingLR(optimizer, cfg.epochs, eta_min=cfg.learning_rate / 100)
    scheduler = CosineLRScheduler(optimizer, t_initial=cfg.epochs,lr_min=cfg.learning_rate/100, cycle_mul=1, 
                                    cycle_decay = 0.5, cycle_limit=1,warmup_t=10, warmup_lr_init=cfg.learning_rate/100)

    # Optionally resume from a checkpoint
    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location = lambda storage, loc: storage.cuda(args.gpu))
            start_epoch = checkpoint['epoch']
            test_acc1 = checkpoint['test_acc']
            class_acc = checkpoint['class_acc']
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
        
    '''DATA LOADING'''
    logger.info('Load dataset ---------------')
    TRAIN_DATASET, TEST_DATASET=dataset_get(cfg.data_dir, model_name = cfg.data_name, num_points=cfg.num_point)
    Train_DataLoader = DataLoader(TRAIN_DATASET, num_workers=cfg.workers, batch_size=cfg.batch_size, shuffle=True, drop_last=True, pin_memory=True)
    Test_DataLoader = DataLoader(TEST_DATASET, num_workers=cfg.workers, batch_size=cfg.test_batch_size, pin_memory=True)
    
    global_epoch = 0
    best_test_acc = 0.0
    best_class_acc = 0.0
    best_epoch = 0
    '''TRANING'''
    logger.info('Start training...\n')
    writer = SummaryWriter(root_dir)
    batch_time = AverageMeter()
    t1 = time.time()
    for epoch in range(start_epoch, cfg.epochs):
        epoch_t1 = time.time()
        lr = optimizer.param_groups[0]['lr']
        logger.info('Epoch %d (%d/%s), lr=%f:' % (global_epoch + 1, epoch + 1, cfg.epochs,lr))

        # train
        train_acc, train_loss, norm = train(model, Train_DataLoader, optimizer, lossfn)
        scheduler.step(epoch+1)
        logger.info('Train Accuracy: %f , Train Loss: %f , norm: %f'\
             % (train_acc, train_loss,norm)) 
        writer.add_scalar('Test/train_Acc', train_acc, epoch)
        writer.add_scalar('Loss/train_Loss', train_loss, epoch)

        # test
        test_acc1, test_acc5, class_acc, test_loss= test(model, Test_DataLoader, cfg.num_classes, lossfn=lossfn, js=cfg.test_js)
        if (class_acc >= best_class_acc):
            best_class_acc = class_acc
        if (test_acc1 >= best_test_acc):
            best_test_acc = test_acc1
        logger.info('Test Acc: %f, Class Acc: %f, Loss:%f, Best: [%f]'% (test_acc1, class_acc, test_loss,best_test_acc))  
        if (test_acc1 >= best_test_acc):
            if (test_acc1 >= 83):
                best_epoch = epoch + 1
                savepath = root_dir+'/best_model.pth'
                logger.info('Save model..., Saving at %s'% savepath)
                state = {
                    'epoch': best_epoch,
                    'test_acc': test_acc1,
                    'class_acc': class_acc,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
        writer.add_scalar('Test/Test_Acc', test_acc1, epoch)
        writer.add_scalar('Best/Best_Acc', best_test_acc, epoch)
        writer.add_scalar('Test/ClassAcc', class_acc, epoch)
        writer.add_scalar('Best/Best_ClassAcc', best_class_acc, epoch)
        writer.add_scalar('Loss/test_loss', test_loss, epoch)

        global_epoch += 1
        batch_time.update(time.time() - epoch_t1)
        logger.info('%.4f h left'%(batch_time.avg/3600*(cfg.epochs-epoch-1)))
    logger.info('End of training...')
    logger.info('trian and eval model time is %.4f h'%((time.time()-t1)/3600))
    writer.close()
    logger.info('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')


def train(model, Train_DataLoader, optimizer, lossfn, ema=None):
    model.train()
    acc_meter = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    Train_DataLoader = tqdm(Train_DataLoader, ncols=100)
    for points, label in Train_DataLoader:
        points, label = points.cuda(non_blocking=True), label.squeeze(-1).cuda(non_blocking=True)
        # TODO : mixup_fn
        optimizer.zero_grad()
        pred = model(points)
        loss = lossfn(pred, label.long())
        
        #
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
        #
        [acc] = accuracy(pred, label, topk=(1,))
        acc_meter.update(acc.item(), label.size(0))
        loss_meter.update(loss.item(), label.size(0))
        norm_meter.update(grad_norm)
    return acc_meter.avg, loss_meter.avg, norm_meter.avg

@torch.no_grad()
def test(model, Test_DataLoader, num_class=40, lossfn=None, js=False,ema=None):
    model.eval()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    
    class_acc = torch.zeros((num_class,3)).cuda()
    Test_DataLoader = tqdm(Test_DataLoader, ncols=100)
    for points, label in Test_DataLoader:# label [b,1]
        points, label = points.cuda(non_blocking=True), label.squeeze(-1).cuda(non_blocking=True)
        # TODO : mixup_fn
        pred = model(points)
        loss = lossfn(pred, label.long())
        
        acc1, acc5 = accuracy(pred, label, topk=(1, 5))
        loss_meter.update(loss.item(), label.size(0))
        acc1_meter.update(acc1.item(), label.size(0))
        acc5_meter.update(acc5.item(), label.size(0))

        if js:
            pred = pred.argmax(dim=1, keepdim=True)
            for cat in torch.unique(label):
                cat_idex = (label==cat)
                classacc = pred[cat_idex].eq(label[cat_idex].view_as(pred[cat_idex])).sum()
                class_acc[cat,0] += classacc
                class_acc[cat,1] += cat_idex.sum()
    if js:
        class_acc[:,2] =  class_acc[:,0] / class_acc[:,1]
    class_acc = torch.mean(class_acc[:,2])
    return acc1_meter.avg, acc5_meter.avg, class_acc*100, loss_meter.avg

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
    gc.collect()
    main()
