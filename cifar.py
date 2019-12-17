'''
Training script for CIFAR-10/100
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function

import argparse
import os
import shutil
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models

import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA

from utils import AverageMeter, accuracy, mkdir_p, cifar_loader
import utils.log
import utils.visualize as vis


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')

# Optimization options
parser.add_argument('--epochs', default=180, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int,nargs='+', default=[80, 150],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, 
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--inlier', default='543210', type=str,
                    help='inliers')

# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoints', 
                    type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--block-name', type=str, default='BasicBlock',
                    help='the building block for Resnet and Preresnet')
parser.add_argument('--cardinality', type=int, default=8, 
                    help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, 
                    help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, 
                    help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, 
                    help='Compression Rate (theta) for DenseNet.')


# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', \
        'Dataset can only be cifar10 or cifar100.'

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy

def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)
    exp_name = args.checkpoint.split('/')[-1]
    logger = utils.log.logger_setting(exp_name, args.checkpoint)
    print('Experiment Name : %s'%exp_name)
    log_prefix = 'Epoch:[%3d | %d] Loss(Tr): %.4f, Acc(Tt): %.4f \n' + \
                 'LR: %.3f, Acc(Tr): %.4f,  Loss(Tt): %.3f'


    # Data
    print('==> Preparing dataset %s' % args.dataset)
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                             (0.2023, 0.1994, 0.2010)),
    ])
    if args.dataset == 'cifar10':
        num_classes = 6
    else:
        num_classes = 100


    inlist = []
    for j in range(len(args.inlier)):
        inlist.append(int(args.inlier[j]))

    data_path = os.path.join('./data',args.dataset)
    trainset = utils.cifar_loader.CIFARLoader(root=data_path, 
                                              train=True,
                                              inlist=inlist,
                                              transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, 
                                  shuffle=True, num_workers=args.workers)



    testset = utils.cifar_loader.CIFARLoader(root=data_path, 
                                               train=False,
                                               inlist=inlist,
                                               transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, 
                                 shuffle=False, num_workers=args.workers)

    # Model
    print("==> creating model '{}'".format(args.arch))
    criterion = [nn.CrossEntropyLoss()]
    #criterion = [nn.BCEWithLogitsLoss()]
    #criterion = [nn.BCELoss()]
    if args.arch.startswith('resnext'):
        model = models.__dict__[args.arch](
                    cardinality=args.cardinality,
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('densenet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    growthRate=args.growthRate,
                    compressionRate=args.compressionRate,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('wrn'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.endswith('resnet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    block_name=args.block_name,
                )
    elif args.arch.endswith('ae'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    block_name=args.block_name,
                )
        criterion.append(nn.MSELoss())
    else:
        model = models.__dict__[args.arch](num_classes=num_classes)

    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    print('    Total params: %.2fM' % \
          (sum(p.numel() for p in model.parameters())/1000000.0))

    optimizer = optim.SGD(model.parameters(), lr=args.lr, 
                          momentum=args.momentum, 
                          weight_decay=args.weight_decay)

    # Resume
    title = 'cifar-10-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        print(args.resume)
        assert os.path.isfile(args.resume), \
                'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = 0
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])



    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc, auroc = test(testloader, model, criterion, 
                                          start_epoch, use_cuda)
        print(' Test Loss: %.8f, Test Acc: %.2f%%' % \
                    (test_loss, test_acc))

        n_components = 5
        # if n_components is not passes, compute full rank
        #train_pca = calc_pca(trainloader, model, use_cuda)
        train_pca = calc_pca(trainloader, model, use_cuda, n_components)
        _, _, auroc = test(testloader, model, criterion, 
                           start_epoch, use_cuda, train_pca)
        print('AUROC: %.4f' % (auroc))
        return

        print(train_pca.singular_values_)
        print('\nEvaluation with Adversarial Attack')
        adv_top1, top1= advattack(trainloader,model, criterion[:1],
                                  start_epoch, use_cuda, 
                                  train_pca, n_components)
        print(' Adv Top1 Acc: %.2f%%, Top1 Acc: %.2f%%' % \
                    (adv_top1, top1))

        return


    # Train and val
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)


        train_loss, train_acc = train(trainloader, model, 
                                      criterion, optimizer, 
                                      epoch, use_cuda)
        test_loss, test_acc, _ = test(testloader, model, 
                                   criterion, epoch, use_cuda)

        msg = log_prefix%(epoch+1, args.epochs, train_loss, \
                          test_acc/100, state['lr'], \
                          train_acc/100, test_loss)
        logger.info(msg)

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=args.checkpoint)


    print('Best acc:')
    print(best_acc)

def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()
    mseloss = nn.MSELoss()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    max_iter = trainloader.__len__()
    num_print = 13
    denom = max_iter // num_print
    per = 1/num_print*100

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(async=True)
        inputs  = torch.autograd.Variable(inputs)
        targets = torch.autograd.Variable(targets)


        # Compute output
        outputs, feats = model(inputs)

        ## Normalization for BCE
        #outputs = outputs / torch.max(outputs, dim=1, keepdim=True)[0]
        #outputs = torch.max(outputs, outputs*0+0.0001)
        #outputs = torch.min(outputs, outputs*0-0.0001+1)


        #BCE_label = torch.zeros_like(outputs)
        #BCE_label.scatter_(1, targets.unsqueeze(1), 1)
        #loss = criterion[0](outputs, BCE_label)

        loss = criterion[0](outputs, targets)

        if len(criterion)==2:
            loss = loss + criterion[1](feats[-1], inputs)

        # Weight orthogonality
        W = model.module.fc.weight
        WWT = torch.matmul(W, W.t())
        eye = torch.autograd.Variable(torch.eye(WWT.shape[0]).cuda())
        loss_orthogonality = mseloss(WWT, eye)
        loss = loss + loss_orthogonality


        # Measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return (losses.avg, top1.avg)

def advattack(dataloader, model, criterion, epoch, use_cuda, 
              pca=None, n_components=5):
    classes = ['airplane', 'automobile', 'bird',  'cat',  'deer',
               'dog',      'frog',       'horse', 'ship', 'truck']
    softmax = nn.Softmax(dim=1)
    top1 = AverageMeter()
    advtop1 = AverageMeter()
    # switch to evaluate mode
    model.eval()
    #model.requires_grad=True

    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        #inputs  = torch.autograd.Variable(inputs, volatile=True)
        init_inputs = inputs
        inputs  = torch.autograd.Variable(inputs, requires_grad=True)
        targets = torch.autograd.Variable(targets)

        # compute output
        model.zero_grad()
        outputs, feats = model(inputs)
        feature = feats[0]
        loss = criterion[0](outputs, targets)
        loss.backward()

        maps  = [ model.module.list_actmap[i] \
                    for i in range(len(model.module.list_actmap)) ]
        grads = [ model.module.list_grad[-i-1] \
                    for i in range(len(model.module.list_grad)) ]

        init_pred = outputs.max(1, keepdim=True)[1]
        num_class = outputs.shape[1]
        #new_target = targets + torch.randint(1, num_class)
        new_target = targets + 1
        new_target = torch.remainder(new_target, num_class)

        # Attack loop
        for i in range(20):
            inputs.grad = None
            adv_out, feat = model(inputs)
            adv_feature = feat[0]
            loss = criterion[0](adv_out, new_target)
            loss.backward()
            data_grad = 0.01*torch.sign(inputs.grad.data)
            inputs.data = inputs.data - data_grad

            adv_maps  = [ model.module.list_actmap[i] \
                        for i in range(len(model.module.list_actmap)) ]
            adv_grads = [ model.module.list_grad[-i-1] \
                        for i in range(len(model.module.list_grad)) ]
            
        prec1 = accuracy(outputs.data, targets.data, topk=(1, ))
        advprec1 = accuracy(adv_out.data, targets.data, topk=(1, ))
        top1.update(prec1[0], inputs.size(0))
        advtop1.update(advprec1[0], inputs.size(0))
        
        if batch_idx == 1:
            idx=10
            img = inputs[idx].cpu().data.cpu().numpy()
            init_img = init_inputs[idx].cpu().numpy()
            recon = feats[-1][idx].data.cpu().numpy()
            img_pert = vis._save_image(img)
            img_init = vis._save_image(init_img)
            #vis._save_image(recon)

            cam_init = gradCAM(torch.autograd.Variable(init_inputs, 
                                                   requires_grad=True), 
                               targets, model, idx)
            cam_pert = gradCAM(inputs, 
                               new_target, model, idx)
            label_init = classes[targets.data.cpu().numpy()[idx]]
            label_pert = classes[new_target.data.cpu().numpy()[idx]]

            vis._save_2x2([img_init, cam_init, img_pert, cam_pert],
                          'gradCAM_%s_%s.jpg'%(label_init, label_pert))

            break

    if pca is not None:
        W = model.state_dict()['module.fc.weight'].cpu().numpy() #10x64
        print(np.matmul(W, np.transpose(W)))
        fit_W = pca.transform(W + pca.mean_)
        for f in [feature, adv_feature]:
            f = f.data.cpu().numpy()
            fit_feature = pca.transform(f)

            PCs  = fit_feature[:,:n_components]
            nPCs = fit_feature[:,n_components:]

            norm_PCs  = np.linalg.norm(PCs,  ord=2, axis=1)
            norm_nPCs = np.linalg.norm(nPCs, ord=2, axis=1)
            print(norm_PCs[:5] / norm_nPCs[:5])
            print(np.mean(norm_PCs), np.mean(norm_nPCs))


    return (advtop1.avg, top1.avg)


def test(testloader, model, criterion, epoch, use_cuda, pca=None):
    # Evaluates closed set accuracy,
    #           opened set accuracy,
    #           openset AUROC
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    softmax = nn.Softmax(dim=1)
    sigmoid = nn.Sigmoid()

    # switch to evaluate mode
    model.eval()

    list_targets = []
    list_feats   = []
    list_softmax = []

    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs  = torch.autograd.Variable(inputs, volatile=True)
        targets = torch.autograd.Variable(targets)

        # compute output
        outputs, feats = model(inputs)

        #BCE_label = torch.zeros_like(outputs)
        #BCE_label.scatter_(1, targets.unsqueeze(1), 1)
        #loss = criterion[0](outputs, BCE_label)


        #loss = criterion[0](outputs, targets)

        list_targets.append(targets.cpu().data.numpy())
        list_feats.append(feats[0].cpu().data.numpy())
        list_softmax.append(softmax(outputs).cpu().data.numpy())

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(0, inputs.size(0))
        #losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        

    feature = np.vstack(list_feats)
    pred    = np.vstack(list_softmax)
    label   = np.concatenate(list_targets)

    num_test = label.shape[0]
    ind_outlier = label==-100
    num_outlier = np.sum(ind_outlier)
    num_inlier  = num_test - num_outlier
    top1_acc = top1.avg * num_test / num_inlier

    if pca is None:
        in_score = np.amax(pred, axis=1)
        #in_score = np.mean(pred, axis=1)
        fpr, tpr, ths = roc_curve(ind_outlier, in_score, pos_label=0)
        print('AUROC for softmax thres : %.4f'%auc(fpr, tpr))
    else:
        W = model.state_dict()['module.fc.weight'].cpu().numpy() #10x64
        print(np.matmul(W, np.transpose(W)))

        ortho_feat = feature - np.matmul(pred, W)
        #in_score = 1. / np.linalg.norm(ortho_feat, axis=1)
        #in_score = in_score * np.linalg.norm(feature, axis=1)
        in_score = np.linalg.norm(feature, axis=1)
        #fit_feature = pca.transform(feature)
        #fit_W = pca.transform(W)
        #pca_output = np.matmul(fit_feature, np.transpose(fit_W))
        #pred = np.argmax(pca_output, axis=1)
        #pca_acc = np.sum(pred==label)/100 * num_test / num_inlier
        #print('PCA accuracy : %.2f'%pca_acc)

        #pca_feat = np.matmul(fit_feature, pca.components_) + pca.mean_
        #in_score = 1. / np.linalg.norm(feature - pca_feat, axis=1) 
        #in_score = in_score * np.linalg.norm(pca_feat, axis=1) 
        #in_score = np.linalg.norm(fit_feature[:,-5:], axis=1)
        fpr, tpr, ths = roc_curve(ind_outlier, in_score, pos_label=0)

    return (losses.avg, top1_acc, auc(fpr, tpr))


def calc_pca(dataloader, model, use_cuda, n_components=None):
    # switch to evaluate mode
    model.eval()
    list_feats   = []
    for batch_idx, (inputs, _) in enumerate(dataloader):
        if use_cuda:
            inputs= inputs.cuda()
        inputs  = torch.autograd.Variable(inputs, volatile=True)
        # compute output
        outputs, feats = model(inputs)
        list_feats.append(feats[0].cpu().data.numpy())
    feature = np.vstack(list_feats)

    # PCA
    if n_components is None:
        n_components = feature.shape[1]
    pca = PCA(n_components=n_components)
    pca.fit(feature)

    return pca


def gradCAM(data_batch, target, model, filename, idx=0):
    data  = data_batch[idx:idx+1]
    label = target[idx]
    outputs, feats = model(data_batch)
    #pred = torch.max(outputs, dim=1)[0]
    #loss = torch.sum(pred)
    loss = torch.sum(outputs[:,label])
    loss.backward()

    maps  = [ model.module.list_actmap[i] \
                for i in range(len(model.module.list_actmap)) ]
    grads = [ model.module.list_grad[-i-1] \
                for i in range(len(model.module.list_grad)) ]

    out = []
    for f,g in zip(maps, grads):
        alpha = torch.mean(g,     dim=2, keepdim=True)
        alpha = torch.mean(alpha, dim=3, keepdim=True)
        local_CAM = torch.sum(alpha * f, dim=1)[idx].cpu().numpy()
        out.append(local_CAM)
    return vis._save_gradCAM(out)


def gradCAM_old(feats, grads, filename, idx=0):
    out = []
    for f,g in zip(feats, grads):
        alpha = torch.mean(g,     dim=2, keepdim=True)
        alpha = torch.mean(alpha, dim=3, keepdim=True)
        local_CAM = torch.sum(alpha * f, dim=1)[idx].cpu().numpy()
        out.append(local_CAM)
    vis._save_gradCAM(out, filename)


def save_checkpoint(state, is_best, checkpoint='checkpoint', 
                    filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        dst = os.path.join(checkpoint, 'model_best.pth.tar')
        shutil.copyfile(filepath, dst)
def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

if __name__ == '__main__':
    main()
