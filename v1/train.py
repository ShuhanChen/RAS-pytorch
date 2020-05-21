import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import pdb, os, argparse
from RAS import RAS
from data import get_loader

# set parameters
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=30, help='epoch number')
parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
parser.add_argument('--batchsize', type=int, default=10, help='training batch size')
parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
parser.add_argument('--decay_epoch', type=int, default=[26,], help='every n epochs decay learning rate')
opt = parser.parse_args()

# build models
model = RAS()
model.cuda()
params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)

# print the network information
num_params = 0
for p in model.parameters():
    num_params += p.numel()
print('RAS Structure')
print(model)
print("The number of parameters: {}".format(num_params))

# dataset path
image_path = '/home/ipal/datasets/DUTS_train/imgs/'
gt_path = '/home/ipal/datasets/DUTS_train/gts/'

train_loader = get_loader(image_path, gt_path, batchsize=opt.batchsize, trainsize=opt.trainsize)
total_step = len(train_loader)

def bce_iou_loss(pred, gt):
    bce  = F.binary_cross_entropy_with_logits(pred, gt, reduction='mean')

    pred  = torch.sigmoid(pred)
    inter = (pred*gt).sum(dim=(2,3))
    union = (pred+gt).sum(dim=(2,3))
    iou  = 1-(inter+1)/(union-inter+1)

    return (bce+iou).mean()

def train(train_loader, model, optimizer, epoch):
    model.train()
    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        image, gt = pack
        image = Variable(image).cuda()
        gt = Variable(gt).cuda()

        pred = model(image)
        loss1 = bce_iou_loss(pred[0], gt)
        loss2 = bce_iou_loss(pred[1], gt)
        loss3 = bce_iou_loss(pred[2], gt)
        loss4 = bce_iou_loss(pred[3], gt)
        loss5 = bce_iou_loss(pred[4], gt)
        loss_fuse = loss1 + loss2 + loss3 + loss4 + loss5
        loss = loss_fuse / opt.batchsize

        loss.backward()
        optimizer.step()

        if i % 20 == 0 or i == total_step:
            print('Learning rate: %g, epoch: [%2d/%2d], iter: [%5d/%5d]  ||  Loss: %10.4f' % (
                        opt.lr, epoch, opt.epoch, i, total_step, loss.data))
            print('Loss1: %10.4f' % (loss1.data / opt.batchsize))
            print('Loss2: %10.4f' % (loss2.data / opt.batchsize))
            print('Loss3: %10.4f' % (loss3.data / opt.batchsize))
            print('Loss4: %10.4f' % (loss4.data / opt.batchsize))
            print('Loss5: %10.4f' % (loss5.data / opt.batchsize))

    save_path = 'models/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if epoch % 5 == 0:
        torch.save(model.state_dict(), save_path + 'RAS.v1' + '.%d' % epoch + '.pth')

for epoch in range(1, opt.epoch+1):
    if epoch in opt.decay_epoch:
        opt.lr = opt.lr * 0.1
        params = model.parameters()
        optimizer = torch.optim.Adam(params, opt.lr)
    train(train_loader, model, optimizer, epoch)
