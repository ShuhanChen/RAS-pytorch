import os
import sys
sys.path.insert(0, '../')
sys.dont_write_bytecode = True

import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import data
from RAS import RAS
import time

class Test(object):
    def __init__(self, Dataset, Network, path):
        ## dataset
        self.cfg    = Dataset.Config(datapath=path, snapshot='./models/RAS.v2.pth', mode='test')
        self.data   = Dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=8)
        ## network
        self.net    = Network(self.cfg)
        self.net.train(False)
        self.net.cuda()
    
    def save(self):
        with torch.no_grad():
            time_t = 0.0

            for image, shape, name in self.loader:
                image = image.cuda().float()
                time_start = time.time()
                res, _, _, _ = self.net(image)
                torch.cuda.synchronize()
                time_end = time.time()
                time_t = time_t + time_end - time_start
                res = F.interpolate(res, shape, mode='bilinear', align_corners=True)
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                res = 255 * res
                save_path  = '/home/ipal/evaluation/SaliencyMaps/'+ self.cfg.datapath.split('/')[-1]+'/RAS-v2/'
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                cv2.imwrite(save_path+'/'+name[0]+'.png', res)
            fps = len(self.loader) / time_t
            print('FPS is %f' %(fps))


if __name__=='__main__':
    for path in ['/home/ipal/datasets/ECSSD', '/home/ipal/datasets/DUTS', '/home/ipal/datasets/DUT-OMRON', '/home/ipal/datasets/HKU-IS']:
        test = Test(data, RAS, path)
        test.save()
