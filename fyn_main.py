#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models.resnet import resnet101
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Resize, Compose, ToPILImage, ToTensor
import pickle
import math

# ============================= Feature Pyramid Network ================================= #
def smooth(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
    )

def predict(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
    )

class I2D(nn.Module):
    def __init__(self, pretrained=True):
        super(I2D, self).__init__()

        resnet = resnet101(pretrained=pretrained)

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = nn.Sequential(resnet.layer1) # 256
        self.layer2 = nn.Sequential(resnet.layer2) # 512
        self.layer3 = nn.Sequential(resnet.layer3) # 1024
        self.layer4 = nn.Sequential(resnet.layer4) # 2048

        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        
        # Depth prediction
        self.predict1 = smooth(256, 64)
        self.predict2 = predict(64, 1)
        
    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y

    def forward(self, x):
        _,_,H,W = x.size() # batchsize N,channel,height,width
        
        # Bottom-up
        c1 = self.layer0(x) 
        c2 = self.layer1(c1) # 256 channels, 1/4 size
        c3 = self.layer2(c2) # 512 channels, 1/8 size
        c4 = self.layer3(c3) # 1024 channels, 1/16 size
        c5 = self.layer4(c4) # 2048 channels, 1/32 size

        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4)) # 256 channels, 1/16 size
        p4 = self.smooth1(p4) 
        p3 = self._upsample_add(p4, self.latlayer2(c3)) # 256 channels, 1/8 size
        p3 = self.smooth2(p3) # 256 channels, 1/8 size
        p2 = self._upsample_add(p3, self.latlayer3(c2)) # 256, 1/4 size
        p2 = self.smooth3(p2) # 256 channels, 1/4 size

        return self.predict2( self.predict1(p2) )     # depth; 1/4 size, mode = "L"
# ============================= Network define ends ================================= #

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOAD_DIR = "."

class NYUv2Dataset(data.Dataset):
    def __init__(self, train=True):
        self.load_type = "train" if train else "test"
        if train:
            self.name_map = pickle.load(open("{}/nyuv2/index1.pkl".format(LOAD_DIR),'rb'))
            self.rgb_paths = list(self.name_map.keys())
        else:
            self.name_map = pickle.load(open("{}/nyuv2/index2.pkl".format(LOAD_DIR),'rb'))
            self.rgb_paths = list(self.name_map.keys())
        self.rgb_transform = Compose([ToTensor()]) # ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), 
        self.depth_transform = Compose([ToTensor()])
        self.length = len(self.rgb_paths)
            
    def __getitem__(self, index):
        path = '{}/nyuv2/{}_rgb/'.format(LOAD_DIR,self.load_type)+self.rgb_paths[index]
        rgb = Image.open(path)
        depth = Image.open('{}/nyuv2/{}_depth/'.format(LOAD_DIR,self.load_type)+self.name_map[self.rgb_paths[index]])
        depth = depth.resize((160,120))
        return self.rgb_transform(rgb).float(), self.depth_transform(depth).float()

    def __len__(self):
        return self.length

class RMSE_log(nn.Module):
    def __init__(self):
        super(RMSE_log, self).__init__()
    
    def forward(self, fake, real):
        if not fake.shape == real.shape:
            _,_,H,W = real.shape
            fake = F.upsample(fake, size=(H,W), mode='bilinear')
        eps = 1e-8
        loss = torch.sqrt( torch.mean( torch.abs(torch.log(real+eps)-torch.log(fake+eps)) ** 2 ) )
        return loss

class RMSE(nn.Module):
    def __init__(self):
        super(RMSE, self).__init__()
    def forward(self, fake, real):
        if not fake.shape == real.shape:
            _,_,H,W = real.shape
            fake = F.upsample(fake, size=(H,W), mode='bilinear')
        loss = torch.sqrt( torch.mean( torch.abs(10.*real-10.*fake) ** 2 ) )
        return loss

if __name__ == '__main__':
    # hyperparams
    lr = 0.001
    bs = 4
    # dataset
    train_dataset = NYUv2Dataset()
    train_size = len(train_dataset)
    eval_dataset = NYUv2Dataset(train=False)
    print(train_size)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=bs,shuffle=True)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=bs,shuffle=True)
    # network initialization
    i2d = I2D().to(DEVICE)
    try:
        i2d.load_state_dict(torch.load('{}/fyn_model.pt'.format(LOAD_DIR)))
        print("loaded model from drive")
    except:
        print('Initializing model...')
        print('Done!')
    # optimizer
    optimizer = torch.optim.Adam(i2d.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=4e-5)
    # evaluation loss function
    rmse = RMSE()
    eval_metric = RMSE_log()
    # train loss
    depth_criterion = RMSE_log() # depth_criterion = nn.MSELoss()
    # start training
    for epoch in range(0, 50):
        try:
            torch.cuda.empty_cache()
        except:
            pass
        i2d.train()
        start = time.time()
        # learning rate decay
        if epoch > 1:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.9
        for i,(data,z) in enumerate(train_dataloader):
            data,z = Variable(data.to(DEVICE)),Variable(z.to(DEVICE))
            optimizer.zero_grad()
            z_fake = i2d(data)
            loss = depth_criterion(z_fake, z)
            loss.backward()
            optimizer.step()
            if (i+1) % 50 == 0:
                print("[epoch %2d][iter %4d] loss: %.4f RMSElog: %.4f" % (epoch, i, loss, depth_loss))
        # save model
        torch.save(i2d.state_dict(),'{}/fyn_model.pt'.format(LOAD_DIR))
        end = time.time()
        print('model saved')
        print('time elapsed: %fs' % (end - start))
        # Evaluation
        if (epoch+1) % 5 == 0:
            try:
                torch.cuda.empty_cache()
            except:
                pass
            i2d.eval()
            print('evaluating...')
            eval_loss = 0
            rmse_accum = 0
            count = 0
            with torch.no_grad():
              for i,(data,z) in enumerate(eval_dataloader):
                  data,z = Variable(data.to(DEVICE)),Variable(z.to(DEVICE))
                  z_fake = i2d(data)
                  depth_loss = float(data.size(0)) * rmse(z_fake, z).item()**2
                  eval_loss += depth_loss
                  rmse_accum += float(data.size(0)) * eval_metric(z_fake, z).item()**2
                  count += float(data.size(0))
            print("[epoch %2d] RMSE_log: %.4f RMSE: %.4f" % (epoch, math.sqrt(eval_loss/count), math.sqrt(rmse_accum/count)))
