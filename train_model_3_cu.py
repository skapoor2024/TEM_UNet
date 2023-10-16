#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 13:27:40 2023
change from BCElogit loss to BCE loss for unet(1,1)->Unet(1,2)
@author: s.kapoor
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from sklearn.model_selection import train_test_split
from model import Unet

import random

import torchvision.transforms as transform
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

class Data(Dataset):

    def __init__(self,list_ids,folder):
        self.list_ids = list_ids
        self.folder = folder
    def __getitem__(self, index):
        
        id = self.list_ids[index]
        pth = os.path.join(self.folder,str(id))
        arr = np.load(pth).astype(np.float32)
        #arr[0] = 1-arr[0]
        x = torch.from_numpy(arr)
        #print(x.size())

        return x
    
    def __len__(self):

        return(len(self.list_ids))
    

#

class DiceBCELoss(nn.Module):
    
    def __inti__(self,weight = None,size_average = True):
        super(DiceBCELoss,self).__init__()
        
    def forward(self,inputs,targets,wc=torch.tensor([1.0,1.0])):
        
        m = nn.LogSoftmax(dim=1)
        inputs_2 = m(inputs)
        
        #inputs_2 = inputs_2.view(-1)
        #targets_2 = targets.view(-1)
        
        #intersection = (inputs_2*targets_2).sum()
        #dice_loss = 1 - (2.*intersection + smooth)/(inputs_2.sum()+targets_2.sum()+smooth)
        
        #m = nn.LogSoftmax(dim=1)
        BCE_loss = nn.NLLLoss(weight = wc).to('cuda:0')
        #inputs_3 = m(inputs)
        
        DICE_BCE = BCE_loss(inputs_2,targets)  #+ dice_loss
        
        return DICE_BCE
    
    
if __name__ == '__main__':

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    
    seed = 13
    torch.manual_seed(seed)
    # getting list of the files in training folder
    random.seed(13)
    curr_folder = os.getcwd()
    train_folder = os.path.join(curr_folder,'train_with_cu')
    files_ = os.listdir(train_folder)
    num_epochs = 50               
    model = Unet(1,1)
    model_path_ = "/home/UFAD/s.kapoor/GrainGrowth/skapoor/tem_data/model_5/weigths/6"
    model.load_state_dict(torch.load(model_path_))
    model.final_conv = nn.Conv2d(64,2,kernel_size = 1)
    model.to(device)
    loss_fn = DiceBCELoss()
    loss_fn.to(device)
    
    
    params = {'batch_size':8,
              'shuffle':True}
    
    params_2 = {'batch_size':1,
              'shuffle':False}
    
    params_optim = {'lr':1e-4,
                    'alpha':0.9,
                    'eps':1e-5,
                    'momentum':0.9}

    optimizer = torch.optim.RMSprop(model.parameters(),**params_optim)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer = optimizer,factor = 0.8,min_lr=1e-6)
    
    #m=nn.LogSoftmax(dim=1)
    #loss_fn.to(device)

    train_ids,val_ids = train_test_split(files_,test_size=0.25,random_state=seed)
    train_set = Data(train_ids,train_folder)
    valid_set = Data(val_ids,train_folder)
    train_generator = DataLoader(train_set,**params)
    valid_generator = DataLoader(valid_set,**params_2)

    #rot = [i for i in range(-180,180,15) if (i!=0 and i!=180 and i!= -180 and i!=90 and i!=-90)]

    best_v_loss = 1_000_000

    pth = os.getcwd()
    model_path = os.path.join(pth,'model_3_cu')
    log_dir = os.path.join(model_path,'runs')
    wts_path = os.path.join(model_path,'weigths')

    if os.path.exists(model_path) == False:
        os.mkdir(model_path)
        
        os.mkdir(log_dir)
        
        os.mkdir(wts_path)
    
    writer = SummaryWriter(log_dir=log_dir)

    for e in range(num_epochs):
        
        print('Epoch {0}'.format(e))
        epoch_loss = 0.0
        run_loss = 0.0
        model.train(True)

        for i,x in enumerate(train_generator):

            #print(x.size())
            x = x[0].to(device)
            #print(x.size())
            x = torch.unsqueeze(x,1)
            #print(x.size())
            v_img = transform.RandomVerticalFlip(p=1)(x)
            h_img = transform.RandomHorizontalFlip(p=1)(x)
            r_img = transform.RandomRotation(degrees=(-180,180))(x)
            ip = torch.cat((x[0],v_img[0],h_img[0],r_img[0]),0)
            ip = torch.unsqueeze(ip,1)

            op = torch.cat((x[1],v_img[1],h_img[1],r_img[1]),0)
            #op = torch.unsqueeze(op,1)
            op = op.long()

            optimizer.zero_grad()
            pred = model.forward(ip)
            #print(pred.size())
            #print(op.size())
            
            gg = op.view(-1)
            b = torch.sum(op).item() #class 1 edge pixels
            a = gg.size(dim = 0) - b #class 0 grain pixels.
            wc = torch.tensor([b/a,a/b])
            
            loss = loss_fn(pred,op,wc)
            loss.backward()

            optimizer.step()

            run_loss+=loss.item()
            epoch_loss+=loss.item()

            #print(run_loss)
            #break
            if (i % 1000 == 999):
                
                print("\tBatch Loss for curent for {0} is {1:.5f}".format(i,run_loss/1000))
                run_loss = 0.0
            
        avg_e_loss = epoch_loss/(i+1)
        print('The average loss for the epoch is {0}'.format(avg_e_loss))
        model.train(False)

        val_loss = 0.0
    
        for k,val in enumerate(valid_generator):
            
            val = val[0].to(device)
            #print(val.size())
            ip = torch.unsqueeze(torch.unsqueeze(val[0],0),0)
            op = torch.unsqueeze(val[1],0)
            op = op.long()
            pred = model.forward(ip)
            #print(pred.size())
            #print(op.size())
            loss = loss_fn(pred,op)

            val_loss+=loss.item()
            #break
        
        avg_val_loss = val_loss/(k+1)
        print('The average validation loss for the epoch is {0:.5f}'.format(avg_val_loss))
        writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_e_loss, 'Validation' : avg_val_loss },
                    e)
        
        if avg_val_loss < best_v_loss:

            best_v_loss = avg_val_loss
            best_e = e
            wt_path = os.path.join(wts_path,str(e))
            

        writer.flush()

        #break

    torch.save(model.state_dict(), wt_path)
