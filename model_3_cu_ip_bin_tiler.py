#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 13:26:57 2023

@author: s.kapoor
"""
#some_edits
import torch
from torch.utils.data import DataLoader,Dataset
import numpy as np
import os
from sklearn.model_selection import train_test_split
from model import Unet
from tqdm import tqdm
import random
import torchvision.transforms.functional as F
import torchvision.transforms as transform
from datetime import datetime
#from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from tiler import Tiler, Merger
from PIL import Image
import cv2
from skimage.morphology import skeletonize

def func(pred):
    pred = np.array(pred,np.uint8)
    kernel_dil = np.ones((3,3),np.uint8)
    dilation = cv2.dilate(pred,kernel_dil,iterations = 3)
    skeleton = skeletonize(dilation)
    return skeleton



class Data(Dataset):

    def __init__(self,list_ids,folder):
        self.list_ids = list_ids
        self.folder = folder
    def __getitem__(self, index):
        
        id = self.list_ids[index]
        print(id)
        pth = os.path.join(self.folder,str(id))
        arr = np.load(pth).astype(np.float32)
        #arr[0] = 1-arr[0]
        x = torch.from_numpy(arr)
        #print(x.size())

        return x

    def __len__(self):

        return(len(self.list_ids))
    
if __name__ == '__main__':
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:2" if use_cuda else "cpu")
    seed = 11
    torch.manual_seed(seed)
    # getting list of the files in training folder
    random.seed(76)
    curr_folder = os.getcwd()
    model = Unet(1,2)
    model_path_ = "model_3_cu/weigths/24"
    model_dir = os.path.join(curr_folder,model_path_)
    model.load_state_dict(torch.load(model_dir,map_location=device))
    model.eval()

    train_file = os.path.join(curr_folder,'gb_cropped750.jpg')
    img = Image.open(train_file).convert("L")
    np_img = np.array(img)/255
    r,c = np_img.shape

    # tiler = Tiler(data_shape = (r,c),
    #          tile_shape = (256,256),
    #          overlap = 0.4,
    #          mode = 'reflect')
    
    tiler = Tiler(data_shape = (r,c),
            tile_shape = (256,256),
            overlap=0.25,
            mode = 'reflect')  
    
    merger = Merger(tiler)

    #new_shape, padding = tiler.calculate_padding()
    #tiler.recalculate(data_shape=new_shape)
    #padded_data = np.pad(np_img,pad_width=padding,mode='reflect')

    for tile_id,tile in tiler(np_img):
        
        val = np.array(tile,np.float32)
        val = torch.from_numpy(val)
        ip = torch.unsqueeze(torch.unsqueeze(val, 0),0).to(device)
        #print(ip.size())
        pred = model.forward(ip)
        pred = torch.squeeze(pred)
        pred = pred.detach().cpu().argmax(0).numpy()
        #adding another preprocessing function before merging
        pred = func(pred)
        merger.add(tile_id,pred)

    final_img = merger.merge()
    save_path = os.path.join(curr_folder,'final_output_tile_preprocess.npy')
    np.save(save_path,final_img)
    del(model)
        
        
        
        
    
    
    