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
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


class Data(Dataset):

    def __init__(self,list_ids,folder):
        self.list_ids = list_ids
        self.folder = folder
    def __getitem__(self, index):
        
        id = self.list_ids[index]
        pth = os.path.join(self.folder,str(id))
        arr = np.load(pth).astype(np.float32)
        arr[0] = 1-arr[0]
        x = torch.from_numpy(arr)
        #print(x.size())

        return x
    
    def __len__(self):

        return(len(self.list_ids))
    

class Wtd_Bce_Loss(torch.nn.Module):

    def __init__(self) -> None:
        super(Wtd_Bce_Loss,self).__init__()

    def forward(self,pred,target):

        """
        Here in loss each individual is multiplied by their respective beta from the given image.
        Each batch has different beta
        In general beta = no. of non-edge pixels/total
        in our case it is no.of edge/totals
        and we will use it in opposite way of general weighted BCE Loss
        
        """
        assert pred.size() == target.size(),"Different size of pred and target"
        
        pred = torch.sigmoid(torch.flatten(pred,1,-1))
        target = torch.flatten(target,1,-1)
        
        pred_2 = torch.log(torch.where(target>0,pred,1-pred))

        b,p = pred_2.size()
        
        beta = torch.matmul(target,torch.ones(p,1)/p).broadcast_to(b,p)
        
        target_2 = torch.where(target>0,1-beta,beta) # targets multiplied by respective betas. 
        
        loss = -1*torch.sum(torch.mul(pred_2,target_2))/b
        
        return loss
    
if __name__ == '__main__':

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    
    seed = 13
    torch.manual_seed(seed)
    # getting list of the files in training folder
    random.seed(13)
    curr_folder = os.getcwd()
    train_folder = os.path.join(curr_folder,'train')
    files_ = os.listdir(train_folder)
    num_epochs = 10
    model = Unet(1,1)
    #use wts such that wts  = [non-edge/total for edge and edge/total for non-edge]
    # measures used are OIS and ODS with F score will determine later. Will check this code for now
    # 
    
    params = {'batch_size':1,
              'shuffle':True}
    
    params_2 = {'batch_size':1,
              'shuffle':False}
    
    params_optim = {'lr':1e-4,
                    'alpha':0.9,
                    'eps':1e-5,
                    'momentum':0.9}

    optimizer = torch.optim.RMSprop(model.parameters(),**params_optim)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer = optimizer,factor = 0.8,min_lr=1e-6)
    loss_fn = Wtd_Bce_Loss()

    train_ids,val_ids = train_test_split(files_,test_size=0.25,random_state=seed)
    train_set = Data(train_ids,train_folder)
    valid_set = Data(val_ids,train_folder)
    train_generator = DataLoader(train_set,**params)
    valid_generator = DataLoader(valid_set,**params_2)

    #rot = [i for i in range(-180,180,15) if (i!=0 and i!=180 and i!= -180 and i!=90 and i!=-90)]

    best_v_loss = 1_000_000

    pth = os.getcwd()
    model_path = os.path.join(pth,'model_1')
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
            op = torch.unsqueeze(op,1)

            optimizer.zero_grad()
            pred = model.forward(ip)

            loss = loss_fn(pred,op)
            loss.backward()

            optimizer.step()

            run_loss+=loss.item()
            epoch_loss+=loss.item()

            #print(run_loss)
            #break
            if (i % 100 == 99):
                
                print("\tBatch Loss for curent for {0} is {1:.5f}".format(i,run_loss/100))
                run_loss = 0.0
            
        avg_e_loss = epoch_loss/(i+1)
        print('The average loss for the epoch is {0}'.format(avg_e_loss))
        model.train(False)

        val_loss = 0.0
    
        for k,val in enumerate(valid_generator):
            
            val = val[0].to(device)
            #print(val.size())
            ip = torch.unsqueeze(torch.unsqueeze(val[0],0),0)
            op = torch.unsqueeze(torch.unsqueeze(val[1],0),0)

            pred = model.forward(ip)
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


    




            


             






            

            

