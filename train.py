from networks import Generator
from discriminator import Discriminator
from arcface import Backbone
from dataset import CustomDataset
import os
import sys
import numpy as np
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch import autograd
import cv2

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from loss import AdvDLoss_fake, AdvDLoss_real,AdvGLoss,DR1Loss
from PIL import Image
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

time = datetime.now()
date_time = time.strftime("%m-%d-%Y-%H-%M")

#training dir
training_dir ='/home/gcpadmin/mounted_drive/FaceSwap/face_swap/training_runs'
checkpoint_dir = os.path.join(training_dir, f'{date_time}/checkpoints')
log_dir = os.path.join(training_dir, f'{date_time}/log')
tensorboard_runs = os.path.join(training_dir, f'{date_time}/runs')

##create dirs
os.makedirs(training_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True) 

#summary writer
writer = SummaryWriter(os.path.join(tensorboard_runs))

class Train:
    def __init__(self, data_dir='', epochs=100, training_dir='./training_runs', checkpoint_path=None, load_ckpt_step=None, save_step=5000):
        self.G = Generator(512, 512, 4)
        self.D = Discriminator(512)
        
        self.dataset = CustomDataset(data_dir)
        
        self.epochs = epochs
        self.lr = 0.001
        
        self.batch_size = 4
        self.d_every_step = 2
        self.d_reg_every = 16
        
        self.global_step = 0
        self.save_step = save_step
        self.log_step = 500
        
        self.lambda_g_adv = 0.1
        self.lambda_id = 100
        self.lambda_g_mse = 0
        
        self.training_dir = training_dir
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
    
        ##resume
        self.load_ckpt_step = load_ckpt_step
        self.checkpoint_path = checkpoint_path
        
        if self.checkpoint_path:
            self.load_checkpoint()    
        
        ##Loss
        self.adv_d_loss_real=AdvDLoss_real()
        self.adv_d_loss_fake=AdvDLoss_fake()
        self.adv_g_loss=AdvGLoss()
        self.d_r1_reg_loss=DR1Loss()
       
        
        self.mseloss = nn.MSELoss()
        self.arcface = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')

    def IdLoss(self, im, src_id):
        
        # 131:291, 176:337
        im=F.interpolate(im[:,:,131:291, 177:337], (112,112))
        gen_id = self.arcface(im)
        id_loss = self.mseloss(gen_id, src_id) 
        
        return id_loss
    
    def setup(self, rank, world_size):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        # initialize the process group
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

    def cleanup(self):
        dist.destroy_process_group()
    
    def run(self, train, world_size):
        mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

    def load_checkpoint(self):
        print('loading checkpoints ...')
        if self.load_ckpt_step is None:
            G_ckpt_pth = os.path.join(self.checkpoint_path, f'gen_{self.global_step}.checkpoint')
            D_ckpt_pth = os.path.join(self.checkpoint_path, f'dis_{self.global_step}.checkpoint')
        else:
            G_ckpt_pth = os.path.join(self.checkpoint_path, f'gen_{self.load_ckpt_step}.checkpoint')
            D_ckpt_pth = os.path.join(self.checkpoint_path, f'dis_{self.load_ckpt_step}.checkpoint')
        
        self.G.load_state_dict(self.remove_module_prefix(torch.load(G_ckpt_pth)))
        self.D.load_state_dict(self.remove_module_prefix(torch.load(D_ckpt_pth)))
        print('checkpoint loaded')
        
    def save_checkpoints(self):
        print('saving checkpoints ...')
        G_ckpt_pth = os.path.join(self.checkpoint_dir, f'gen_{self.global_step}.checkpoint')
        D_ckpt_pth = os.path.join(self.checkpoint_dir, f'dis_{self.global_step}.checkpoint')
        torch.save(self.G.state_dict(), G_ckpt_pth)
        torch.save(self.D.state_dict(), D_ckpt_pth)
        print('checkpoint saved')
        
    def remove_module_prefix(self, state_dict,prefix ='module.'):
        new_state_dict={}
        
        for k,v in state_dict.items():
            new_key=k.replace(prefix,"",1)
            new_state_dict[new_key]=v

        return new_state_dict

    def log_images(self, im, target, mask, r_img=None):
        # print(im.shape, type(im))
        log_path = os.path.join(self.log_dir, f'{self.global_step}.png')
        
        im = im.detach().cpu().numpy()[0]
        mask = mask.detach().cpu().numpy()[0]
        target = target.detach().cpu().numpy()[0]
        mask = (mask*255).astype(np.uint8)[0]
        
        im = ((im*127.5)+127.5).astype(np.uint8)
        target = ((target*127.5)+127.5).astype(np.uint8)
        
        ##permute
        im = np.transpose(im, (1, 2, 0))
        target = np.transpose(target, (1, 2, 0))
        mask = cv2.merge((mask, mask, mask))

        if r_img is not None:
            r_img = r_img.numpy()[0]
            r_img = ((r_img*127.5)+127.5).astype(np.uint8)
            r_img = np.transpose(r_img, (1, 2, 0))
            res = np.concatenate((im, mask, target, r_img), axis = 1)
            
        else:
            res = np.concatenate((im, mask, target), axis = 1)
        
        img = Image.fromarray(res)
        img.save(log_path)
    
    def train(self, rank, world_size):
        print(f"Running on rank {rank}.")
        self.setup(rank, world_size)

        # create model and move it to GPU with id rank
        self.G = self.G.to(rank)
        self.G = DDP(self.G, device_ids=[rank], find_unused_parameters=True)
        
        self.D = self.D.to(rank)
        self.D = DDP(self.D, device_ids=[rank], find_unused_parameters=True)
        
        self.optim_G = torch.optim.Adam(self.G.parameters())
        self.optim_D = torch.optim.Adam(self.D.parameters())
        
        ##DataLoader
        dataloader = DataLoader(self.dataset, batch_size = self.batch_size, shuffle=False)
        
        ##Archface
        self.arcface.load_state_dict(torch.load('/home/gcpadmin/workplace_eth_change/pretrained_ckpts/auxiliray/model_ir_se50.pth'))
        self.arcface = self.arcface.to(rank).eval()
        self.arcface = DDP(self.arcface, device_ids=[rank])
        
        rflag = False
        ##training_loop
        for epoch in range(0, self.epochs):
            for src_id, tgt_mask, tgt_img, r_id, r_img in dataloader:
                
                loss_dict={}
                rn = np.random.rand()
                
                if rn < 0.25:
                    rflag = True
                    src_id = r_id.to(rank).float()
                else:
                    rflag = False
                    src_id = src_id.to(rank).float()
                
                ## moving to gpu                  
                # src_id = src_id.to(rank).float()
                tgt_mask = tgt_mask.to(rank).float()
                tgt_img = tgt_img.to(rank).float()
            
                ##Discriminator step
                if self.global_step % self.d_every_step == 0:
                    
                    ## real images 
                    self.optim_D.zero_grad()
                    # tgt_img.requires_grad=True
                    # real_pred = self.D(tgt_img)
                    
                    # d_loss_r1_real = self.d_r1_reg_loss(real_pred, tgt_img)
                    # loss_dict['d_loss_r1_real'] = d_loss_r1_real.item()
                    
                    real_pred = self.D(tgt_img)
                    d_loss_adv_real = self.adv_d_loss_real(real_pred)
                    loss_dict['d_loss_adv_real'] = d_loss_adv_real.item()
                    
                    d_loss_real = d_loss_adv_real #+ d_loss_r1_real
                    loss_dict['d_loss_real'] = d_loss_real.item()
                    
                    d_loss_real.backward()
                    self.optim_D.step()
                    
                    ## fake images
                    self.optim_D.zero_grad()
                    
                    with torch.no_grad():
                        gen_img = self.G(src_id, tgt_mask, (1-tgt_mask)*tgt_img)
                        gen_img = (tgt_mask*gen_img) + ((1-tgt_mask)*tgt_img)

                    fake_pred = self.D(gen_img)
                    
                    # d_loss_r1_fake = self.d_r1_reg_loss(fake_pred, gen_img)
                    # loss_dict['d_loss_r1_fake'] = d_loss_r1_fake.item()
                    
                    d_loss_adv_fake = self.adv_d_loss_fake(fake_pred)
                    loss_dict['d_loss_adv_fake'] = d_loss_adv_fake.item()

                    d_loss_fake = d_loss_adv_fake #+ d_loss_r1_fake
                    loss_dict['d_loss_fake'] = d_loss_fake.item()
                    
                    d_loss_fake.backward()
                    self.optim_D.step()
                    
                    if rank==0:
                        writer.add_scalar("Loss/d_loss_real", loss_dict['d_loss_real'], self.global_step)
                        writer.add_scalar("Loss/d_loss_fake", loss_dict['d_loss_fake'], self.global_step)
                    
                ##Generator step
                self.optim_G.zero_grad()        
                im = self.G(src_id, tgt_mask, (1-tgt_mask)*tgt_img)
                im = (tgt_mask*im) + ((1-tgt_mask)*tgt_img)
                pred = self.D(im)
                
                g_loss_adv = self.adv_g_loss(pred)
                g_loss_id = self.IdLoss(im, src_id[:,0:512])
                
                
                g_loss_mse = self.mseloss(im, tgt_img)
                
                loss_dict['g_loss_adv'] = g_loss_adv.item()
                loss_dict['g_loss_id'] = g_loss_id.item()
                loss_dict['g_loss_mse'] = g_loss_mse.item()
                
                g_loss = self.lambda_id*g_loss_id + self.lambda_g_adv*g_loss_adv #+ self.lambda_g_mse*g_loss_mse
                loss_dict['g_loss'] = g_loss.item()
                
                if rank==0:
                    writer.add_scalar("Loss/g_loss_adv", loss_dict['g_loss_adv'], self.global_step)
                    writer.add_scalar("Loss/g_loss_id", loss_dict['g_loss_id'], self.global_step)
                    writer.add_scalar("Loss/g_loss_mse", loss_dict['g_loss_mse'], self.global_step)
                    writer.add_scalar("Loss/g_loss", g_loss.item(), self.global_step)
                
                g_loss.backward()
                self.optim_G.step()
                
                if rank==0 and self.global_step%self.log_step==0 and self.global_step<2000 and self.global_step%100 == 0:
                    if rflag:
                        self.log_images(im, tgt_img, tgt_mask, r_img)
                    else:
                        self.log_images(im, tgt_img, tgt_mask)
                    # self.log_loss(loss_dict)
                    
                if rank==0:
                    print(f"{self.global_step} : {epoch}/{self.epochs}")
                    print(loss_dict)
                    print()
                    
                self.global_step+=1
                
                if rank==0 and self.global_step%self.save_step==0:
                    self.save_checkpoints()
            
        self.cleanup()

if __name__=="__main__":
    
    data_dir = '/home/gcpadmin/mounted_drive/FaceSwap/face_swap/data'  
    epochs = 100
    checkpoint_path = '/home/gcpadmin/mounted_drive/FaceSwap/face_swap/training_runs/08-21-2023-08-01/checkpoints/'
    load_ckpt_step = 10000

    print(data_dir)
    
    T = Train(data_dir=data_dir, epochs = epochs, training_dir=training_dir, checkpoint_path=checkpoint_path, load_ckpt_step = load_ckpt_step)
   
    T.run(T.train, 2)