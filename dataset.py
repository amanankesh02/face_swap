import os
import numpy as np

from torchvision.io import read_image
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms


tgt_transform = transforms.Compose([
    transforms.Normalize((127.5, 127.5, 127.5), (127.5, 127.5, 127.5))
])

mask_transform = transforms.Compose([
    transforms.Normalize((127.5), (127.5))
])

class CustomDataset(Dataset):
    
    def __init__(self, dir, tgt_transform=None, mask_transform=None):    
        
        self.Id_dir = os.path.join(dir, 'id')
        self.Id_list = os.listdir(self.Id_dir)
        
        self.mask_dir = os.path.join(dir, 'mask')
        self.mask_list = [el.replace('.npy', '.png') for el in self.Id_list] #os.listdir(self.mask_dir)
        
        self.tgt_dir = os.path.join(dir, 'target')
        self.tgt_list = [el.replace('.npy', '.png') for el in self.Id_list]  #os.listdir(self.tgt_dir)
        
        self.l = len(self.Id_list)
        
        self.mask_transform = mask_transform
        
        self.tgt_transform = transforms.Compose([
            transforms.Normalize((127.5, 127.5, 127.5), (127.5, 127.5, 127.5))
        ])

    def __len__(self):
        return self.l

    def __getitem__(self, idx):
        
        r_idx = np.random.randint(0, self.l)
        # print(r_idx)
        r_id_path = os.path.join(self.Id_dir, self.Id_list[r_idx])
        r_img_path = os.path.join(self.tgt_dir, self.tgt_list[r_idx])
        
        id_path = os.path.join(self.Id_dir, self.Id_list[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_list[idx])
        tgt_path = os.path.join(self.tgt_dir, self.tgt_list[idx])
        
        r_id = torch.from_numpy(np.load(r_id_path)).float()
        r_img = read_image(r_img_path).float()
        
        id = torch.from_numpy(np.load(id_path)).float()
        mask = read_image(mask_path).float()
        tgt = read_image(tgt_path).float()
        
        if self.tgt_transform is not None:
            tgt = tgt_transform(tgt)
            r_img = tgt_transform(r_img)
            
        if self.mask_transform is not None:
            mask = mask_transform(mask)
        
        return id, mask, tgt, r_id, r_img
    
if __name__=="__main__":
    
    dir = '/home/gcpadmin/mounted_drive/FaceSwap/face_swap/data'
    dataset = CustomDataset(dir, tgt_transform, mask_transform)
 
    for id, mask, tgt, r_id, r_img in dataset:
        print(id.shape, mask.shape, tgt.shape, r_id.shape, r_img.shape, mask[0,:2,:2], tgt[0,:2, :2])
        break
    
    print(np.random.rand())