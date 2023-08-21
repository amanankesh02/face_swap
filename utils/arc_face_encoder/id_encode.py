import torch
from PIL import Image
import os
import numpy as np
from arcface import Backbone
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm

facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se').to('cuda')
facenet.load_state_dict(torch.load('model_ir_se50.pth'), )
facenet.eval()

T = transforms.Compose([ transforms.ToTensor()#,
    # transforms.Normalize((127.5, 127.5, 127.5), (127.5, 127.5, 127.5))
])

ls = os.listdir('/home/gcpadmin/mounted/res_image_done')
for i in tqdm(ls):
    img = Image.open('/home/gcpadmin/mounted/res_image_done/'+i)
    img = T(img)
    img = img[:,131:291, 176:337]

    img = Imgae.fromarray(img.numpy())

    img.save('')


    
    # img = img.unsqueeze(0).to('cuda')
    # print(img.size())
    # img = F.interpolate(img, (112,112))
    # ops = facenet(img).detach().cpu().numpy()
    # with open('ids/'+i.replace('png', 'npy'), 'wb') as f:
    #     np.save(f,ops)
    # print(i)