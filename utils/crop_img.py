import os
import cv2
import numpy as np
from tqdm import tqdm

ls = os.listdir('/home/gcpadmin/mounted_drive/FaceSwap/face_swap/data/target')

for i in tqdm(ls):
    img = cv2.imread('/home/gcpadmin/mounted_drive/FaceSwap/face_swap/data/target/'+i)
    img  = img[131:291, 176:337, :]
    cv2.imwrite('/home/gcpadmin/mounted_drive/FaceSwap/face_swap/data/face_crop/'+i, img)