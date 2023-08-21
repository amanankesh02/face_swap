import cv2
import numpy as np
import os
from tqdm import tqdm

mskp = '/home/gcpadmin/mounted_drive/FaceSwap/face_swap/res_img_cihp/cihp_parsing_maps'
mpth = '/home/gcpadmin/mounted_drive/FaceSwap/face_swap/data/mask'

ls = os.listdir(mskp)

for i in tqdm(ls):
    mks = cv2.imread(mskp+'/'+i, cv2.IMREAD_GRAYSCALE)
    bin_msk = np.where(np.logical_or(mks==2, np.logical_or(mks==10, mks==13)), 1, 0).astype(np.uint8)
    cv2.imwrite('/home/gcpadmin/mounted_drive/FaceSwap/face_swap/data/mask/'+i, bin_msk)

