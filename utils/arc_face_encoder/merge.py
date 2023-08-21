import json
import os
import numpy as np
from tqdm import tqdm

idp = '/home/gcpadmin/mounted_drive/FaceSwap/face_swap/utils/arc_face_encoder/ids'
ls = os.listdir(idp)

with open('/home/gcpadmin/mounted_drive/FaceSwap/face_swap/utils/arc_face_encoder/head_pose.json') as json_file:
    pose_dict = json.load(json_file)


for i in tqdm(ls):
    idn = np.load(idp+'/'+i)
    # print(idn.shape)
    pose = np.array(pose_dict[i.replace('npy', 'png')])/50
    idnc = np.concatenate((idn[0],pose),axis = 0)

    with open('/home/gcpadmin/mounted_drive/FaceSwap/face_swap/data/id/'+i, 'wb') as f:
        np.save(f, idnc)

