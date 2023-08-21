from sixdrepnet import SixDRepNet
import cv2
import os
import json
from tqdm import tqdm
import numpy as np


# Create model
# Weights are automatically downloaded
model = SixDRepNet()
ls = os.listdir('/home/gcpadmin/mounted/res_image_done')

pose_dict = {}

for i in tqdm(ls):
    img = cv2.imread('/home/gcpadmin/mounted/res_image_done/'+i)
    pitch, yaw, roll = model.predict(img)
    # print(i)
    pose_dict[i]=[float(yaw[0]),float(pitch[0]),float(roll[0])]
    # print(roll, yaw)
    # break


with open('head_pose.json', 'w') as f:
    json.dump(pose_dict,f)