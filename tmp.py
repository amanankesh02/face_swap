import torch
import numpy as np
from PIL import Image
# arr = torch.ones((512))
# np.save('data/id/0.npy', arr.numpy())
# # img = Image.open('/home/gcpadmin/mounted_drive/FaceSwap/face_swap/data/target/0.png').resize((512,512))
# # img.save('/home/gcpadmin/mounted_drive/FaceSwap/face_swap/data/target/0.png')

# # arr = np.array(img)[:,:,0]
# # img = Image.fromarray(arr)
# # img.save('/home/gcpadmin/mounted_drive/FaceSwap/face_swap/data/mask/0.png')

# print(arr.shape)
from datetime import datetime
now = datetime.now()
date_time = now.strftime("%m-%d-%Y-%H-%M-%S")
print("date and time:",date_time)
# print(time.time())