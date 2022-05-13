import albumentations
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
from torch import Tensor
from torch.nn import Parameter
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import json
from human_motion_analysis_with_gru import MMD_NCA_Net
import time
import matplotlib.pyplot as plt
from torchvision import datasets
from PIL import Image

use_cuda = torch.cuda.is_available()
torch.cuda.is_available()
torch.cuda.current_device()
torch.cuda.get_device_name(0)
torch.cuda.memory_allocated()
torch.cuda.memory_cached()

# initialize a model with the same architecture as the model which parameters you saved into the .pt/h file
model = MMD_NCA_Net()

# load the parameters into the model
model.load_state_dict(torch.load("log/model_new_999.pth"))  # load

cap = cv2.VideoCapture(0)
if (cap.isOpened() == False):
    print('Error while trying to read video. Plese check again...')
# get the frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
# define codec and create VideoWriter object
#out = cv2.VideoWriter(0, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width,frame_height))

aug = albumentations.Compose([
    albumentations.Resize(34, 34),
    ])
# read until end of video
while(cap.isOpened()):
    # capture each frame of the video
    ret, frame = cap.read()
    if ret == True:
        model.eval()
        with torch.no_grad():
            # conver to PIL RGB format before predictions
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pil_image = aug(image=np.array(pil_image))['image']
            pil_image = torch.Tensor(pil_image).permute(2,0,1)

            #pil_image = np.transpose(pil_image, (0,2,1)).astype(np.float32)
            #pil_image = torch.tensor(pil_image, dtype=torch.float).cuda()
            #pil_image = pil_image.unsqueeze(0)
            outputs = model(pil_image)
            _, preds = torch.max(outputs.data, 1)

        cv2.putText(frame, lb.classes_[preds], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 2)
        cv2.imshow('image', frame)
        out.write(frame)
        # press `q` to exit
        if cv2.waitKey(27) & 0xFF == ord('q'):
            break
    else:
        break
# release VideoCapture()
cap.release()
# close all frames and video windows
cv2.destroyAllWindows()