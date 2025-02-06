# -*- coding: utf-8 -*-
#gausscalib
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
from torch.utils.data import TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
# from tailcalib import tailcalib
from PIL import Image
import collections
import torch
import torch.nn as nn
import numpy as np
import scipy.spatial as sp
import time
from PIL import Image, ImageOps
from collections import Counter
from torchvision import utils
import torchvision
import random
import numpy
import math
import cv2
import os
import PIL

state = 42

def set_seeds(seed_value, use_cuda):
  np.random.seed(seed_value)  # cpu vars
  torch.manual_seed(seed_value)  # cpu  vars
  random.seed(seed_value)  # Python
  os.environ['PYTHONHASHSEED'] = str(seed_value) 
  if use_cuda:
      torch.cuda.manual_seed(seed_value)
      torch.cuda.manual_seed_all(seed_value)  # gpu vars
      torch.backends.cudnn.deterministic = True  # needed
      torch.backends.cudnn.benchmark = False


set_seeds(state, True)

# applying grayscale method
def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    f = plt.figure(figsize=(25, 6)) 
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    npimg = np.transpose(npimg, (1, 2, 0))
    plt.imsave('5553.png', npimg)
    plt.show()


combx = torch.load("images_val.pt")
comby = torch.load("labels_val.pt")
combx = np.array(combx)
comby = np.array(comby)


PATH = "./validation/"
if os.path.exists(PATH) == False:
        os.mkdir(PATH)

PATH1 = PATH + "./val/"
if os.path.exists(PATH1) == False:
    os.mkdir(PATH1)

ctr = 1
from collections import Counter
ar = [1 for i in range(10)]
comby = comby.astype("int32")

for i in range(combx.shape[0]):
    img = combx[i]
    img = img / 2 + 0.5     # unnormalize
    temp = comby[i]
    npimg = img.transpose(1,2,0)
    npimg = npimg[:, :, ::-1]
    npimg = npimg *255
    PATH2 = os.path.join(PATH1+str(temp)+"/")
    if os.path.exists(PATH2) == False:
        os.mkdir(PATH2)

    PATH3 = os.path.join(PATH2, '%05d.png' % (ar[temp],))
    cv2.imwrite(PATH3, cv2.cvtColor(npimg, cv2.COLOR_RGB2BGR))

    ar[temp] = ar[temp] + 1
    ctr += 1