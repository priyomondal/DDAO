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


##############################################################################
"""args for models"""

args = {}
args['dim_h'] = 64          # factor controlling size of hidden layers
args['n_channel'] = 1       # number of channels in the input data 
args['n_z'] = 300 #600      # number of dimensions in latent space. 
args['sigma'] = 1.0         # variance in n_z
args['lambda'] = 0.01       # hyper param for weight of discriminator loss
args['lr'] = 0.0002         # learning rate for Adam optimizer .000
args['epochs'] = 1 #50      # how many epochs to run for
args['save'] = True         # save weights at each epoch of training if True
args['train'] = False       # train networks if True, else load networks from
args['temperature'] = 0.01  # temperature hyperparameter for supervised contrastive loss 
args['batch_size'] = 100    # batch size while training
args['k'] = 5               # number of nearest neighbours considered for sample generation
args['eta'] = 4             # imbalanced tuner hyperparameter eta

##############################################################################


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
    print("in imshow:::::",img.shape)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    f = plt.figure(figsize=(25, 6)) 
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    npimg = np.transpose(npimg, (1, 2, 0))
    plt.imsave('5553.png', npimg)
    plt.show()

t0 = time.time()

## create encoder model and decoder model
class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.n_channel = args['n_channel']
        self.dim_h = args['dim_h']
        self.n_z = args['n_z']
        
        # convolutional filters, work excellent with image data
        self.conv = nn.Sequential(
            nn.Conv2d(self.n_channel, self.dim_h, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.dim_h, self.dim_h * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.dim_h * 2, self.dim_h * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.dim_h * 4, self.dim_h * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 8), 
            nn.LeakyReLU(0.2, inplace=True) )
        
        # final layer is fully connected
        self.fc = nn.Linear(self.dim_h * (2 ** 3), self.n_z)
        

    def forward(self, x):
        x = self.conv(x)
        x = x.squeeze()
        x = self.fc(x)
        return x


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.n_channel = args['n_channel']
        self.dim_h = args['dim_h']
        self.n_z = args['n_z']

        # first layer is fully connected
        self.fc = nn.Sequential(
            nn.Linear(self.n_z, self.dim_h * 8 * 7 * 7),
            nn.ReLU())

        # deconvolutional filters, essentially inverse of convolutional filters
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(self.dim_h * 8, self.dim_h * 4, 4),
            nn.BatchNorm2d(self.dim_h * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h * 4, self.dim_h * 2, 4),
            nn.BatchNorm2d(self.dim_h * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h * 2, 1, 4, stride=2),
            nn.Tanh())

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.dim_h * 8, 7, 7)
        x = self.deconv(x)
        return x

##############################################################################

def biased_get_class1(c,dec_x,dec_y):
    xbeg = dec_x[dec_y == c]
    ybeg = dec_y[dec_y == c]
    return xbeg, ybeg


def gaussian(X, y):

    if len(X) > 100:
        n_comps = math.ceil(len(X)/100)
    else:
        n_comps = 1

    gm = GaussianMixture(n_components=n_comps, random_state=42).fit(X)
    y = gm.fit(X)
    return y, n_comps

def generate(xclass, yclass, n, i):


    gm, n_comps = gaussian(xclass,yclass)
    mean = gm.means_
    covariance = gm.covariances_
    a = np.random.multivariate_normal(mean[0], covariance[0], n)
    a = []
    for iter in range(n_comps):
        a.append(np.random.multivariate_normal(mean[iter], covariance[iter], math.ceil(n/n_comps)))
    a = np.vstack(a)
    index = np.random.permutation(n)
    a = a[index]
    a = a[:n,]
    print("GENERATED SAMPLES...",a.shape)
    return a, [i]*n, gm, n_comps


def distribution_calibration(query, base_means, base_cov, k,alpha=0.21):
    dist = []
    for i in range(len(base_means)):
        dist.append(np.linalg.norm(query-base_means[i]))
    index = np.argpartition(dist, k)[:k]
    mean = np.concatenate([np.array(base_means)[index], query[np.newaxis, :]])
    calibrated_mean = np.mean(mean, axis=0)
    calibrated_cov = np.mean(np.array(base_cov)[index], axis=0)+alpha

    return calibrated_mean, calibrated_cov

#############################################################################
np.printoptions(precision=5,suppress=True)

def knn(X,x,k):
    distances = sp.distance.cdist(X, np.expand_dims(x, axis=0)).squeeze()
    topk_idx = np.argsort(-distances)[::-1][:k]
    return topk_idx


num_workers = 0

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
