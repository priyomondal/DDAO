# -*- coding: utf-8 -*-
#gausscalib
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
from torch.utils.data import TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from sklearn.neural_network import MLPClassifier
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
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
args['gamma'] = 6             # imbalanced tuner hyperparameter eta

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

def plot_decision_regions_3class(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # Standardize the features
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)
    # Train the classifier
    classifier = MLPClassifier(alpha=0.7, max_iter=1000)
    classifier.fit(X_train_std, y_train)
    # Plot the training points
    plt.figure(figsize=(8, 6))
    scatter = sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette=sns.color_palette("Paired", 10), legend='full', marker='o', s=50)
    # Turn off top and right axis
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Remove legend and create a custom legend at the top
    handles, labels = scatter.get_legend_handles_labels()
    plt.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=5)

class CustomDataset(Dataset):
  def __init__(self, pred, y):
      self.data = []
      predictor = pred
      response = y
      for i in range(len(predictor)): 
        self.data.append([predictor[i],response[i]])
  def __len__(self):
      return len(self.data)
  def __getitem__(self, idx):
      data_instance, class_name = self.data[idx]
      return data_instance, class_name

#############################################################################
np.printoptions(precision=5,suppress=True)

def knn(X,x,k):
    distances = sp.distance.cdist(X, np.expand_dims(x, axis=0)).squeeze()
    topk_idx = np.argsort(-distances)[::-1][:k]
    return topk_idx

def gaussian(X, y):
    if len(X) > 160:
        n_comps = math.ceil(len(X)/160)
    else:
        n_comps = 1

    gm = GaussianMixture(n_components=n_comps, random_state=42).fit(X)
    y = gm.fit(X)
    
    return y, n_comps


def gen(xclass, yclass):
    gm, n_comps = gaussian(xclass,yclass)
    mean = gm.means_
    covariance = gm.covariances_
    return gm, n_comps

def nonzero_eigenvalues(matrix):
    # Calculate the eigenvalues
    eigenvalues = np.linalg.eigvals(matrix)
    # Filter out the zero eigenvalues
    nonzero_eigenvalues = eigenvalues[eigenvalues != 0]
    return nonzero_eigenvalues

def forces(mass,mass_comp, density_all,X,x,k):
    distances = sp.distance.cdist(X, np.expand_dims(x, axis=0)).squeeze()
    orig_indices = np.arange(len(distances))
    idx = np.where(distances == 0)[0]
    distances[idx] = -99
    force = np.array([np.sign(distances[iter]) * density_all[iter]*mass[iter]*mass_comp/pow(distances[iter],2) for iter in range(len(distances))])
    force_max_idx = np.argsort(force)[::-1]
    nz_idx = force_max_idx
    topk_idx = nz_idx[0:k+1]
    return distances, force, topk_idx, nz_idx

#path on the computer where the models are stored
modpth = './models/mnist'
device = 'cpu'

encf = []
decf = []
temperature = args['temperature']
k = args['k']
gamma = args['gamma']
batch_size = args['batch_size']

#classifier
for p in range(1):
    enc = modpth + '/encoder'+str(temperature)+'.pth'
    dec = modpth + '/decoder'+str(temperature)+'.pth'
    encf.append(enc)
    decf.append(dec)

dec_x = torch.load("data/mnist/images.pt")
dec_y = torch.load("data/mnist/labels.pt")
dec_x = dec_x.reshape(dec_x.shape[0],1,28,28)
classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

#generate some images 
train_on_gpu = torch.cuda.is_available()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
path_enc = encf[0]
path_dec = decf[0]
encoder = Encoder(args)
encoder.load_state_dict(torch.load(path_enc), strict=False)
encoder = encoder.to(device)
decoder = Decoder(args)
decoder.load_state_dict(torch.load(path_dec), strict=False)
decoder = decoder.to(device)
encoder.eval()
decoder.eval()
imbal = [4000, 2000, 1000, 750, 500, 350, 200, 100, 60, 40]

resx = []
resy = []
feature_x = []
feature_y = []
features_ours2_X = []
features_ours2_Y = []

for i in range(0,10):
    xclass, yclass = biased_get_class1(i,dec_x,dec_y)
    #encode xclass to feature space
    xclass = torch.Tensor(xclass)
    xclass = xclass.to(device)
    xclass = encoder(xclass)
    xclass = xclass.detach().cpu().numpy()
    n = imbal[0] - imbal[i]
    feature_x.append(xclass)
    feature_y.append(yclass)
    gm, n_comps = gen(xclass, yclass)
    mean = gm.means_
    covariance = gm.covariances_

    for i1 in range(len(covariance)):
        scale_factor = np.max(np.abs(covariance[i1]))
        covariance[i1] = covariance[i1] / scale_factor
        if np.linalg.det(covariance[i1]) == 0:
            covariance = np.linalg.pinv(covariance)

    xclass = torch.tensor(xclass).to(device)
    """to generate samples for resnet"""   
    ximg = decoder(xclass)
    ximn = ximg.detach().cpu().numpy()
    resx.append(ximn)

X = np.vstack(feature_x)
Y = np.hstack(feature_y)

#Step1: Performing the Gaussian Mixture Model
gamma = args['gamma']
min_elt_gauss = gamma * imbal[len(imbal)-1]
after_gauss_X = []
after_gauss_Y = []
mean_gauss = []
covariance_gauss = []
class_samples = []
flag = 0
k = args['k']
density_all = []
mass_all = []

max_elt_class = np.argmax(imbal)
max_elt_value = np.max(imbal)
I = np.identity(300)
I = 1*I
for i in range(0,10):
    xclass, yclass = biased_get_class1(i,X,Y)
    temp_mean = []
    temp_cov = []
    n = imbal[0] - imbal[i]
    if len(xclass) == max_elt_value:
        gm, n_comps = gen(xclass, yclass)

        mean = gm.means_
        covariance = gm.covariances_

        weights = gm.weights_
        mass = weights*len(xclass)

        threshold = 1e-2
        density = [1/(np.sqrt(np.linalg.norm(np.prod(nonzero_eigenvalues(covariance[i3])[np.abs(nonzero_eigenvalues(covariance[i3])) > threshold])))) for i3 in range(len(covariance))]
        temp = gm.weights_
        b = np.array(imbal[i]*temp).astype('int32')

        density_all.append(density)
        mass_all.append(mass)

        class_samples.append(b.tolist())
        mean_gauss.append(mean)
        covariance_gauss.append(covariance)

    elif (len(xclass) > min_elt_gauss) and (len(xclass) < max_elt_value):
        gm, n_comps = gen(xclass, yclass)

        mean = gm.means_
        covariance = gm.covariances_

        weights = gm.weights_
        mass = weights*len(xclass)      
        xclass1 = torch.tensor(xclass)
        membership = np.zeros(len(xclass))

        temp = gm.weights_
        threshold = 1e-2
        density = [1/(np.sqrt(np.linalg.norm(np.prod(nonzero_eigenvalues(covariance[i3])[np.abs(nonzero_eigenvalues(covariance[i3])) > threshold])))) for i3 in range(len(covariance))]
        density_all.append(density)
        mass_all.append(mass)

        mean_step = mean
        covariance_step = covariance
        density_step = np.hstack(density_all)
        gen_elt_per_comps = math.ceil((imbal[0]-imbal[i])/n_comps)
        gen_X = []
        gen_Y = []

        for iter in range(len(mean)):
            distances, force, idx,_ = forces(mass, mass[iter], density,mean,mean[iter],k)
            dist = distances[idx]
            inv_dist = np.array([1/pow(d,2) for d in dist]) 
            sum = np.zeros(mean.shape[1])
            sum = np.array([inv_dist[i2]*mean[idx[i2]] for i2 in range(len(idx))])
            sum = np.sum(sum,axis=0)
            weight = 1-(np.sum(inv_dist))
            new_mean = sum + weight*mean[iter]
            temp_cov = [inv_dist[p]*covariance[idx[p]] for p in range(len(idx))]
            sum_cov = np.sum(np.array(temp_cov), axis=0)
            new_covariance = sum_cov + weight * covariance[iter]
            gen_elts = np.random.multivariate_normal(new_mean, new_covariance, gen_elt_per_comps)
            gen_X.append(gen_elts)
            gen_Y.append(np.array([i]*gen_elt_per_comps))
        
        gen_X = np.vstack(gen_X)
        gen_Y = np.hstack(gen_Y).astype('int32')
        after_gauss_X.append(gen_X)
        after_gauss_Y.append(gen_Y)
        mean_gauss.append(mean)
        covariance_gauss.append(covariance)
        temp = gm.weights_
        b = np.array(imbal[i]*temp).astype('int32')
        class_samples.append(b.tolist())
    else:
        if flag == 0:
            mean_step = np.vstack(mean_gauss)
            covariance_step = np.vstack(covariance_gauss)
            class_samples = np.hstack(class_samples)
            density_step = np.hstack(density_all)
            mass_step = np.hstack(mass_all)
            class_samples = np.array([1/samp for samp in class_samples])
            flag = 1
        gen_elt_per_elt = math.ceil(imbal[0]/len(xclass))
        gen_X = []
        gen_Y = []
        gm = GaussianMixture(n_components=1, random_state=42).fit(xclass)
        mean = gm.means_
        mean = np.reshape(mean,(mean.shape[1]))
        covariance = gm.covariances_
        covariance = np.reshape(covariance,(covariance.shape[1],covariance.shape[1]))
        for x in xclass:
            distances, force, idx, _ = forces(mass_step,len(xclass), density_step,mean_step,x,k)
            dist = distances[idx]
            dense = density_step[idx]
            inv_dist = np.array([1/pow(d,2) for d in dist])
            sum = np.zeros(mean_step.shape[1])
            sum = np.array([inv_dist[i2]*mean_step[i2] for i2 in range(len(idx))])
            sum = np.sum(sum,axis=0)
            weight = 1-(np.sum(inv_dist))
            new_mean = sum+weight*x
            temp_cov = [inv_dist[iter]*covariance_step[idx[iter]] for iter in range(len(idx))]
            sum_cov = np.sum(np.array(temp_cov), axis=0)
            new_covariance = sum_cov + weight * covariance
            gen_elts = np.random.multivariate_normal(new_mean, new_covariance, gen_elt_per_elt)
            gen_X.append(gen_elts)
            gen_Y.append(np.array([i]*gen_elt_per_elt))
        gen_X = np.vstack(gen_X)
        gen_Y = np.hstack(gen_Y).astype('int32') 
        after_gauss_X.append(gen_X)
        after_gauss_Y.append(gen_Y)
after_gauss_X = np.vstack(after_gauss_X)
after_gauss_Y = np.hstack(after_gauss_Y).astype('int32') 

#Code for t-SNE plot
'''
tsne_x = np.vstack((after_gauss_X,X))
tsne_y = np.hstack((after_gauss_Y,Y))
PATH = "./results/mnist/"
model = TSNE(n_components=2, random_state=42)
tsne_data = model.fit_transform(tsne_x)
data_set = CustomDataset(tsne_data, tsne_y)
plot_decision_regions_3class(tsne_data,tsne_y)
plt.savefig(PATH+"rcs"+".jpg")
# plt.savefig(PATH+"deepsmote_ae"+".jpg")
plt.cla()
'''
path_enc = encf[0]
path_dec = decf[0]
encoder = Encoder(args)
encoder.load_state_dict(torch.load(path_enc), strict=False)
encoder = encoder.to(device)
decoder = Decoder(args)
decoder.load_state_dict(torch.load(path_dec), strict=False)
decoder = decoder.to(device)
encoder.eval()
decoder.eval()
X_train_new = after_gauss_X         
Y_train_new = after_gauss_Y         
X_train_new = torch.tensor(X_train_new)
Y_train_new = torch.tensor(Y_train_new)
X_train_new = X_train_new.to(device).float()

mnist_bal = TensorDataset(X_train_new,Y_train_new) 
num_workers = 0
train_loader = torch.utils.data.DataLoader(mnist_bal, batch_size=batch_size,shuffle=True,num_workers=num_workers)
img_new = []
label_new = []

for x,y in train_loader:
    X = x.cpu().numpy()
    Y = y.cpu().numpy()
    img = decoder(x)
    ximn = img.detach().cpu().numpy()
    img_new.append(ximn)
    label_new.append(y.cpu().numpy())

img_new = np.vstack(img_new)
label_new = np.hstack(label_new) 
combx = np.vstack((img_new,dec_x)) 
comby = np.hstack((label_new,dec_y)) 
combx = combx.reshape(combx.shape[0],1,28,28)
num_workers = 0
tensor_x = torch.Tensor(combx)
tensor_y = torch.tensor(comby,dtype=torch.long)
ar = [1 for i in range(10)]
ctr = 1
imbalance_train_X = []
imbalance_train_Y = []
imbal = [4000, 2000, 1000, 750, 500, 350, 200, 100, 60, 40]

PATH = "./data/mnist/supcon"+str(temperature)+"_"+str(k)+"_"+str(gamma)+"/"
if os.path.exists(PATH) == False:
        os.mkdir(PATH)

PATH1 = PATH + "./train/"
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

print("SAMPLES GENERATED")