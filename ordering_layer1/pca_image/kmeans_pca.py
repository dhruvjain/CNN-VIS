# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans,vq
from numpy import random
import sys
from matplotlib.mlab import PCA as p
# display plots in this notebook

# set display defaults
plt.rcParams['figure.figsize'] = (10, 10)        # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap


import os


def rearrange(a):
      pca = p(a)
      Y = pca.Y[:,0]
      X = a
      sorted_a= [x for (y,x) in sorted(zip(Y,X))]
      return sorted_a

def  kmeansclustering(data, slice,k):
    data = (data - data.min()) / (data.max() - data.min())
    newimg=[]
    newimg1 = []
    
    for i in range (0,96):	
    	newimg.append(data[i][:][slice])
    
    newimg=np.asarray(newimg) 
    random.seed(123)
    centroids,_ = kmeans(newimg, k, 1000)
    idx,_ = vq(newimg,centroids)

    count =0
    prev_num = min(idx)
    for i in range (0,96):

        cur_num = min(idx)
        if(cur_num!=prev_num):
            prev_num=cur_num
            print count
            count = 0
            newimg1.append([1]*55)
        else:
            count = count +1
            newimg1.append(newimg[np.argmin(idx)])

    	idx[np.argmin(idx)] = k
    
    plt.imshow(newimg1); plt.axis('off'); plt.show()

def  principal_comp_anlysis(data, slice):
    data = (data - data.min()) / (data.max() - data.min())
    newimg=[]
    indices=[]
    for i in range (0,96):  
        newimg.append(data[i][:][slice])
    
    newimg1=np.asarray(newimg)

    display = rearrange(newimg1)

    newimg1=[]

    for elet in newimg:
        elet=elet.tolist()
        newimg1.append(elet)

    for elet in display:
        elet=elet.tolist()
        index = [i for i, x in enumerate(newimg1) if x == elet]
        indices.append(index[0])        
    
    plt.imshow(display); plt.axis('off'); plt.show()
    print indices
    



if __name__ == '__main__':

    data=np.load('data.npy')
    print data.shape
    # feat = net.blobs['conv1'].data[0]
    # print sys.argv[0],sys.argv[1]
    k = int(sys.argv[1])
    # principal_comp_anlysis(data,15)
    kmeansclustering(data,15,k);
