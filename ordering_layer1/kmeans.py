# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans,vq
from numpy import random
import sys
from matplotlib.mlab import PCA as p
import caffe
import os

# The caffe module needs to be on the Python path;
#  we'll add it here explicitly.
import sys
caffe_root = '../'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')

caffe.set_mode_cpu()


# set display defaults
plt.rcParams['figure.figsize'] = (10, 10)        # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap



model_def = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
model_weights = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print 'mean-subtracted values:', zip('BGR', mu)

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

# set the size of the input (we can skip this if we're happy
#  with the default; we can also change it later, e.g., for different batch sizes)
net.blobs['data'].reshape(50,        # batch size
                          3,         # 3-channel (BGR) images
                          227, 227)  # image size is 227x227

image = caffe.io.load_image(caffe_root + 'examples/images/cat.jpg')
transformed_image = transformer.preprocess('data', image)
#plt.imshow(image) ; plt.show()

# copy the image data into the memory allocated for the net
net.blobs['data'].data[...] = transformed_image

### perform classification
output = net.forward()

def rearrange(a):
      pca = p(a)
      Y = pca.Y[:,0]
      X = a
      sorted_a= [x for (y,x) in sorted(zip(Y,X))]
      return sorted_a

def  kmeansclustering(data, slice,k):
    data = (data - data.min()) / (data.max() - data.min())
    newimg=[] # will contain the 96 slices of slice which represent output of 1st layer filters
    newimg1 = []
    
    for i in range (0,96):	
    	newimg.append(data[i][:][slice])
    
    newimg=np.asarray(newimg) 
 
   
    random.seed(123)
    centroids,_ = kmeans(newimg, k, 1000) # k means
    idx,_ = vq(newimg,centroids)

    count =0
    prev_num = min(idx)
    # code to rearrange clusters one below another seperated by white line
    for i in range (0,96):

        cur_num = min(idx)
        if(cur_num!=prev_num):
            prev_num=cur_num
            print count # number of elements in that cluster
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
    indices=[] # to get indices of original image in rearranged image
    for i in range (0,96):  
        newimg.append(data[i][:][slice])
    
    newimg1=np.asarray(newimg)

    display = rearrange(newimg1) # display is new image with slices rearranged according to pca 

    newimg1=[] # to store  newimg (input slices) as list

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

    feat = net.blobs['conv1'].data[0] # conv1 output (96,55,55)
    data = feat
    data = (data - data.min())*255 / (data.max() - data.min()) #normalizing data
    np.save('data.npy',data)

    print sys.argv[0],sys.argv[1] 
    k = int(sys.argv[1])
    kmeansclustering(data,15,k)
    # principal_comp_anlysis(feat,15)

