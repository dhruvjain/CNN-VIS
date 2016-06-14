# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans,vq
from numpy import random
# display plots in this notebook

# set display defaults
plt.rcParams['figure.figsize'] = (10, 10)        # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap

# The caffe module needs to be on the Python path;
#  we'll add it here explicitly.
import sys
caffe_root = '../'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')

import caffe
# If you get "No module named _caffe", either you have not built pycaffe or you have the wrong path.

import os

caffe.set_mode_cpu()

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

output_prob = output['prob'][0]  # the output probability vector for the first image in the batch

print 'predicted class is:', output_prob.argmax()

# load ImageNet labels
labels_file = caffe_root + 'data/ilsvrc12/synset_words.txt'
    
labels = np.loadtxt(labels_file, str, delimiter='\t')

print 'output label:', labels[output_prob.argmax()]

# sort top five predictions from softmax output
top_inds = output_prob.argsort()[::-1][:5]  # reverse sort and take five largest items

print 'probabilities and labels:'
zip(output_prob[top_inds], labels[top_inds])

# for each layer, show the output shape
for layer_name, blob in net.blobs.iteritems():
    print layer_name + '\t' + str(blob.data.shape)

for layer_name, param in net.params.iteritems():
    print layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape)

def vis_square(data, slice):
    data = (data - data.min()) / (data.max() - data.min())
    newimg=[]
    newimg1=[]
    mindiff=[10 for x in range(96)]
    minindx=[100 for x in range(96)]
    diff=[[10 for x in range(96)] for y in range(96)]
    
    for i in range (0,96):	
    	newimg.append(data[i][:][slice])
    
    #plt.imshow(newimg); plt.axis('off'); plt.show()
    
    # for i in range (0,95):
    # 	for j in range(i+1,96):
    # 		diff[i][j] = sum((newimg[i]-newimg[j])**2)
     	
    # for i in range (0,95):
    # 	mindiff[i] = min(diff[i])
    # 	minindx[i] = np.argmin(diff[i])
    
    # for i in range (0,95):
    # 	temp = newimg[np.argmin(mindiff) + 1]
    # 	newimg[np.argmin(mindiff) + 1] = newimg[minindx[np.argmin(mindiff)]]
    # 	newimg[minindx[np.argmin(mindiff)]] = temp
    # 	temp = mindiff[np.argmin(mindiff) + 1]
    # 	mindiff[np.argmin(mindiff) + 1] = mindiff[minindx[np.argmin(mindiff)]]
    # 	mindiff[minindx[np.argmin(mindiff)]] = temp
    # 	mindiff[np.argmin(mindiff)] = 10

    random.seed(123)
    centroids,_ = kmeans(newimg, 10, 1000)
    idx,_ = vq(newimg,centroids)


    for i in range (0,96):
        print np.argmin(idx)
    	newimg1.append(newimg[np.argmin(idx)])
    	idx[np.argmin(idx)] = 10

    plt.imshow(newimg1); plt.axis('off'); plt.show()

feat = net.blobs['conv1'].data[0]
vis_square(feat,15)

