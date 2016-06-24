################################################################
#               CODE TO FIND IMPORTANCE OF A LAYER             #
#                 WRT A FIXED NEURON IN LAST LAYER             #
#                   (Class specific importance)                #
################################################################

import caffe
import numpy as np
from PIL import Image
import glob
image_list = []
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans,vq
from numpy import random
# display plots in this notebook

# set display defaults
plt.rcParams['figure.figsize'] = (10, 10)        # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap




def get_gradients(imagepath,neuron_num):
	##Loading the model where 2nd caffemodel has weights after innerproduct layer is incorporated
	caffe.set_mode_cpu()
	net = caffe.Net('models/bvlc_reference_caffenet/deploy.prototxt',
	'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
	caffe.TEST)


	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
	transformer.set_transpose('data', (2,0,1))
	transformer.set_mean('data', np.load('caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
	transformer.set_raw_scale('data', 255) 
	transformer.set_channel_swap('data', (2,1,0))

	net.blobs['data'].reshape(1,3,227,227)
	net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(imagepath))
	net.blobs['label'].data[...]=281
	out = net.forward()                                                                       
	net.blobs['fc8'].diff[0,neuron_num] = 1                                                             
	#print("Predicted class is #{}.".format(out['prob'].argmax()))
	back = net.backward()
	j = back['data'].copy()

	df_dwi= net.layers[1].blobs[0].diff                           #Gradients obtained                    
	

	num=[]
	for i in range(0,df_dwi.shape[0]):
     		num.append(np.linalg.norm(df_dwi[i,:,:,:]))

        return num


grad = get_gradients('examples/images/cat.jpg',0)

##################someway to get all cat images ,there are around 1600 in the dataset

output_grad = np.zeros([1,96])
for filename in glob.glob('data/cats/*.jpg'): #assuming gif
      grad = get_gradients(filename,0)
      output_grad += grad
      
output_grad = output_grad/273        #importance of each of the 96 filters with respect to neuron_num

output_grad = (output_grad - output_grad.min())/(output_grad.max()-output_grad.min())

o = np.array(output_grad)
im = o.reshape(16,6)
plt.imshow(im); plt.axis('off'); plt.show()

        
print output_grad
