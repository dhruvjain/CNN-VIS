import numpy as np
import matplotlib.pyplot as plt
import caffe
from scipy import ndimage
from scipy import misc
from array import array
import struct
def binary(num):
    return ''.join(bin(ord(c)).replace('0b', '').rjust(8, '0') for c in struct.pack('!f', num))


def vis_square(data,fname,shape_info):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""

    # normalize data for display
    # print data.max()
    data = (data - data.min())*255 / (data.max() - data.min())
    data = np.rint(data)
    print "done ",fname
    # nr_channels = data.shape[0]
    # image_width = data.shape[1];
    # image_height = data.shape[2];
    output_file = open(fname+'.vox', 'wb')
    output_file2 = open(fname+'.hd', 'w')

    arr = np.int8(data)
    output_file.write(arr)
    output_file.close()

    shape_info
    line1 = 'Size '+ str(shape_info[2]) +'x'+ str(shape_info[3]) +'x'+ str(shape_info[1]) +' unsigned byte \n'
    line2 = 'Spacing 1.0x1.0x1.0'
    output_file2.write(line1)
    output_file2.write(line2)
    output_file2.close()



if __name__ == '__main__':

	plt.rcParams['figure.figsize'] = (10, 10)        # large images
	plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
	plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap


	caffe_root='../'
	caffe.set_mode_cpu()
	model_def = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
	model_weights = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
	net = caffe.Net(model_def,      # defines the structure of the model
	                model_weights,  # contains the trained weights
	                caffe.TEST)     # use test mode (e.g., don't perform dropout)

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


	# copy the image data into the memory allocated for the net
	net.blobs['data'].data[...] = transformed_image

	### perform classification
	output = net.forward()

	output_prob = output['prob'][0]  # the output probability vector for the first image in the batch

	print 'predicted class is:', output_prob.argmax()

	# feat = net.blobs['conv1'].data[0, :96]
	feat = net.blobs['conv2'].data[0, :256]
	for layer_name, blob in net.blobs.iteritems():
		if(layer_name.startswith('conv')):
			print layer_name + '\t' + str(blob.data.shape)
			feat = net.blobs[layer_name].data[0, :blob.data.shape[1]]
			fname = 'L'+str(layer_name[-1])+'_'+layer_name
			vis_square(feat,fname,blob.data.shape)
