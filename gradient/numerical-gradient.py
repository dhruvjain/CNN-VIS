import caffe
import numpy as np

##Loading the model where 2nd caffemodel has weights after innerproduct layer is incorporated
caffe.set_mode_cpu()
net = caffe.Net('models/bvlc_reference_caffenet/deploy.prototxt',
'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
caffe.TEST)

#a= np.zeros([1,1000])
#a[0,0]=1                                                                          #Step executed once to save the modified parameters
#net.params['fc9'][0].data[...] = a
#net.save('models/bvlc_reference_caffenet/bvlc_reference_caffenet4.caffemodel')

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load('python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
transformer.set_raw_scale('data', 255) 
transformer.set_channel_swap('data', (2,1,0))

net.blobs['data'].reshape(1,3,227,227)
net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image('examples/images/cat.jpg'))
net.blobs['label'].data[...]=281



#a= np.zeros([1000,1000])
#a[0,0]=1                                                                          #Step executed once to save the modified parameters
#net.params['fc9'][0].data[...] = a
#net.save('models/bvlc_reference_caffenet/bvlc_reference_caffenet2.caffemodel')

a = net.params['conv1'][0].data[...]
out = net.forward()
before_change = net.blobs['fc8'].data[0,263]

df_dwi =np.zeros([96,3,11,11])

for i in range(0,96):
   for j in range(0,3):
      for k in range(0,11):
         for l in range(0,11):
		a[i,j,k,l] += 0.000000001
		net.params['conv1'][0].data[...] = a
		net.blobs['data'].reshape(1,3,227,227)
		net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image('data/input_images/chihuahua.jpg'))
		net.blobs['label'].data[...]=281
		out = net.forward()
		after_change = net.blobs['fc8'].data[0,263]
		df_dwi[i,j,k,l]=(after_change-before_change)/0.000000001



np.save('chihuahua.npy',df_dwi)
