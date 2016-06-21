import caffe
import numpy as np

##Loading the model where 2nd caffemodel has weights after innerproduct layer is incorporated
caffe.set_mode_cpu()
net = caffe.Net('models/bvlc_reference_caffenet/deploy.prototxt',
'models/bvlc_reference_caffenet/bvlc_reference_caffenet5.caffemodel',
caffe.TEST)

#a= np.zeros([1000,1000])
#a[0,0]=1                                                                          #Step executed once to save the modified parameters
#net.params['fc9'][0].data[...] = a
#net.save('models/bvlc_reference_caffenet/bvlc_reference_caffenet2.caffemodel')

a = net.params['conv1'][0].data[...]


transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load('caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
transformer.set_raw_scale('data', 255) 
transformer.set_channel_swap('data', (2,1,0))

net.blobs['data'].reshape(1,3,227,227)
net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image('examples/images/cat.jpg')) 
net.blobs['label'].data[...]=281
out = net.forward()
before_change = net.blobs['fc9'].data


a[0,0,0,0] += 0.01
net.params['conv1'][0].data[...] = a
net.save('models/bvlc_reference_caffenet/bvlc_reference_caffenet3.caffemodel')
net = caffe.Net('models/bvlc_reference_caffenet/deploy.prototxt',
'models/bvlc_reference_caffenet/bvlc_reference_caffenet3.caffemodel',
caffe.TEST)
net.blobs['data'].reshape(1,3,227,227)
net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image('examples/images/cat.jpg'))
net.blobs['label'].data[...]=281
out = net.forward()
after_change = net.blobs['fc9'].data
df_dwi = after_change-before_change

print np.linalg.norm(df_dwi/0.01)





