import caffe
import numpy as np

##Loading the model where 2nd caffemodel has weights after innerproduct layer is incorporated
caffe.set_mode_cpu()
net = caffe.Net('models/bvlc_reference_caffenet/deploy.prototxt',
'models/bvlc_reference_caffenet/bvlc_reference_caffenet2.caffemodel',
caffe.TEST)

#a= np.zeros([1000,1000])
#a[0,0]=1                                                                          #Step executed once to save the modified parameters
#net.params['fc9'][0].data[...] = a
#net.save('models/bvlc_reference_caffenet/bvlc_reference_caffenet2.caffemodel')

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load('caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
transformer.set_raw_scale('data', 255) 
transformer.set_channel_swap('data', (2,1,0))

net.blobs['data'].reshape(1,3,227,227)
net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image('examples/images/cat.jpg'))
out = net.forward()                                                                       
net.blobs['prob'].diff[0,281]=1                                                                 
print("Predicted class is #{}.".format(out['prob'].argmax()))
back = net.backward()
j = back['data'].copy()
print j


i=1



df_dwi= net.layers[i].blobs[0].diff                           #Gradients obtained                    
print df_dwi.shape

#gradient = df_dwi.reshape(96,3*11*11)

num=[]
for i in range(0,df_dwi.shape[0]):
     num.append(np.linalg.norm(df_dwi[i,:,:,:]))


print num




#df_dg = net.blobs['prob'].diff
#print df_dg


#num=[]
#for i in range(0,999):
      
      #if (df_dg[0,i]!=0):
         #t = np.linalg.norm(df_dwi[1,:,:,:])/(df_dg[0,i])
         #num.append(t)
      #else:
         #num.append(0)
    

#print num
