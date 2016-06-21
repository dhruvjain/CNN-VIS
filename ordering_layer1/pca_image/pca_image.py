from PIL import Image
import pylab
import numpy as np
from numpy import *

def rearrange(a):
  Y = pca.Y[:,0]
  X = a
  sorted_a= [x for (y,x) in sorted(zip(Y,X))]
  return sorted_a

def getorder(Principal_image,imlist):
  a = Principal_image
  sorted_list =[]
  for im in imlist:
    sorted_list.append(np.sum(np.multiply(a,im)))

  sorted_a= [x for (y,x) in sorted(zip(sorted_list,imlist))]
  print sorted_list
  return sorted_a

def pca(X):
  # Principal Component Analysis
  # input: X, matrix with training data as flattened arrays in rows
  # return: projection matrix (with important dimensions first),
  # variance and mean

  #get dimensions
  num_data,dim = X.shape

  #center data
  mean_X = X.mean(axis=0)
  for i in range(num_data):
      X[i] -= mean_X

  if dim>100:
      print 'PCA - compact trick used'
      M = dot(X,X.T) #covariance matrix
      e,EV = linalg.eigh(M) #eigenvalues and eigenvectors
      tmp = dot(X.T,EV).T #this is the compact trick
      V = tmp[::-1] #reverse since last eigenvectors are the ones we want
      S = sqrt(e)[::-1] #reverse since eigenvalues are in increasing order
  else:
      print 'PCA - SVD used'
      U,S,V = linalg.svd(X)
      V = V[:num_data] #only makes sense to return the first num_data

  #return the projection matrix, the variance and the mean
  return V,S,mean_X

if __name__ == '__main__':
	
	imlist = np.load('data.npy')
	imlistcopy=imlist
	im = imlist[0]

	# im = imlist[0] #open one image to get the size
	m,n = im.shape[0:2] #get the size of the images
	imnbr = len(imlist) #get the number of images

	# #create matrix to store all flattened images
	# imlist = imlist.tolist()
	immatrix = imlist.reshape(96,55*55)

	# #perform PCA
	V,S,immean = pca(immatrix)

	imlist = np.load('data.npy')

	mode = V[0].reshape(m,n) # the principal image
	print mode

	data=getorder(mode,imlist) # get the order according to mode
	data = np.rint(data)

	output_file = open('vis_pca.vox', 'wb')

	arr = np.int8(data)
	output_file.write(arr)
	output_file.close()
