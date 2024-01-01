#coding:utf-8
#Importe integration libraries
import math
import time
import numpy as np
import matplotlib.image as mp
import matplotlib.pyplot as plt

# Importing the required third-party library files
import  numpy as np    #numpy package
from PIL import Image  #pillow package


#Read the image and turn it into an array of numpy type
img = Image.open('lena_RGB.jpg')
im = img.convert('L')

#Generate gaussian random measurement Matrices
sampleRate=0.0625  #Sampling rate
Phi=np.random.randn(512,512)
u, s, vh = np.linalg.svd(Phi)
Phi = u[:int(512*sampleRate),] #Orthogonalize the measurement matrix


#Generate sparse basis DCT matrix
mat_dct_1d=np.zeros((512,512))
v=range(512)
for k in range(0,512):
    dct_1d=np.cos(np.dot(v,k*math.pi/512))
    if k>0:
        dct_1d=dct_1d-np.mean(dct_1d)
    mat_dct_1d[:,k]=dct_1d/np.linalg.norm(dct_1d)

#Random measurement
img_cs_1d=np.dot(Phi,im)

#IHT algorithm function
def cs_IHT(y,D):
    K=math.floor(y.shape[0]/3)  #Sparsity ratio
    result_temp=np.zeros((512))  #Initialize the rebuild signal
    u=0.5  #Impact factor
    result=result_temp
    for j in range(K):  #Iteration times
        x_increase=np.dot(D.T,(y-np.dot(D,result_temp)))    #x=D*(y-D*y0)
        result=result_temp+np.dot(x_increase,u) #   x(t+1)=x(t)+D*(y-D*y0)
        temp=np.fabs(result)
        pos=temp.argsort()
        pos=pos[::-1]#Reverse to get the first L large positions
        result[pos[K:]]=0
        result_temp=result
    return  result

start=time.time()

#Reconstruction
sparse_rec_1d=np.zeros((512,512))   #Initial sparse coefficient matrix
Theta_1d=np.dot(Phi,mat_dct_1d)   #Measurement matrix multiplied by basis matrix
for i in range(512):
    print('Column ',i,'is being reconstructed...')
    column_rec=cs_IHT(img_cs_1d[:,i],Theta_1d)  #Calculation of sparse coefficients using IHT algorithm
    sparse_rec_1d[:,i]=column_rec;
img_rec=np.dot(mat_dct_1d,sparse_rec_1d)          #Sparse coefficients multiplied by basis matrix


end=time.time()
time_consume = end - start
print('Time occupied by the OMP method',time_consume)

#Show the reconstructed image
image2=Image.fromarray(img_rec)
plt.imshow(image2,cmap='gray')
mp.imsave('IHT0.0625.jpg',image2,cmap='gray')

error = np.linalg.norm(img_rec-im)/np.linalg.norm(im)
print('Reconstruction accuracy',1-error)
