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
#print (im.shape, im.dtype)uint8

#Generate gaussian random measurement Matrices
sampleRate = 0.0625  #Sampling rate
Phi = np.random.randn(512, 512)
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

#IRLS algorithm function
def cs_irls(y,T_Mat):
    L=math.floor((y.shape[0])/4)
    hat_x_tp=np.dot(T_Mat.T ,y)
    epsilong=1
    p=1 # solution for l-norm p
    times=1
    while (epsilong>10e-9) and (times<L):  #Iteration times
        weight=(hat_x_tp**2+epsilong)**(p/2-1)
        Q_Mat=np.diag(1/weight)
        #hat_x=Q_Mat*T_Mat'*inv(T_Mat*Q_Mat*T_Mat')*y
        temp=np.dot(np.dot(T_Mat,Q_Mat),T_Mat.T)
        temp=np.dot(np.dot(Q_Mat,T_Mat.T),np.linalg.inv(temp))
        hat_x=np.dot(temp,y)
        if(np.linalg.norm(hat_x-hat_x_tp,2) < np.sqrt(epsilong)/100):
            epsilong = epsilong/10
        hat_x_tp=hat_x
        times=times+1
    return hat_x

start=time.time()

#Reconstruction
sparse_rec_1d=np.zeros((512,512))   #Initial sparse coefficient matrix
Theta_1d=np.dot(Phi,mat_dct_1d)   #Measurement matrix multiplied by basis matrix
for i in range(512):
    print('Column ',i,'is being reconstructed...')
    column_rec=cs_irls(img_cs_1d[:,i],Theta_1d)  #Calculation of sparse coefficients using IRLS algorithm
    sparse_rec_1d[:,i]=column_rec;
img_rec=np.dot(mat_dct_1d,sparse_rec_1d)          #Sparse coefficients multiplied by basis matrix

end=time.time()
time_consume = end - start
print('Time occupied by the IRLS method',time_consume)

#Show the reconstructed image
image2=Image.fromarray(img_rec)
plt.imshow(image2,cmap='gray')
mp.imsave('IRLS0.0625.jpg',image2,cmap='gray')

error = np.linalg.norm(img_rec-im)/np.linalg.norm(im)
print('Reconstruction accuracy',1-error)
