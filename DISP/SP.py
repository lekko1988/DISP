#coding:utf-8
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# The DCT basis is used as a sparse basis, the reconstruction algorithm is the SP algorithm, and the images are processed by column
# Reference: W. Dai and O. Milenkovic, “Subspace Pursuit for Compressive
# Sensing Signal Reconstruction,” 2009.
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#Importe integration libraries
import math
import time
import matplotlib.image as mp
import matplotlib.pyplot as plt

# Importing the required third-party library files
import  numpy as np    #numpy package
from PIL import Image  #pillow package


#Read the image and turn it into an array of numpy type
img = Image.open('lena_RGB.jpg')
im = img.convert('L')
#im=img[:,:,0]

#Generate gaussian random measurement Matrices
sampleRate=0.0625  #Sampling rate
Phi=np.random.randn(int(512*sampleRate),512)

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

#SP algorithm function
def cs_sp(y,D):
    K=math.floor(y.shape[0]/3)
    pos_last=np.array([],dtype=np.int64)
    result=np.zeros((512))

    product=np.fabs(np.dot(D.T,y))
    pos_temp=product.argsort()
    pos_temp=pos_temp[::-1]#Reverse to get the first L large positions
    pos_current=pos_temp[0:K]#Initialize the index set, corresponding to step 1
    residual_current=y-np.dot(D[:,pos_current],np.dot(np.linalg.pinv(D[:,pos_current]),y))#Initialize residual, corresponding to step 2
    while True:  #Iteration times
        product=np.fabs(np.dot(D.T,residual_current))
        pos_temp=np.argsort(product)
        pos_temp=pos_temp[::-1]#Reverse to get the first L large positions
        pos=np.union1d(pos_current,pos_temp[0:K])#Corresponding to step 1
        pos_temp=np.argsort(np.fabs(np.dot(np.linalg.pinv(D[:,pos]),y)))#Corresponding to step 2
        pos_temp=pos_temp[::-1]
        pos_last=pos_temp[0:K]#Corresponding to step 3
        residual_last=y-np.dot(D[:,pos_last],np.dot(np.linalg.pinv(D[:,pos_last]),y))#Update residual #Corresponding to step 1
        if np.linalg.norm(residual_last)>=np.linalg.norm(residual_current): #Corresponding to step 5
            pos_last=pos_current
            break
        residual_current=residual_last
        pos_current=pos_last
    result[pos_last[0:K]]=np.dot(np.linalg.pinv(D[:,pos_last[0:K]]),y) #Corresponding to output
    return  result

start=time.time()

#Reconstruction
sparse_rec_1d=np.zeros((512,512))   #Initial sparse coefficient matrix
Theta_1d=np.dot(Phi,mat_dct_1d)   #Measurement matrix multiplied by basis matrix
for i in range(512):
    print('Column ',i,'is being reconstructed...')
    column_rec=cs_sp(img_cs_1d[:,i],Theta_1d)  #Calculation of sparse coefficients using SP algorithm
    sparse_rec_1d[:,i]=column_rec;
img_rec=np.dot(mat_dct_1d,sparse_rec_1d)          #Sparse coefficients multiplied by basis matrix

end=time.time()
time_consume = end - start
print('Time occupied by the SP method',time_consume)

#Show the reconstructed image
image2=Image.fromarray(img_rec)
plt.imshow(image2,cmap='gray')
mp.imsave('lena_sp0.0625.jpg',image2,cmap='gray')

error = np.linalg.norm(img_rec-im)/np.linalg.norm(im)
print('Reconstruction accuracy',1-error)
