
import math
import time
import numpy as np
from PIL import Image
import matplotlib.image as mp
import matplotlib.pyplot as plt


img = Image.open('lena_RGB.jpg')
im = img.convert('L')
#im=img[:,:,0]
plt.imshow(im,cmap='gray')
plt.axis('off')
plt.show()

print(np.array(im).shape)

rsize=np.array(im).shape[0]
csize=np.array(im).shape[1]


#Generate gaussian random measurement Matrices
sampleRate=0.0625  #Sampling rate
Phi=np.random.randn(int(rsize*sampleRate),csize)
# Phi=np.random.randn(256,256)
# u, s, vh = np.linalg.svd(Phi)
# Phi = u[:256*sampleRate,] #Orthogonalize the measurement matrix



#Generate sparse basis DCT matrix
mat_dct_1d=np.zeros((rsize,csize))
v=range(csize)
for k in range(0,csize):
    dct_1d=np.cos(np.dot(v,k*math.pi/csize))
    if k>0:
        dct_1d=dct_1d-np.mean(dct_1d)
    mat_dct_1d[:,k]=dct_1d/np.linalg.norm(dct_1d)

#Random measurement
img_cs_1d=np.dot(Phi,im)

#CoSaMP algorithm function
def cs_CoSaMP(y,D):
    S=math.floor(y.shape[0]/4)  #Sparsity ratio
    residual=y  #Initial residual
    pos_last=np.array([],dtype=np.int64)
    result=np.zeros((csize))

    for j in range(S):  #Iteration times
        product=np.fabs(np.dot(D.T,residual))
        pos_temp=np.argsort(product)
        pos_temp=pos_temp[::-1]#Reverse to get the first L large positions
        pos_temp=pos_temp[0:2*S]#Correspond to step 3
        pos=np.union1d(pos_temp,pos_last)

        result_temp=np.zeros((csize))
        result_temp[pos]=np.dot(np.linalg.pinv(D[:,pos]),y)

        pos_temp=np.argsort(np.fabs(result_temp))
        pos_temp=pos_temp[::-1]#Reverse to get the first L large positions
        result[pos_temp[:S]]=result_temp[pos_temp[:S]]
        pos_last=pos_temp
        residual=y-np.dot(D,result)
    return  result


sparse_rec_1d=np.zeros((rsize,csize))   #Initial sparse coefficient matrix
Theta_1d=np.dot(Phi,mat_dct_1d)   #Measurement matrix multiplied by basis matrix

start=time.time()
for i in range(csize):
    print('Column ',i,'is being reconstructed...')
    column_rec=cs_CoSaMP(img_cs_1d[:,i],Theta_1d)  #Calculation of sparse coefficients using CoSaMP algorithm
    sparse_rec_1d[:,i]=column_rec;
img_rec=np.dot(mat_dct_1d,sparse_rec_1d)          #Sparse coefficients multiplied by basis matrix

end=time.time()
time_consume = end - start
print(time_consume)

'''
#Mean-pooling
def avg_pooling(data, m, n):
    a,b = data.shape
    img_new = []
    for i in range(0,a,m):
        line = []#Record every line
        for j in range(0,b,n):
            x = data[i:i+m,j:j+n]#select pooling regions
            line.append(np.sum(x)/(n*m))
        img_new.append(line)
    return np.array(img_new)
'''


#Show the reconstructed image
image2=Image.fromarray(img_rec)
plt.imshow(image2,cmap='gray')
mp.imsave('lena_CoSoMP0.0625.jpg',image2,cmap='gray')

error = np.linalg.norm(img_rec-im)/np.linalg.norm(im)
print('Reconstruction accuracy',1-error)








