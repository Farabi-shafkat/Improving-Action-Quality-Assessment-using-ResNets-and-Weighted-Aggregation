#The line above is necesary to show Matplotlib's plots inside a Jupyter Notebook

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage


#Task 3: (Noise Removal)
#A.	Remove Salt&Pepper noise from RGB color image using Median Filter.


#this function adds salt and pepper noise




import numpy as np
import random
import cv2

def sp_noise(image,prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


image = cv2.imread("study materials/Image Set-20200714T090200Z-001/Image Set/images_chapter_06/Fig6.30(01).jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


noise_img = sp_noise(image,0.05)
plt.imshow(image)
plt.title('origianl')
plt.show()
plt.imshow(noise_img)
plt.title('noisy')
plt.show()





n = 3

pad=np.zeros([h+2*(n-1),w+2*(n-1),3],dtype='double')




h,w,c=noise_img.shape


pad[n-1:h+n-1,n-1:w+n-1] = noise_img[0:h,0:w,:]

out=np.zeros([h+n-1,w+n-1,3],dtype='double')


h,w,c = out.shape

for ch in range(c):
    for x in range(h):
        for y in range(w):
            val = pad[x:x+n,y:y+n,ch]
            val = val.reshape((-1,1))
            val = np.median(val)
           # print(np.median(val))
            out[x,y,ch] = val
            
mx = np.max(out)
norm = out/mx


plt.imshow(norm)
plt.title('filtered')