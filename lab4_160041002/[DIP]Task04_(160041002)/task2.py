#The line above is necesary to show Matplotlib's plots inside a Jupyter Notebook

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
image = cv2.imread("study materials/Image Set-20200714T090200Z-001/Image Set/images_chapter_06/Fig6.30(01).jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
if image is None:
    print("did not load")

fil = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],dtype='uint8')

hsv_image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)

 
k=1

plt.title('original')
plt.imshow(image)
plt.show()


sharpen_img_1 = np.zeros(image.shape,dtype='uint8')

sharpen_img_1[:,:,0]=cv2.filter2D(image[:,:,0],-1,fil)
sharpen_img_1[:,:,1]=cv2.filter2D(image[:,:,1],-1,fil)
sharpen_img_1[:,:,2]=cv2.filter2D(image[:,:,2],-1,fil)


sharpen_img_1 = image + k*sharpen_img_1
#highboost_image_rgb = k*high+image
plt.imshow(sharpen_img_1)
plt.title('highboost rgb')
plt.show()


hsv_image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
 


sharpen_img_hsv = np.zeros(hsv_image.shape,dtype='uint8')

sharpen_img_hsv = hsv_image

sharpen_img_hsv[:,:,2]=cv2.filter2D(hsv_image[:,:,2],-1,fil)


sharpen_img_hsv[:,:,2]=hsv_image[:,:,2]+sharpen_img_hsv[:,:,2]*k


color_image = cv2.cvtColor(sharpen_img_hsv,cv2.COLOR_HSV2RGB)   

plt.imshow(color_image)
plt.title('HSV sharp')
plt.show()



dif = sharpen_img_1-color_image
#histogram equalize the dif
img = dif

hist=np.zeros((256),dtype='uint32')

out = np.zeros_like(img)

h,w,c = dif.shape
for ch in range(c):
    for i in range(h):
        for j in range(w):
            hist[img[i][j][ch]]+=1
    pdf=hist/(w*h)
    cdf=np.zeros_like(pdf)
    for i,el in enumerate(pdf):
        if i==0:
            cdf[i]=el
        else:
            cdf[i]=el+cdf[i-1]
    cdf=255*cdf
 
    for (x, y),pixel in np.ndenumerate(img[:,:,ch]):
        out[x][y][ch]=cdf[pixel]

plt.imshow(out)
plt.title('different')
plt.show()