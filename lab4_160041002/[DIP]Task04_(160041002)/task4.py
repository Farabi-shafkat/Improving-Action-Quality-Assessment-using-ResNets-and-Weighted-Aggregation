###Task 4: (Image Enhancement with contrast stretching)
#A.	Implement histogram equalization (HE) using RGB and HSI color models (separately).
#B.	Show the differences between the two outputs obtained. 


#The line above is necesary to show Matplotlib's plots inside a Jupyter Notebook

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage

image = cv2.imread("study materials/Image Set-20200714T090200Z-001/Image Set/images_chapter_06/Fig6.30(01).jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#histogram equalize the image in RGB

def histogram_rgb(image):

    
    
    img = image
    out = np.zeros_like(img)
    
    
    h,w,c = img.shape
    for ch in range(c):
        hist=np.zeros((256),dtype='uint32')
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
    plt.title('Histogram RGB')
    plt.show()
    return out

HE_rgb=histogram_rgb(image)


def histogram_HSI(image):

    
    
    img = image
    out = np.zeros_like(img)
    
    
    h,w,c = img.shape
    
    hist=np.zeros((256),dtype='uint32')
    for i in range(h):
        for j in range(w):
            hist[img[i][j][2]]+=1
    pdf=hist/(w*h)
    cdf=np.zeros_like(pdf)
    for i,el in enumerate(pdf):
        if i==0:
            cdf[i]=el
        else:
            cdf[i]=el+cdf[i-1]
    cdf=255*cdf

    for (x, y),pixel in np.ndenumerate(img[:,:,2]):
        out[x][y][2]=cdf[pixel]
    
    out = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    plt.imshow(out)
    plt.title('Histogram HSI')
    plt.show()
    return out
    
hsv_image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
HE_hsi=histogram_HSI(hsv_image)



dif = abs(HE_rgb-HE_hsi)

out = histogram_HSI(dif)
