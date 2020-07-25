
#The line above is necesary to show Matplotlib's plots inside a Jupyter Notebook

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage

image = cv2.imread("study materials/Image Set-20200714T090200Z-001/Image Set/images_chapter_06/Fig6.30(01).jpg")
if image is None:
    print("did not load")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    

hsv_image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
print(hsv_image.shape)


plt.imshow(hsv_image[:,:,0],cmap="gray")
plt.title('Hue')
plt.show()

plt.imshow(hsv_image[:,:,1],cmap="gray")
plt.title('Saturation',)
plt.show()


plt.imshow(hsv_image[:,:,2],cmap="gray")
plt.title('Intensity')
plt.show()



#Apply Negative Transform on both RGB and HSI color models.

#rgb

invert_rgb = 255 - image
plt.imshow(invert_rgb)
plt.title('RGB inverse')
plt.show()


invert_HSI = np.zeros_like(image)

for x in range (hsv_image.shape[0]):
    for y in range (hsv_image.shape[1]):
        if hsv_image[x,y,0]<90:   # Hue range is [0,179]   0...0.5 range maps to 0--90 and0.5...1maps to 90-179 mapr
            invert_HSI[x,y,0] = hsv_image[x,y,0]+90
        else:
            invert_HSI[x,y,0] = hsv_image[x,y,0]-90
invert_HSI[:,:,1] =  hsv_image[:,:,1]    #sat
invert_HSI[:,:,2] =  255-hsv_image[:,:,2]  #intensity
color_image = cv2.cvtColor(invert_HSI,cv2.COLOR_HSV2RGB)   

plt.imshow(color_image)
plt.title('HSV inverse')
plt.show()