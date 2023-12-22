#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


def mid_point_filter(img, kernel_sz):
  

    output_image = np.zeros_like(img)

    
   
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):

            neighborhood = img[i:i+kernel_sz, j:j+kernel_sz]
            
            miin=np.min(neighborhood)
            maax=np.max(neighborhood)
            

            mid_point=1/2*(miin+maax)
  
            output_image[i, j] = mid_point
    
    return output_image


# In[5]:


img = cv2.imread('salt&papper.png', cv2.IMREAD_GRAYSCALE)

filtered_image = mid_point_filter(img, kernel_sz=3)


# In[8]:


plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')


plt.subplot(1, 2, 2)
plt.imshow(filtered_image, cmap='gray')
plt.title('Mid point filter')


plt.show()


# In[ ]:




