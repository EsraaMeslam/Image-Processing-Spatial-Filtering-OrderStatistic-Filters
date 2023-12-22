#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[5]:


def min_filter(img, kernel_sz):
  

    output_image = np.zeros_like(img)

    
   
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):

            neighborhood = img[i:i+kernel_sz, j:j+kernel_sz]

            min_=np.min(neighborhood)
  
            output_image[i, j] = min_
    
    return output_image


# In[6]:


img = cv2.imread('salt.jpeg', cv2.IMREAD_GRAYSCALE)

filtered_image = min_filter(img, kernel_sz=3)


# In[7]:


plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')


plt.subplot(1, 2, 2)
plt.imshow(filtered_image, cmap='gray')
plt.title('Min Filtered Image')


plt.show()


# In[ ]:




