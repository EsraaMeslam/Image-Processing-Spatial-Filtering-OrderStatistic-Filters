#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


def max_filter(img, kernel_sz):
  

    output_image = np.zeros_like(img)

    
   
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):

            neighborhood = img[i:i+kernel_sz, j:j+kernel_sz]

            max_=np.max(neighborhood)
  
            output_image[i, j] = max_
    
    return output_image


# In[3]:


img = cv2.imread('papper.jpeg', cv2.IMREAD_GRAYSCALE)

filtered_image = max_filter(img, kernel_sz=3)


# In[4]:


# Display the original and filtered images
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')


plt.subplot(1, 2, 2)
plt.imshow(filtered_image, cmap='gray')
plt.title('Max Filtered Image')


plt.show()


# In[ ]:




