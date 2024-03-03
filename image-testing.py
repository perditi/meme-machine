# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 23:43:29 2024

@author: timot
"""

from PIL import Image
import numpy as np

img = Image.open("C:\\Users\\Owner\\Dropbox\\PC\\Desktop\\meme-machine\\meme-machine\\chelsea.jpg")
print(img)
numpy_array = np.array(img)
print(numpy_array.shape)

