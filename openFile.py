# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 18:16:53 2015

@author: wuyi
"""
import pylab
import numpy as np
from PIL import Image

def save(path, img):
    tmp = np.asarray(img*255.0, dtype=np.uint8)
    Image.fromarray(tmp).save(path)
 
image_id = 2
dirty_image_path = "../train/%d.png" % image_id
out_path = "../out"

dirty = Image.open(dirty_image_path)
#clean = Image.open(clean_image_path)
dirty_array = np.asarray(dirty)

print dirty_array.shape

img = pylab.imshow(dirty_array,0)

#save(out_path, dirty_array)