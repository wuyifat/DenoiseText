# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 21:54:33 2015

@author: wuyi
"""

import os
from PIL import Image
import numpy as np

size = set()
baseDir = "/home/wuyi/Code/Kaggle/DenoiseText/input/train"
countfor = 0
counttwo = 0
for img in os.listdir(baseDir):
    i = Image.open(baseDir + '/' + img)
    size.add(np.asarray(i).shape)
    if np.asarray(i).shape[0] == 420:
        countfor += 1
    else:
        counttwo += 1
print size, countfor, counttwo