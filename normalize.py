#goal: convert image to 28x28 monochrome format
#output: nparray
import sys
import numpy as np
from matplotlib.image import imsave
from matplotlib.image import imread


image = imread(sys.argv[1])
shape = image.shape

image_ = np.zeros(shape)
for i in range(shape[0]-1):
    for j in range(shape[1]):
        for k in range(shape[2]):
            image_[i][j][k] = image[i+1][j][k]


edge = np.absolute(image - image_)/256
imsave('edge.png', edge)