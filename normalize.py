import numpy as np
from PIL import Image, ImageOps


def normalize_image(path):
    img = Image.open(path).resize((28,28)) #resize
    img = Image.Image.convert(img, "L")  #convert to grayscale
    img = ImageOps.invert(img) #invert color
    img = np.array(img)/255.0 #convert to ndarray
    return img

