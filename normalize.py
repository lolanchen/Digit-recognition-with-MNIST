import numpy as np
from PIL import Image


def normalize_image(path):
    image = Image.open(path).resize((28,28))
    image = Image.Image.convert(image, "L")
    image = np.array(image)/255.0
    return image
