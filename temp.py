from normalize import normalize_image
from PIL import Image
import os


path = './my_pics'
for r,d,f in os.walk(path):
    print(f)
    