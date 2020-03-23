import os
import tensorflow as tf
from tensorflow import keras
from normalize import normalize_image
from PIL import Image


(train_images, train_labels), (test_images, test_labels) =  keras.datasets.mnist.load_data()
train_images =  train_images/255.0
test_images = test_images/255.0

model = keras.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation = tf.nn.relu),
    keras.layers.Dense(10, activation = tf.nn.softmax),
])
model.compile(
    optimizer = 'adam', 
    loss = 'sparse_categorical_crossentropy', 
    metrics = ['accuracy']
)
model.fit(train_images, train_labels, epochs = 3)
#model.evaluate(test_images, test_labels)

model.save('./trained_models/mlp')

print(10*'.\n')


my_pics = []
path = './my_pics'
for r,d,f in os.walk(path):
    for file in f:
        img =  normalize_image('my_pics/' + file).reshape(1,28,28,1) #reshape(#pic, W, H, channels)
        my_pics.append(img)

for img in my_pics:
    print(model.predict_classes(img))
 
