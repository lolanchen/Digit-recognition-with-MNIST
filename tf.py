import tensorflow as tf
from tensorflow import keras
from normalize import normalize_image
import sys

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
model.fit(train_images, train_labels, epochs = 5)

model.evaluate(test_images, test_labels)

img3 = normalize_image('3.jpeg').reshape(1,28,28,1)
img5 = normalize_image('5.jpeg').reshape(1,28,28,1)
img6 = normalize_image('6.jpeg').reshape(1,28,28,1)


print(model.predict_classes(img3))
print(model.predict_classes(img5))
print(model.predict_classes(img6))



#model.evaluate(test_images, test_labels)
