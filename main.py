import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

print("train data shape: " , np.shape(train_images))
print("test data shape: ", np.shape(test_images))

'''
plt.figure(1)
for i in range(3):
    for j in range(3):
        plt.subplot(3, 3, i*3+j+1)
        plt.imshow(train_images[i*3+j+1,:,:])
        plt.colorbar()
plt.show()
'''

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# preprocessing
train_images = train_images/255.0
test_images = test_images/255.0

# make model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(train_images,train_labels,epochs=10)

test_loss, test_acc = model.evaluate(test_images,test_labels,verbose = 2)


