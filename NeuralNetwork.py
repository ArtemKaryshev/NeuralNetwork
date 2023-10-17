import os

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten

(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_train=x_train / 255
x_test=x_test / 255

y_train_cat=keras.utils.to_categorical(y_train,10)
y_test_cat=keras.utils.to_categorical(y_test,10)

model = keras.Sequential([
      Flatten(input_shape=(28,28,1)),
      Dense(128,activation='relu'),
      Dense(10,activation='softmax')
])



model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train,y_train_cat,batch_size=100,epochs=10,validation_split=0.2)

model.save('16_model.keras')