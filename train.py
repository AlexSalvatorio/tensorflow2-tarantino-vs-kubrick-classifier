

import cv2
import numpy as np
import os
from random  import shuffle
from tqdm import tqdm
import tensorflow as ts

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
from IPython import get_ipython
ipy = get_ipython()
if ipy is not None:
    ipy.run_line_magic('matplotlib', 'inline')

print('Tensor version: '+ts.__version__)

train_data = '../../../Trainning/training-google-images/train'
test_data = '../../../Trainning/training-google-images/test'

#print(os.listdir(train_data))

def one_hot_label(img):
    label = img.split('.')[0]
    if label == 'western':
        ohl = np.array([1,0])
    elif label == 'horror':
        ohl = np.array([0,1])
    return ohl

def train_data_with_label():
    train_images = []
    for i in tqdm(os.listdir(train_data)):
        path = os.path.join(train_data,i)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        if str(type(img)).find('NoneType') != -1: 
            print('train '+i+' is empty!')
        else:
            img = cv2.resize(img, (64,64) )
            train_images.append([np.array(img),one_hot_label(i)])
    return train_images

def test_data_with_label():
    test_images = []
    for i in tqdm(os.listdir(test_data)):
        path = os.path.join(test_data,i)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        if str(type(img)).find('NoneType') != -1: 
            print('test '+i+' is empty!')
        else:
            img = cv2.resize(img, (64,64) ) 
            test_images.append([np.array(img),one_hot_label(i)])
    return test_images

training_images = train_data_with_label()
test_images = test_data_with_label()

print (len(training_images))
print(np.array([i[1] for i in training_images]).shape)

tr_image_data = np.array([i[0] for i in training_images]).reshape(-1,64,64,1)
#why reshapping? creating a Numpy Matrix
tr_label_data = np.array([i[1] for i in training_images])

#creating model
model = Sequential([
    layers.InputLayer(input_shape=[64, 64,1]),

    layers.Conv2D(filters=32, kernel_size=5, strides=1, padding='same', activation='relu'),
    layers.MaxPool2D(pool_size=5,padding='same'),

    layers.Conv2D(filters=50, kernel_size=5, strides=1, padding='same', activation='relu'),
    layers.MaxPool2D(pool_size=5,padding='same'),

    layers.Conv2D(filters=80, kernel_size=5, strides=1, padding='same', activation='relu'),
    layers.MaxPool2D(pool_size=5,padding='same'),

    layers.Dropout(0.35),
    layers.Flatten(),
    layers.Dense(512,activation='relu'),
    layers.Dropout(rate=0.5),
    layers.Dense(2,activation='softmax'),
])

optimizer = Adam(lr=1e-3)

model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x=tr_image_data,y=tr_label_data,epochs=50,batch_size=100)
model.summary()

#displaying images
fig=plt.figure(figsize=(14,14))
for cnt, data in enumerate(test_images[10:40]):
    y = fig.add_subplot(6,5,cnt+1)
    img = data[0]
    data = img.reshape(1,64,64,1)
    data = ts.cast(data, ts.float32) #cast data to the float32 format, the int8 being not compatible with tensorflow
    model_out = model.predict([data])

    if np.argmax(model_out) == 1:
        str_label='western'
    else:
        str_label='horror'
    y.imshow(img,cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)

plt.show()