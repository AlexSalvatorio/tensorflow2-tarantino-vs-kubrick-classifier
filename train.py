import cv2
import numpy as np
import os
from random  import shuffle
from tqdm import tqdm
import tensorflow as ts

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential

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
        #print('---------'+i+'---------'+str(type(img)))
        if str(type(img)).find('NoneType') != -1: 
            print('train '+i+' is empty!')
        else:
            img = cv2.resize(img, (64,64) ) 
            #print('---------'+i+'---------')
            #print(img)
            #print(np.array(img))
            train_images.append([np.array(img),one_hot_label(i)])
    return train_images

def test_data_with_label():
    test_images = []
    for i in tqdm(os.listdir(test_data)):
        path = os.path.join(test_data,i)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        #print('---------'+i+'---------'+str(type(img)))
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
model = Sequential()
"""
model.add(InputLayer(input_shape=[64,64,1])) #iput of the model
model.add(Conv2D(filters=32,kernel_size=5,stride=1,padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=5,padding='same' ))
"""