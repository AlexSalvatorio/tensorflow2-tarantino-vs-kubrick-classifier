#Classification using Tensorflow 2
#https://www.tensorflow.org/tutorials/load_data/images

#To use these multiple images
#https://www.tensorflow.org/tutorials/images/classification
import pathlib2
import os
import numpy as np
from tqdm import tqdm
import cv2

import tensorflow as ts
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from IPython import get_ipython
ipy = get_ipython()
if ipy is not None:
    ipy.run_line_magic('matplotlib', 'inline')

print("TensorFlow version: {}".format(ts.__version__))
print("Eager execution: {}".format(ts.executing_eagerly()))

train_dir = '../../../Trainning/training-movie-director/train'
validation_dir = '../../../Trainning/training-movie-director/validation'
test_dir = '../../../Trainning/training-movie-director/test'
types = ('*/*.png', '*/*.gif','*/*.jpg')


train_dir = pathlib2.Path(train_dir)
validation_dir = pathlib2.Path(validation_dir)

#counting the image number
def countImagesInDir(dir_to_count):
    files_grabbed = []
    for files in types:
        files_grabbed.extend(dir_to_count.glob(files))

    return len(files_grabbed)

total_train = countImagesInDir(train_dir)
total_val = countImagesInDir(validation_dir)


#get the diff class
CLASS_NAMES = np.array([item.name for item in train_dir.glob('*') if item.name != ".DS_Store"])
print(CLASS_NAMES)

"""
#Test the display of pictures
tarantino = list(data_dir.glob('Tarantino/*'))
for image_path in tarantino[:3]:
    print(image_path)
    img = mpimg.imread(image_path)
    imgplot = plt.imshow(img)
    plt.show()
"""

# The 1./255 is to convert from uint8 to float32 in range [0,1].
train_image_generator = ts.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
horizontal_flip=True,
#width_shift_range=.15,
#height_shift_range=.15,
zoom_range=0.15
)
validation_image_generator = ts.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

BATCH_SIZE = 32
#For movie ratio
IMG_HEIGHT = 100
IMG_WIDTH = 239
#STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)
EPOCHS = 15
#EPOCHS = 1

train_data_gen = train_image_generator.flow_from_directory(directory=str(train_dir),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT ,IMG_WIDTH),
                                                     #classes = list(CLASS_NAMES) #FOR MULTIPLE CLASSE
                                                     class_mode='binary')

validation_data_gen = train_image_generator.flow_from_directory(directory=str(validation_dir),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT ,IMG_WIDTH),
                                                     #classes = list(CLASS_NAMES) #FOR MULTIPLE CLASSE
                                                     class_mode='binary')

#display un extrait
def show_batch(image_batch, label_batch):
    plt.figure(figsize=(10,10))

    for n in range(25):
        ax = plt.subplot(5,5,n+1)
        plt.imshow(image_batch[n])
        plt.title(CLASS_NAMES[label_batch[n]==1][0].title())
        plt.axis('off')
    plt.show()

image_batch, label_batch = next(train_data_gen)
#show_batch(image_batch, label_batch)

######################################
# LAUNCH THE MODEL PROCESSING
######################################

IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)
model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    #Dropout(0.2),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    #Dropout(0.2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])
"""
#old stuff from another tuts
model =  ts.keras.Sequential([
    ts.keras.Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    ts.keras.MaxPooling2D(),
    ts.keras.Conv2D(32, 3, padding='same', activation='relu'),
    ts.keras.MaxPooling2D(),
    ts.keras.Conv2D(64, 3, padding='same', activation='relu'),
    ts.keras.MaxPooling2D(),
    ts.keras.Flatten(),
    ts.keras.Dense(512, activation='relu'),
    ts.keras.Dense(1, activation='sigmoid')
])
"""

model.compile(optimizer='adam',
              loss='binary_crossentropy', #binary pretty sure I should change this for multiple stuff
              metrics=['accuracy'])

model.summary()

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_data_gen,
    validation_steps=total_val // BATCH_SIZE
)

######################################
# Display model's result
######################################

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

######################################
# Launch the TEST
######################################

def get_test_data():
    test_images = []
    test_paths = []
    for i in tqdm(os.listdir(test_dir)):
        path = os.path.join(test_dir,i)
        test_paths.append(path)
        img = cv2.imread(path)
        if str(type(img)).find('NoneType') != -1: 
            print('test '+i+' is empty!')
        else:
            img = cv2.resize(img, (IMG_WIDTH,IMG_HEIGHT) ) 
            test_images.append([np.array(img)])
    return test_images, test_paths

TEST_DATA, TEST_PATH = get_test_data()
#displaying images
fig=plt.figure(figsize=(14,14))

for cnt, img in enumerate(TEST_DATA[:24]):
    y = fig.add_subplot(6,5,cnt+1)
    #data = img.reshape(1,239,100,1)
    data = ts.cast(img, ts.float32) #cast data to the float32 format, the int8 being not compatible with tensorflow
    model_out = model.predict([data])
    argmax = np.argmax(model_out)
    
    print('---------------')
    print(argmax)
    argmax = round(argmax)
    print(argmax)
    argmax = int(argmax)
    print(argmax)
    
    str_label = CLASS_NAMES[argmax]

    #img = img[0].reshape(1,IMG_WIDTH,IMG_HEIGHT,1)
    y.imshow(plt.imread(TEST_PATH[cnt]))

    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)

plt.show()