#Classification using Tensorflow 2
#https://www.tensorflow.org/tutorials/load_data/images

#To use these multiple images
#https://www.tensorflow.org/tutorials/images/classification
import pathlib2
import os
import numpy as np
import tensorflow as ts

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from IPython import get_ipython
ipy = get_ipython()
if ipy is not None:
    ipy.run_line_magic('matplotlib', 'inline')

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

data_dir = '../../../Trainning/training-movie-director'
types = ('*/*.png', '*/*.gif','*/*.jpg')

data_dir = pathlib2.Path(data_dir)
files_grabbed = []
for files in types:
    files_grabbed.extend(data_dir.glob(files))

image_count = len(files_grabbed)
print(image_count)

CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != ".DS_Store"])
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
image_generator = ts.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

BATCH_SIZE = 32
#For movie ratio
IMG_HEIGHT = 100
IMG_WIDTH = 239
STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)

train_data_gen = image_generator.flow_from_directory(directory=str(data_dir),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(IMG_WIDTH,IMG_HEIGHT),
                                                     classes = list(CLASS_NAMES))

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

IMG_SHAPE = (IMG_WIDTH,IMG_HEIGHT, 3)
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

model.compile(optimizer='adam',
              loss='binary_crossentropy', #binary pretty sure I should change this for multiple stuff
              metrics=['accuracy'])