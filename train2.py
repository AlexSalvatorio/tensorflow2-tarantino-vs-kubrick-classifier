#Adpation of the [Easy Image Classification with TensorFlow 2.0]
#https://towardsdatascience.com/easy-image-classification-with-tensorflow-2-0-f734fee52d13

import os
from tqdm import tqdm
import tensorflow as tf

#1. get labels and files
path_images = '../../../Trainning/training-movie-director'
MAX_VALIDATION_FILES = 10;

def getFilesAndLabels():

    data_train_paths = []
    data_train_labels = []

    data_test_paths = []
    data_test_labels = []
    
    for image_dir in tqdm(os.listdir(path_images)):
        if image_dir != '.DS_Store':
            path = os.path.join(path_images,image_dir)
            numForabel = 0
            valided = MAX_VALIDATION_FILES
            for y in os.listdir(path):
                if y != '.DS_Store':
                    path_file = os.path.join(path,y)
                    if valided > 0:
                         #add path
                        data_test_paths.append(path_file)
                        #add label
                        data_test_labels.append(image_dir)
                        valided -= 1
                    else:
                        #add path
                        data_train_paths.append(path_file)
                        #add label
                        data_train_labels.append(image_dir)
                    numForabel+=1
            print(str(numForabel)+' pictures for label '+image_dir)
    return data_train_paths, data_train_labels, data_test_paths, data_test_labels

train_imgs, train_labels, test_imgs, test_labels= getFilesAndLabels()
print(len(train_labels))

train_data = tf.data.Dataset.from_tensor_slices(
  (tf.constant(train_imgs), tf.constant(train_labels))
)

val_data = tf.data.Dataset.from_tensor_slices(
  (tf.constant(test_imgs), tf.constant(test_labels))
)

IMAGE_SIZE = 96 # Minimum image size for use with MobileNetV2
BATCH_SIZE = 32

# Function to load and preprocess each image
def _parse_fn(filename, label):
    img = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img)
    img = (tf.cast(img, tf.float32)/127.5) - 1
    img = tf.image.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    return img, label
# Run _parse_fn over each example in train and val datasets
# Also shuffle and create batches
train_data = (train_data.map(_parse_fn)
             .shuffle(buffer_size=10000)
             .batch(BATCH_SIZE)
             )
val_data = (val_data.map(_parse_fn)
           .shuffle(buffer_size=10000)
           .batch(BATCH_SIZE)
           )