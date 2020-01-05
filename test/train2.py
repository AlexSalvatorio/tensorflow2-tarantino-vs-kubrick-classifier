#Adpation of the [Easy Image Classification with TensorFlow 2.0]
#IN PROGRESS!
#https://towardsdatascience.com/easy-image-classification-with-tensorflow-2-0-f734fee52d13

import os
import imghdr
from tqdm import tqdm
import tensorflow as tf

import matplotlib.pyplot as plt
from IPython import get_ipython
ipy = get_ipython()
if ipy is not None:
    ipy.run_line_magic('matplotlib', 'inline')

#1. get labels and files
path_images = '../../../Trainning/training-movie-director' #REPLACE BY THE PLACE OF YOUR DATA SET FOLDER
MAX_VALIDATION_FILES = 10;
MAX_TEST_FILES = 3;

def getFilesAndLabels():

    data_train_paths = []
    data_train_labels = []

    data_val_paths = []
    data_val_labels = []

    data_test_paths = []
    data_test_labels = []

    dict_dir = []
    dir_num = 0
    
    for image_dir in tqdm(os.listdir(path_images)):
        if image_dir != '.DS_Store':
            path = os.path.join(path_images,image_dir)
            numForabel = 0
            valided = MAX_VALIDATION_FILES
            tested = MAX_TEST_FILES
            dict_dir.append(image_dir)
            for y in os.listdir(path):
                path_file = os.path.join(path,y)
                if imghdr.what(path_file) is not None and (imghdr.what(path_file) == 'jpeg' or imghdr.what(path_file) == 'png'):
                #if '.jpg' not in y.lower() or '.jpeg' not in y.lower() or '.png' not in y.lower() or  '.gif' not in y.lower():
                   
                    #probably need to test the jpg
                    #https://www.tensorflow.org/api_docs/python/tf/io/decode_jpeg?version=stable

                    print(imghdr.what(path_file))

                    if valided > 0:
                         #add path
                        data_val_paths.append(path_file)
                        #add label
                        data_val_labels.append(dir_num)
                        valided -= 1
                    elif tested > 0:
                             #add path
                        data_test_paths.append(path_file)
                        #add label
                        data_test_labels.append(dir_num)
                        tested -= 1
                    else:
                        #add path
                        data_train_paths.append(path_file)
                        #add label
                        data_train_labels.append(dir_num)
                    numForabel+=1
            print(str(numForabel)+' pictures for label '+image_dir)
            dir_num +=1
    return data_train_paths, data_train_labels, data_val_paths, data_val_labels, data_test_paths

train_imgs, train_labels, val_imgs, val_labels, test_imgs = getFilesAndLabels()

train_data = tf.data.Dataset.from_tensor_slices(
  (tf.constant(train_imgs), tf.constant(train_labels))
)

val_data = tf.data.Dataset.from_tensor_slices(
  (tf.constant(val_imgs), tf.constant(val_labels))
)

IMAGE_SIZE = 96 # Minimum image size for use with MobileNetV2
BATCH_SIZE = 16

# Function to load and preprocess each image
def _parse_fn(filename, label):
    img = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img)
    img = (tf.cast(img, tf.float32)/127.5) - 1
    img = tf.image.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    return img, label
# Run _parse_fn over each example in train and val datasets
# Also shuffle and create batches
train_data =  train_data.map(_parse_fn)
train_data = (train_data
             .shuffle(buffer_size=10000)
             .batch(BATCH_SIZE)
             )

val_data = val_data.map(_parse_fn)
val_data = (val_data
           .shuffle(buffer_size=10000)
           .batch(BATCH_SIZE)
           )

IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SHAPE,
    include_top=False,
    weights='imagenet'
)

#Freeze the pretrained model
base_model.trainable = False

#Create layer for training
maxpool_layer = tf.keras.layers.GlobalMaxPooling2D()
prediction_layer = tf.keras.layers.Dense(1, activation='sigmoid')

#creating the model
model = tf.keras.Sequential([
    base_model,
    maxpool_layer,
    prediction_layer
])

learning_rate=0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

num_epochs = 30
steps_per_epoch = len(train_imgs)//BATCH_SIZE
val_steps = 20
model.fit(train_data.repeat(),
          epochs=num_epochs,
          steps_per_epoch = steps_per_epoch,
          validation_data=val_data.repeat(), 
          validation_steps=val_steps
          )

model.summary()

#displaying images
fig= plt.figure(figsize=(14,14))
for cnt, data in test_imgs:
    y = fig.add_subplot(6,5,cnt+1)
    img = data[0]
    data = img.reshape(1,64,64,1)
    data = tf.cast(data, tf.float32) #cast data to the float32 format, the int8 being not compatible with tensorflow
    model_out = model.predict([data])
    print(model_out)
    """
    str_label= [np.argmax(model_out)]
   
    y.imshow(img,cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
    """

plt.show()