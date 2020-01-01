#Classification using Tensorflow 2
#https://www.tensorflow.org/tutorials/load_data/images
import pathlib2
import numpy as np
import tensorflow as ts

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from IPython import get_ipython
ipy = get_ipython()
if ipy is not None:
    ipy.run_line_magic('matplotlib', 'inline')


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