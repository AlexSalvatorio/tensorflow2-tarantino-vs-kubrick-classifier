#Adpation of the [Easy Image Classification with TensorFlow 2.0]
#https://towardsdatascience.com/easy-image-classification-with-tensorflow-2-0-f734fee52d13

import os
from tqdm import tqdm

#1. get labels and files
path_images = '../../../Trainning/training-movie-director'

def getFilesAndLabels():

    img_paths = []
    img_labels = []

    for image_dir in tqdm(os.listdir(path_images)):
        if image_dir != '.DS_Store':
            path = os.path.join(path_images,image_dir)
            for y in os.listdir(path):
                if y != '.DS_Store':
                    path_file = os.path.join(path,y)
                    #add path
                    img_paths.append(path_file)
                    #add label
                    img_labels.append(image_dir)

    return img_paths, img_labels

print(getFilesAndLabels())