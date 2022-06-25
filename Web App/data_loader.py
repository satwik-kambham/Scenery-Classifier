import numpy as np
from PIL import Image
from os import listdir
from random import shuffle

CATEGORIES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
TRAINING_DATA_PATH = '../data/train/seg_train/'
TESTING_DATA_PATH = '../data/test/seg_test/'

def load_images(path: str, category1: str, category2: str, reshape=True, standardize=True, shuffle_images=True):
    """Loads all images from specified directory into a numpy array and shuffles it.

    Args:
        path (str): Path to folder containing categories of images.
        category1 (str): name of folder containing images of category 1.
        category2 (str): name of folder containing images of category 2.
        reshape (bool, optional): Should the images be reshaped into a column matrix. Defaults to True.
        standardize (bool, optional): Should the rgb values be standardized. Defaults to True.
        shuffle_images (bool, optional): Should the images be shuffled. Defaults to True.

    Returns:
        images (ndarray): Numpy array containing all images
        labels (ndarray): Numpy array containing all labels
    """
    print(category1, category2)

    categories = (category1, category2)

    images = []
    labels = []

    for i in range(2):
        path_to_category = path + categories[i] + '/'
        for filename in listdir(path_to_category):
            img = Image.open(path_to_category + filename)
            img = img.resize([150, 150])
            img = np.array(img)
            if reshape: img = img.reshape(150*150*3)
            if standardize: img = img / 255
            images.append(img)
            labels.append(i)

    if shuffle_images: 
        data = list(zip(images, labels))
        shuffle(data)
        images, labels = zip(*data)

    labels = np.stack(labels)

    return np.stack(images), np.stack(labels)
