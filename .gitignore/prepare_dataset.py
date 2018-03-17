import os
import numpy as np
import pandas as pd
from PIL import Image
import random

base_path = 'C:/Hardik/Development/Python/MachineLearning/Kaggle/PlantSeedingClassification/Dataset/train'
base_test_path = 'C:/Hardik/Development/Python/MachineLearning/Kaggle/PlantSeedingClassification/Dataset/test'

categ_dict = {
    'Black-grass': 0,
    'Charlock': 1,
    'Cleavers': 2,
    'Common Chickweed': 3,
    'Common wheat': 4,
    'Fat Hen': 5,
    'Loose Silky-bent': 6,
    'Maize': 7,
    'Scentless Mayweed': 8,
    'Shepherds Purse': 9,
    'Small-flowered Cranesbill': 10,
    'Sugar beet': 11
}

inv_categ_dict = {
    0: 'Black-grass',
    1: 'Charlock',
    2: 'Cleavers',
    3: 'Common Chickweed',
    4: 'Common wheat',
    5: 'Fat Hen',
    6: 'Loose Silky-bent',
    7: 'Maize',
    8: 'Scentless Mayweed',
    9: 'Shepherds Purse',
    10: 'Small-flowered Cranesbill',
    11: 'Sugar beet'
}


image_height = 300
image_width = 300


def get_one_hot_vector(category_name):
    vector = np.zeros((12,), dtype=int)
    vector[categ_dict[category_name]] = 1
    return vector


def get_image_data(image: Image):
    resized_image = image.resize((image_width, image_height), Image.ANTIALIAS)
    data = resized_image.load()
    vector = np.zeros((image_width, image_height, 3), dtype=int)
    for i in range(image_width):
        for j in range(image_height):
            vector[i, j, 0] = (data[i, j][0] - 128) / 128
            vector[i, j, 1] = (data[i, j][1] - 128) / 128
            vector[i, j, 2] = (data[i, j][2] - 128) / 128
    return vector


def get_training_data_line(image: Image, category):
    data = {
        'input':    get_image_data(image),
        'output':   get_one_hot_vector(category),
        'category': category
    }
    return data


def get_testing_data_line(image:Image):
    data = {
        'input':    get_image_data(image),
        'category': ''
    }
    return data


def prepare_training_files():
    dataset = []

    file_dir_list = os.listdir(base_path)
    for name in file_dir_list:
        dir_file_list = os.listdir(base_path + '/' + name)
        for file_name in dir_file_list:
            dataset.append(name + '/' + file_name)

    ds = pd.DataFrame(data=dataset)
    ds = ds.sample(frac=1)

    return ds.as_matrix()


def prepare_testing_files():
    dataset = []
    test_image_list = os.listdir(base_test_path)

    for name in test_image_list:
        dataset.append(name)

    return dataset


def prepare_training_dataset(file_name):
    name, _ = file_name[0].split("/")
    image = Image.open(base_path + '/' + file_name[0])
    new_image = image.resize((image_width, image_height), Image.ANTIALIAS)
    return get_training_data_line(new_image,name)


def prepare_testing_dataset():

    dataset = []
    test_image_list = os.listdir(base_test_path)

    for name in test_image_list:
        print(name)
        image = Image.open(base_test_path + '/' + name)
        dataset.append(get_testing_data_line(image))

    return dataset
