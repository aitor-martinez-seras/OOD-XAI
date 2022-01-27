import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
#from tensorflow import keras
#from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
# Utils
import argparse
from tqdm import tqdm
from sklearn.utils import shuffle
# Clustering
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
# Image processing and transformations
from skimage.transform import resize
from skimage.transform import rotate
from skimage import transform as trfm
from skimage.metrics import structural_similarity as ssim
from skimage import color
# To handle files
import os
import pickle
from zipfile import ZipFile
# Scipy functions
from scipy.optimize import minimize_scalar
from scipy.io import loadmat
from scipy.cluster.hierarchy import dendrogram
# Import created functions
from utils import *


def download_or_load_dataset(dataset_name:str, return_train=True):
    '''
    Dowloads the dataset
    :param dataset_name: str containing the name of the dataset
    :param return_train: if True, returns the train split of the dataset
    :return:
    '''
    print('Only SVHN_Cropped is downloaded directly to the datasets folder, the other datasets are stored'
          'locally in ~/.keras/datasets')
    try:
        if os.path.isdir(DATASET_DIR):
            pass
        else:
            os.mkdir(DATASET_DIR)
            print(f'The directory {DATASET_DIR} has been created')
    except Exception as e:
        print(f'Exception {e.__class__} occurred while creating the {DATASET_DIR} directory')
    if dataset_name == 'MNIST':
        # Load the dataset
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
        # Reduce the range of the images to [0,1]
        train_images = train_images / 255
        test_images = test_images / 255
        # Format images
        train_images = train_images.reshape(60000, 28, 28, 1)
        train_images = train_images.astype('float32')
        test_images = test_images.reshape(10000, 28, 28, 1)
        test_images = test_images.astype('float32')
        # Labels to categorical (10 dimensions with a 1 in the correspondent class)
        train_labels = to_categorical(train_labels)
        test_labels = to_categorical(test_labels)
        # Definition of the constants of the dataset
        class_names = list(np.linspace(0, 9, 10).astype('int'))
        class_names = [str(i) for i in class_names]
    elif dataset_name == 'Fashion_MNIST':
        # Load F_MNIST dataset
        fashion_mnist = tf.keras.datasets.fashion_mnist
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
        # Reduce the range of the images to [0,1]
        train_images = train_images / 255
        test_images = test_images / 255
        # Format images
        train_images = train_images.reshape(60000, 28, 28, 1)
        train_images = train_images.astype('float32')
        test_images = test_images.reshape(10000, 28, 28, 1)
        test_images = test_images.astype('float32')
        # Labels to categorical (10 dimensions with a 1 in the correspondent class)
        train_labels = to_categorical(train_labels)
        test_labels = to_categorical(test_labels)
        # Definition of the constants of the dataset
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
                       'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    elif dataset_name == 'Cifar10':
        cifar = tf.keras.datasets.cifar10
        (train_images, train_labels_clases), (test_images, test_labels_clases) = cifar.load_data()
        # Format images
        train_images = train_images.reshape(50000, 32, 32, 3)
        train_images = train_images.astype('float32') / 255
        test_images = test_images.reshape(10000, 32, 32, 3)
        test_images = test_images.astype('float32') / 255
        # Labels to categorical (10 dimensions with a 1 in the correspondent class)
        train_labels = to_categorical(train_labels_clases)
        test_labels = to_categorical(test_labels_clases)
        # Definition of the constants of the dataset
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
    elif dataset_name == 'SVHN_Cropped':
        # Download SVHN
        SVHN_ZIP_PATH = os.path.join(DATASET_DIR, SVHN_ZIP_FILE_NAME)
        SVHN_DIR_PATH = unzip_file(download_file_from_google_drive(SVHN_ID,SVHN_ZIP_PATH))
        # Load SVHN
        train_images, train_labels = load_svhn(SVHN_DIR_PATH, 'train_32x32.mat')
        test_images, test_labels = load_svhn(SVHN_DIR_PATH, 'test_32x32.mat')
        # Definition of the constants of the dataset
        class_names = list(np.linspace(0, 9, 10).astype('int'))
        class_names = [str(i) for i in class_names]
    else:
        raise NameError('Dataset name not found in the dataset options')

    if return_train is True:
        num_classes = len(class_names)
        return (train_images, train_labels), (test_images, test_labels), class_names, num_classes
    else:
        return test_images, test_labels


def transform_to_MNIST_format():
    '''
    Transforms a given dataset to MNIST format (28x28x1)
    :return:
    '''


def load_model_weigths():


def main():
    # Parse the arguments of the call
    parser = argparse.ArgumentParser(description='Script that trains the detector on a specific ')
    parser.add_argument('-i', '--ind', type=str, help='in distribution dataset',
                        choices=['MNIST', 'Fashion_MNIST', 'Cifar10'], required=True)
    parser.add_argument('-o', '--ood', type=str, help='out of distribution dataset',
                        choices=['MNIST', 'Fashion_MNIST', 'Cifar10', 'SVHN_Cropped'], required=True)
    parser.add_argument('-m', '--model_arch', type=str, choices=['LeNet', 'ResNet32'],
                        help='model architecture', required=True)
    args = vars(parser.parse_args())
    print(args)


if __name__ == '__main__':
    # Constants definition
    DATASET_DIR = 'datasets'
    PRETRAINED_WEIGHTS_ID = '1kubVcEv8ORheY0_3NuGb7VwE8OMbX5G9'
    PRETRAINED_WEIGHTS_DIR = 'pretrained_weigths'
    PRETRAINED_WEIGHTS_ZIP_FILE_NAME = 'OoD_xAI.zip'
    SVHN_ID = '1Qezu-SHyjBF_fGwdFYUSioVbAu3GMfBj'
    SVHN_ZIP_FILE_NAME = 'SVHN_Cropped.zip'

    # Run main code
    main()
