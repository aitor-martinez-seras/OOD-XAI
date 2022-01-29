import keras
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
from models import create_model


def transform_to_MNIST_format():
    '''
    Transforms a given dataset to MNIST format (28x28x1)
    :return:
    '''


def transform_to_Cifar10_like():
    '''
    Transforms a given gray dataset to Cifar10 format (32x32x3)
    :return:
    '''




def main():
    # Parse the arguments of the call
    parser = argparse.ArgumentParser(description='Script that trains the detector on a specific ')
    parser.add_argument('-r_a', '--run_all', help='If used, it runs all the test of the paper', required=False)
    parser.add_argument('-i', '--ind', type=str, help='in distribution dataset', nargs='+',
                        choices=['MNIST', 'Fashion_MNIST', 'Cifar10'], required=False)
    parser.add_argument('-o', '--ood', type=str, help='out of distribution dataset', nargs='+',
                        choices=['MNIST', 'Fashion_MNIST', 'Cifar10', 'SVHN_Cropped'], required=False)
    parser.add_argument('-m', '--model_arch', type=str, choices=['LeNet', 'ResNet32'],
                        help='model architecture', required=True)
    parser.add_argument('-avg', '--average_mode', type=str, nargs='+',
                        help='average modes to be computed: Posible choices are Mean, Median or '
                             'an integer represeting the percentage', required=False)
    parser.add_argument('-ap', '--approach', type=str, choices=['g_r', 'g_all'], nargs='+',
                        help='approaches to be computed', required=False)
    #args = vars(parser.parse_args())
    args = {'ind': 'Cifar10',
            'ood': 'SVHN_Cropped',
            'model_arch': 'ResNet32',
            'average_mode': 'Mean',
            'approach': 'g_r'}
    # Visualize the arguments introduced
    print(args)
    # TOODO LO SIGUIENTE VA DENTRO DEL BLUCLE
    # Download the datasets
    (train_images, train_labels), (test_images, test_labels), class_names, num_classes = download_or_load_dataset(
        args['ind'], DATASET_DIR)
    print(class_names)
    # Create model
    model = create_model(args['model_arch'])
    # Load weights
    load_model_weights(model, dataset_name=args['ind'], model_name=args['model_arch'], weights_dir=PRETRAINED_WEIGHTS_DIR)
    metrics = model.evaluate(test_images, test_labels)
    print('Accuracy obtained is', str(round(metrics[1] * 100, 2)) + '%')
    # For every In-Distribution dataset, it has to run
    #for in_dataset in args['ind']:
        #train_od_detector()


if __name__ == '__main__':
    # Constants definition
    DATASET_DIR = 'datasets/'
    PRETRAINED_WEIGHTS_DIR = 'pretrained_weights/'
    # Run main code
    main()
