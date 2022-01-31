import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
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
from train_detector import train_od_detector
from constants import *


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
                        help='model architecture', required=False)
    parser.add_argument('-l_or_t', '--load_or_train', type=str, choices=['Load', 'Train'],
                        help='model architecture', required=False)
    parser.add_argument('-avg', '--average_mode', type=str, nargs='+',
                        help='average modes to be computed: Possible choices are Mean, Median or '
                             'an integer representing the percentage', required=False)
    parser.add_argument('-ap', '--approach', type=str, choices=['g_r', 'g_all'], nargs='+',
                        help='approaches to be computed', required=False)
    parser.add_argument('-s', '--seed', type=int, help='Seed for shuffling the train images and labels', required=False)
    parser.add_argument('-n_htmaps', '--n_heatmaps', type=int, help='Select the number of heatmaps per class '
                                                                    'for the clustering', required=False)
    #args = vars(parser.parse_args())
    # Here we have to put code for ensuring the arguments are correct
    # # First, ensure if -r_a is passed, because that means all test of the paper should be runned

    # ##
    # Create the necessary directories
    create_dir(FIGURES_DIR_NAME)

    # Define the rcParams of matplotlib to plot using LaTex font Helvetica
    try:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica"]})
    except Exception as e:
        print(f'Exception {e.__class__} ocurred while trying to set LaTex fonts for matplotlib,'
              f'therefore it will be ignored')
    args = {'ind': ['Fashion_MNIST'],
            'ood': ['MNIST'],
            'model_arch': 'LeNet',
            'load_or_train': 'Load',
            'average_mode': 'Mean',
            'approach': 'g_r',
            'seed': 8,
            'n_heatmaps': 1000
            }
    # Visualize the arguments introduced
    print(args)
    # For every In-Distribution dataset, it has to run
    for in_dataset in args['ind']:
        # Download the datasets
        (train_images, train_labels), (test_images, test_labels), class_names, num_classes = download_or_load_dataset(
            in_dataset)
        # Create model
        model = create_model(args['model_arch'])
        if args['load_or_train'] == 'Load':
            # Load weights
            load_model_weights(model, dataset_name=in_dataset, model_name=args['model_arch'])
        elif args['load_or_train'] == 'Train':
            print()
        else:
            raise NameError('Wrong option between "Load" or "Train" selected')
        metrics = model.evaluate(test_images, test_labels)
        print('Accuracy obtained is', str(round(metrics[1] * 100, 2)) + '%')
        train_od_detector(in_dataset=in_dataset,
                          args=args,
                          train_images_and_labels = (train_images, train_labels),
                          model=model,
                          class_names=class_names
        )


if __name__ == '__main__':
    # Data range constant for SSIM defined in the utils.py
    # Run main code
    main()
