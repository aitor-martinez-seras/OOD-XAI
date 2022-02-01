import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
# Utils
import warnings
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
from test_detector import test_od_detector, create_or_load_average_heatmaps
from plots import plot_average_heatmaps


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
                        choices=IN_D_CHOICES, required=False)
    parser.add_argument('-o', '--ood', type=str, help='out of distribution dataset', nargs='+',
                        choices=OUT_D_CHOICES, required=False)
    parser.add_argument('-m', '--model_arch', type=str, choices=['LeNet', 'ResNet32'],
                        help='model architecture, only one a each call', required=False)
    parser.add_argument('-l_or_t', '--load_or_train', type=str, choices=['Load', 'Train'],
                        help='model architecture', required=False)
    parser.add_argument('-avg', '--average_mode', type=str, nargs='+',
                        help='average modes to be computed: Possible choices are Mean, Median or '
                             'an integer representing the percentage', required=False)
    parser.add_argument('-comp_f', '--comparison_function', type=str, choices=['g_r', 'g_all'], nargs='+',
                        help='comparison functions to be computed', required=False)
    parser.add_argument('-s', '--seed', type=int, help='Seed for shuffling the train images and labels', required=False)
    parser.add_argument('-n_htmaps', '--n_heatmaps', type=int, help='Select the number of heatmaps per class '
                                                                    'for the clustering', required=False)
    # args = vars(parser.parse_args())
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
        print(f'Exception {e.__class__} occurred while trying to set LaTex fonts for matplotlib,'
              f'therefore it will be ignored')
    args = {'ind': ['Fashion_MNIST'],
            'ood': ['MNIST'],
            'model_arch': ['LeNet'],
            'load_or_train': 'Load',
            'average_mode': 'Mean',
            'comparison_function': 'g_r',
            'seed': 8,
            'n_heatmaps': 1000
            }
    # Visualize the arguments introduced
    print(args)
    # Classify datasets taking into account the format of data
    mnist_like_ind_datasets = [dataset for dataset in args['ind'] if dataset in MNIST_LIKE_DATASETS]
    cifar10_like_ind_datasets = [dataset for dataset in args['ind'] if dataset in CIFAR10_LIKE_DATASETS]
    mnist_like_ood_datasets = [dataset for dataset in args['ood'] if dataset in MNIST_LIKE_DATASETS]
    cifar10_like_ood_datasets = [dataset for dataset in args['ood'] if dataset in CIFAR10_LIKE_DATASETS]

    # Initiate the main loop
    for model_arch in args["model_arch"]:

        # Compute test only on in distributions compatible with the model architecture
        if model_arch in MODELS_FOR_MNIST:
            in_datasets = mnist_like_ind_datasets
        elif model_arch in MODELS_FOR_CIFAR10:
            in_datasets = cifar10_like_ind_datasets
        else:
            raise NameError(f'Model {model_arch} does not exist, please include it in the constants.py file')

        # If there is no dataset compatible with that model architecture, raise a warning and go to next iteration
        if in_datasets == []:
            warnings.warn(f'No in distribution dataset is compatible with {model_arch} architecture, be aware that the'
                          f'following in distribution dataset are not going to be computed: {str(in_datasets)}')
            continue
        else:
            print(f'Following in distribution datasets are going to be simulated for the model {model_arch}:', end=' ')
            for in_dataset in in_datasets:
                print(in_dataset, end=', ')
            print('')

        # For every In-Distribution dataset, it has to run all the tests
        for in_dataset in in_datasets:

            # Download the datasets
            (train_images, train_labels), (test_images, test_labels), class_names = download_or_load_dataset(
                in_dataset)

            # Create model
            model = create_model(model_arch)

            # Load weights or train the model
            if args['load_or_train'] == 'Load':
                # Load weights
                load_model_weights(model, dataset_name=in_dataset, model_name=model_arch)
            elif args['load_or_train'] == 'Train':
                print()
            else:
                raise NameError('Wrong option between "Load" or "Train" selected')

            # Print the accuracy of the model for the in_dataset
            metrics = model.evaluate(test_images, test_labels)
            print('Accuracy obtained is', str(round(metrics[1] * 100, 2)) + '%')

            # Train the Out-of-Distribution detector by creating the cluster space
            train_od_detector(in_dataset=in_dataset,
                              args=args,
                              train_images_and_labels=(train_images, train_labels),
                              model=model,
                              model_arch=model_arch,
                              class_names=class_names
                              )

            # Generate test heatmaps of the in_dataset
            file_name_heatmaps_test = f'heatmaps_ood_{in_dataset}_{model_arch}' \
                                      f'_{args["load_or_train"]}_seed{args["seed"]}.npy'
            path_heatmaps_test = os.path.join(OBJECTS_DIR_NAME, file_name_heatmaps_test)
            if os.path.isfile(path_heatmaps_test):
                test_heatmaps = np.load(path_heatmaps_test)
                print(f'Test heatmaps of {in_dataset} exist, they have been loaded from file!')
            else:
                print('Heatmap generation:')
                test_predictions = np.argmax(model.predict(test_images), axis=1)
                test_heatmaps = generate_heatmaps(test_images, test_predictions, model)
                np.save(path_heatmaps_test, test_heatmaps, allow_pickle=False)

            # Configure the ood_datasets to be the ones which match the format of the in_dataset
            if in_dataset in MNIST_LIKE_DATASETS:
                ood_datasets = mnist_like_ood_datasets
                assert model_arch in MODELS_FOR_MNIST, \
                    f'{model_arch} not compatible with selected datasets'
            else:
                ood_datasets = cifar10_like_ood_datasets
                assert model_arch in MODELS_FOR_CIFAR10, \
                    f'{model_arch} not compatible with selected datasets'

            # For every OoD dataset, the required approaches will be computed
            for ood_dataset in ood_datasets:

                # If in_dataset and ood_dataset are the same, go to next iteration
                if in_dataset == ood_dataset:
                    continue

                # Generate the OoD heatmaps of the ood_dataset
                file_name_heatmaps_ood = f'heatmaps_ood_{ood_dataset}_{model_arch}' \
                                         f'_{args["load_or_train"]}_seed{args["seed"]}.npy'
                path_heatmaps_ood = os.path.join(OBJECTS_DIR_NAME, file_name_heatmaps_ood)
                if os.path.isfile(path_heatmaps_ood):
                    ood_heatmaps = np.load(path_heatmaps_ood)
                    print(f'OoD heatmaps of {ood_dataset} exist, they have been loaded from file!')
                else:
                    print('Heatmap generation:')
                    ood_images = download_or_load_dataset(ood_dataset, only_test_images=True)
                    ood_predictions = np.argmax(model.predict(ood_images), axis=1)
                    ood_heatmaps = generate_heatmaps(ood_images, ood_predictions, model)
                    np.save(path_heatmaps_ood, ood_heatmaps, allow_pickle=False)

                # Compute all the approaches for every combination of in and out distribution dataset
                for average_mode in args['average_mode']:

                    # If the average_mode is a percentage, we must convert it to it
                    if isinstance(average_mode, int):
                        fig_name = f'percent{average_mode}_average_heatmaps_per_class_and_cluster_{in_dataset}' \
                                   f'_{model_arch}_{args["load_or_train"]}_seed{args["seed"]}'
                        average_mode = average_mode * 0.01
                    else:
                        fig_name = f'{average_mode}_average_heatmaps_per_class_and_cluster_{in_dataset}' \
                                   f'_{model_arch}_{args["load_or_train"]}_seed{args["seed"]}'

                    # Compute or load the average
                    average_heatmaps_per_class_and_cluster = create_or_load_average_heatmaps(
                        in_dataset,
                        model_arch,
                        args,
                        average_mode
                    )

                    # Create the plot for the average heatmaps
                    plot_average_heatmaps(average_heatmaps_per_class_and_cluster, class_names, fig_name,
                                          superimposed=False)
                    for comp_funct in args['comparison_function']:
                        if isinstance(average_mode, float) and comp_funct == 'g_all':
                            pass
                        else:
                            test_od_detector(in_dataset,
                                             ood_dataset,
                                             test_heatmaps,
                                             ood_heatmaps,
                                             model,
                                             model_arch,
                                             average_mode,
                                             comp_funct,
                                             )


if __name__ == '__main__':
    # Data range constant for SSIM defined in the utils.py
    # Run main code
    main()
