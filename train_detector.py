import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras import layers
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
from constants import *


def train_od_detector(in_dataset: str, args: dict, train_images_and_labels: tuple, model: keras.Model,
                      class_names: list):
    '''
    Computes the training of the Out-of-Distribution detector by creating a clusterized space by using SSIM Distance
    :param in_dataset: str of the In-Distribution dataset name
    :param args: arguments introduced by the user in the main programm execution
    :param train_images_and_labels: tuple with the train images and labels (height, width, channels)
    :param model: model instance already trained
    :return: saves the arrays in the objects directory
    '''
    create_dir(OBJECTS_DIR_NAME)
    # First, generate the heatmaps to create the cluster space for each class
    file_name_heatmaps_train_per_class = f'heatmaps_train_per_class_{in_dataset}_{args["model_arch"]}' \
                                         f'_{args["load_or_train"]}_seed{args["seed"]}.npy'
    path_heatmaps_train_per_class = os.path.join(OBJECTS_DIR_NAME, file_name_heatmaps_train_per_class)
    if os.path.isfile(path_heatmaps_train_per_class):
        heatmaps_train_per_class = np.load(path_heatmaps_train_per_class)
        print('Heatmaps loaded from file')
    else:
        print('Heatmap generation:')
        # Unpack and shuffle
        train_images, train_labels = train_images_and_labels
        train_images_shuffled, train_labels_shuffled = shuffle(train_images, train_labels, random_state=args['seed'])
        # Creation of the array with N heatmaps per class. This heatmaps are the ones used for creating the clusters.
        # heatmaps_train_per_class.shape = (number_of_classes, number_of_heatmaps, height, width)
        heatmaps_train_per_class = creation_of_heatmaps_per_class(args['n_heatmaps'], train_images_shuffled,
                                                                  train_labels_shuffled, model, class_names)
        np.save(path_heatmaps_train_per_class, heatmaps_train_per_class, allow_pickle=False)
        print('Train heatmaps generated and saved successfully!')

    # Second, compute the pairwise SSIM distances
    file_name_pairwise_dist_matrix = f'ssim_distance_matrix_per_class_{in_dataset}_{args["model_arch"]}' \
                                     f'_{args["load_or_train"]}_seed{args["seed"]}.npy'
    path_pairwise_dist_matrix = os.path.join(OBJECTS_DIR_NAME, file_name_pairwise_dist_matrix)
    if os.path.isfile(path_pairwise_dist_matrix):
        ssim_distance_matrix_per_class = np.load(path_pairwise_dist_matrix)
        print('\nDistance matrix loaded from file')
    else:
        print('Pairwise distances matrix computation:')
        ssim_distance_matrix_per_class = np.zeros(
            (len(class_names), heatmaps_train_per_class.shape[1], heatmaps_train_per_class.shape[1]))
        for class_index in tqdm(range(len(heatmaps_train_per_class))):
            ssim_distance_matrix_per_class[class_index] = ssim_distance_matrix_creation(
                heatmaps_train_per_class[class_index])
        np.save(path_pairwise_dist_matrix, ssim_distance_matrix_per_class, allow_pickle=False)
        print('Pairwise SSIM distance matrix computed successfully!')

    file_name_clustering = f'clustering_labels_per_class_{in_dataset}_{args["model_arch"]}' \
                           f'_{args["load_or_train"]}_seed{args["seed"]}.npy'
    path_clustering = os.path.join(OBJECTS_DIR_NAME, file_name_clustering)
    if os.path.isfile(path_clustering):
        print('Clustering already exist!')
    else:
        print('Generating the cluster space:')
        clustering_labels_per_class = create_cluster_space_for_each_class_agglomerative_clustering(
            ssim_distance_matrix_per_class,
            class_names,
            in_dataset,
            args
        )
        np.save(path_clustering, clustering_labels_per_class, allow_pickle=False)
        print('Cluster space created and saved!')


def create_cluster_space_for_each_class_agglomerative_clustering(ssim_distance_matrix_per_class, class_names: list,
                                                                 in_dataset: str, args: dict):
    '''
    Generates the cluster space
    :param in_dataset:
    :param args:
    :param class_names:
    :param ssim_distance_matrix_per_class:
    :return:
    '''
    # First:
    # Definition of the distance threshold (parameter for the Agglomerative clustering) for each class
    # Done by optimizing the silhouette score for a given range of threshold distances
    print(
        'Selecting the parameter "distance_threshold" of Agglomerative Clustering by optimizing the Silhouette score\n')
    distance_threshold = []
    range_of_dist_thrs = np.linspace(0.52, 0.90, 39)
    silhouette_scores_per_class = []
    clustering_labels_per_class = np.zeros((len(class_names), ssim_distance_matrix_per_class.shape[1]))
    for class_index in tqdm(range(len(class_names))):
        silh_scores_one_class = []
        for distance in range_of_dist_thrs:
            # For each distance threshold, we compute the clustering and the silhouette score
            cluster_model = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='complete',
                                                    distance_threshold=distance)
            cluster_model.fit(ssim_distance_matrix_per_class[class_index])
            clustering_labels_per_class[class_index] = cluster_model.labels_
            try:
                silh_scores_one_class.append(silhouette_score(ssim_distance_matrix_per_class[class_index],
                                                              clustering_labels_per_class[class_index],
                                                              metric='precomputed'))
            except ValueError:
                silh_scores_one_class.append(0)
        silhouette_scores_per_class.append(silh_scores_one_class)

        positions = []
        max_score = max(silh_scores_one_class)
        for index, score in enumerate(silh_scores_one_class[::-1]):
            if score == max_score:
                positions.append(index)
        # We select the max silhouette score closer to distance_threshold = 0
        distance_threshold.append(range_of_dist_thrs[len(range_of_dist_thrs) - positions[-1] - 1])

    print('')
    # Save the plot with silhouette scores
    plt.subplots(2, 5, figsize=(25, 10))
    for class_index, position in enumerate(range(1, 11)):
        plt.subplot(2, 5, position).plot(range_of_dist_thrs, silhouette_scores_per_class[class_index], color='red')
        plt.title(class_names[class_index])
    silhouette_path_pdf = os.path.join(FIGURES_DIR_NAME, f'silhouetteScores_{in_dataset}_{args["model_arch"]}'
                                                         f'_{args["load_or_train"]}_seed{args["seed"]}.pdf')
    plt.savefig(silhouette_path_pdf)
    # plt.show()

    # Function that creates de clusters of each class
    # Initialization of the array containing the labels of the labels for each image in each class
    clustering_labels_per_class = np.zeros((len(class_names), ssim_distance_matrix_per_class.shape[1]))
    # plot the top three levels of the dendrogram
    w = 40
    h = 15
    fig, ax = plt.subplots(2, 5, figsize=(w, h))
    fig.suptitle('Hierarchical Clustering Dendrogram', fontsize=h + w * 0.1, y=0.94)
    for class_index in range(len(class_names)):
        if isinstance(distance_threshold, list):
            cluster_model = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='complete',
                                                    distance_threshold=distance_threshold[class_index])
        else:
            cluster_model = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='complete',
                                                    distance_threshold=distance_threshold)
        cluster_model.fit(ssim_distance_matrix_per_class[class_index])
        clustering_labels_per_class[class_index] = cluster_model.labels_
        if class_index < 5:
            i = 0
            j = class_index
        else:
            i = 1
            j = class_index - 5
        plot_dendrogram(cluster_model, truncate_mode='level', p=2, ax=ax[i, j])
        ax[i, j].set_title('Class {}'.format(class_names[class_index]), fontsize=h)
        # ax[i,j].set_xlabel("Number of points in node",fontsize=h)
    dendrogram_path_pdf = os.path.join(FIGURES_DIR_NAME, f'DendrogramPerClass_{in_dataset}_{args["model_arch"]}'
                                                         f'_{args["load_or_train"]}_seed{args["seed"]}.pdf')
    plt.savefig(dendrogram_path_pdf)
    # fig.show()
    print('-' * 100, '\n')
    print('Cluster per class:')
    print('------------------\n')
    for class_index in range(len(class_names)):
        unique, counts = np.unique(clustering_labels_per_class[class_index], return_counts=True)
        print('Class', class_names[class_index].ljust(15), '\t', dict(zip(unique, counts)))
        print('-' * 75)
    print('\n' + '-' * 100)
    return clustering_labels_per_class
