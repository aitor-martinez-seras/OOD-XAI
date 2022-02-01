# Imports
import os
import pickle
from tqdm import tqdm
import numpy as np
from constants import *
from utils import compute_average_heatmaps_per_cluster


def test_od_detector(in_dataset: str, ood_dataset: str, test_heatmaps: np.ndarray,
                     ood_heatmaps: np.ndarray, model, model_arch: str,
                     average_mode: str or int, comp_funct: str):
    return


def create_or_load_average_heatmaps(in_dataset: str, model_arch: str, args: dict, average_mode: str):

    # Different name depending on the average mode
    if isinstance(average_mode, float):
        percentage_threshold = average_mode
        average_mode = 'Percentage'
        file_name_average_heatmaps = f'average_heatmaps_per_class_and_cluster_{in_dataset}_{model_arch}' \
                                     f'_{args["load_or_train"]}_seed{args["seed"]}_percent{average_mode}.pkl'
    else:
        percentage_threshold = None
        file_name_average_heatmaps = f'average_heatmaps_per_class_and_cluster_{in_dataset}_{model_arch}' \
                                 f'_{args["load_or_train"]}_seed{args["seed"]}_{average_mode}.pkl'
    path_heatmaps_average_heatmaps = os.path.join(OBJECTS_DIR_NAME, file_name_average_heatmaps)
    # Checks if it exists
    if os.path.isfile(path_heatmaps_average_heatmaps):
        print('File exist, it will be loaded')
        with open(path_heatmaps_average_heatmaps, "rb") as f:
            average_heatmaps_per_class_and_cluster = pickle.load(f)
    else:
        # Load the trained subspaces
        # Heatmaps train
        file_name_heatmaps_train_per_class = f'heatmaps_train_per_class_{in_dataset}_{model_arch}' \
                                             f'_{args["load_or_train"]}_seed{args["seed"]}.npy'
        path_heatmaps_train_per_class = os.path.join(OBJECTS_DIR_NAME, file_name_heatmaps_train_per_class)
        heatmaps_train_per_class = np.load(path_heatmaps_train_per_class)
        # SSIM matrix
        file_name_pairwise_dist_matrix = f'ssim_distance_matrix_per_class_{in_dataset}_{model_arch}' \
                                         f'_{args["load_or_train"]}_seed{args["seed"]}.npy'
        path_pairwise_dist_matrix = os.path.join(OBJECTS_DIR_NAME, file_name_pairwise_dist_matrix)
        ssim_distance_matrix_per_class = np.load(path_pairwise_dist_matrix)
        # Clustering
        file_name_clustering = f'clustering_labels_per_class_{in_dataset}_{model_arch}' \
                               f'_{args["load_or_train"]}_seed{args["seed"]}.npy'
        path_clustering = os.path.join(OBJECTS_DIR_NAME, file_name_clustering)
        clustering_labels_per_class = np.load(path_clustering)

        # Compute the averages
        average_heatmaps_per_class_and_cluster = []
        print('Computing the averages:')
        for class_index in tqdm(range(len(heatmaps_train_per_class))):
            average_heatmaps_per_class_and_cluster.append(
                compute_average_heatmaps_per_cluster(clustering_labels_per_class[class_index],
                                                     heatmaps_train_per_class[class_index],
                                                     ssim_distance_matrix_per_class[class_index],
                                                     avg_mode=average_mode,
                                                     thr=percentage_threshold))
        with open(path_heatmaps_average_heatmaps, "wb") as f:
            pickle.dump(average_heatmaps_per_class_and_cluster, f)

    return average_heatmaps_per_class_and_cluster
