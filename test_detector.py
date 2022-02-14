# Imports
import os
import pickle
import warnings
import pandas as pd
from tqdm import tqdm
import numpy as np
from constants import *
from utils import create_dir, compute_average_heatmaps_per_cluster, \
    compute_ssim_against_cluster_averages, compute_ssim_against_all_heatmaps_of_closest_cluster, \
    similarity_thresholds_for_each_TPR, compute_precision_tpr_fpr_for_test_and_OoD_similarity
from plots import plot_histograms_per_class_test_vs_ood, plot_AUROC, plot_AUPR


def test_od_detector(average_heatmaps_per_class_and_cluster, in_dataset: str, ood_dataset: str,
                     test_heatmaps: np.ndarray, test_predictions: np.ndarray,
                     ood_heatmaps: np.ndarray, ood_predictions: np.ndarray, model, model_arch: str,
                     average_mode: str or int, comp_funct: str, class_names: list, args: dict):

    # Load the heatmaps train and the clustering labels
    file_name_heatmaps_train_per_class = f'heatmaps_train_per_class_{in_dataset}_{model_arch}' \
                                         f'_{args["load_or_train"]}_seed{args["seed"]}.npy'
    path_heatmaps_train_per_class = os.path.join(OBJECTS_DIR_NAME, file_name_heatmaps_train_per_class)
    heatmaps_train_per_class = np.load(path_heatmaps_train_per_class)
    file_name_clustering = f'clustering_labels_per_class_{in_dataset}_{model_arch}' \
                           f'_{args["load_or_train"]}_seed{args["seed"]}.npy'
    path_clustering = os.path.join(OBJECTS_DIR_NAME, file_name_clustering)
    clustering_labels_per_class = np.load(path_clustering)

    # Depending on the function, execute different functions
    if comp_funct == 'g_r':
        ssim_per_image_test = compute_ssim_against_cluster_averages(test_heatmaps, test_predictions,
                                                                    average_heatmaps_per_class_and_cluster)
        ssim_per_image_ood = compute_ssim_against_cluster_averages(ood_heatmaps, ood_predictions,
                                                                  average_heatmaps_per_class_and_cluster)

    elif comp_funct == 'g_all':
        # Test (Compare the test_images of the in-dataset against the clusters)
        file_name_comparison_approach_test = f'ssim_all_heatmaps_of_closest_cluster_{in_dataset}_vs_{in_dataset}' \
                                             f'_{model_arch}_{average_mode}.npy'
        path_comparison_approach_test = os.path.join(OBJECTS_DIR_NAME, file_name_comparison_approach_test)
        if os.path.isfile(path_comparison_approach_test):
            print(f'SSIM values of {in_dataset}_vs_{in_dataset}_{model_arch} exist, they will be loaded')
            ssim_per_image_test = np.load(path_comparison_approach_test)
        else:
            print(
                f'Comparison of {in_dataset} against all heatmaps of the closest cluster of {in_dataset} '
                f'is being performed:')
            ssim_per_image_test = compute_ssim_against_all_heatmaps_of_closest_cluster(
                test_heatmaps, test_predictions,
                average_heatmaps_per_class_and_cluster,
                heatmaps_train_per_class,
                clustering_labels_per_class)
            np.save(path_comparison_approach_test, ssim_per_image_test, allow_pickle=False)
        # OOD
        file_name_comparison_approach_od = f'ssim_all_heatmaps_of_closest_cluster_{in_dataset}_vs_{ood_dataset}' \
                                           f'_{model_arch}_{average_mode}.npy'
        path_comparison_approach_od = os.path.join(OBJECTS_DIR_NAME, file_name_comparison_approach_od)
        if os.path.isfile(path_comparison_approach_od):
            print(f'SSIM values of {ood_dataset}_{model_arch} exist, they will be loaded')
            ssim_per_image_ood = np.load(path_comparison_approach_od)
        else:
            print(
                f'Comparison of {ood_dataset} against all heatmaps of the closest cluster of {in_dataset} '
                f'is being performed:')
            ssim_per_image_ood = compute_ssim_against_all_heatmaps_of_closest_cluster(
                ood_heatmaps, ood_predictions,
                average_heatmaps_per_class_and_cluster,
                heatmaps_train_per_class,
                clustering_labels_per_class)
            np.save(path_comparison_approach_od, ssim_per_image_ood, allow_pickle=False)

    else:
        raise NameError(f'{comp_funct} comparison function does not exist.')

    # Create the figure with histograms per class
    fig_name = f'Histograms_{in_dataset}_vs_{ood_dataset}_{average_mode}_{comp_funct}_{model_arch}_' \
               f'{args["load_or_train"]}_seed{args["seed"]}.pdf'
    plot_histograms_per_class_test_vs_ood(ssim_per_image_test, test_predictions,
                                          ssim_per_image_ood, ood_predictions,
                                          class_names, fig_name)

    # Compute and save the AUROC and the AUPR
    compute_and_save_results(ssim_per_image_test, test_predictions,
                             ssim_per_image_ood, ood_predictions,
                             in_dataset, ood_dataset, average_mode, comp_funct,
                             model_arch, class_names, args)


def compute_and_save_results(ssim_per_image_test: np.ndarray, test_predictions: np.ndarray,
                             ssim_per_image_ood: np.ndarray, ood_predictions: np.ndarray,
                             in_dataset: str, ood_dataset: str, average_mode: str, comparison_function: str,
                             model_arch: str, class_names: list, args: dict):
    # Create results directory in case it does not exist
    create_dir(RESULTS_DIR_NAME)
    # Creation of the array with the thresholds for each TPR (class, dist_per_TPR)
    ssim_per_class_per_image_test = [ssim_per_image_test[np.where(test_predictions == class_index)] for class_index in
                                     range(len(class_names))]
    ssim_per_class_per_image_od = [ssim_per_image_ood[np.where(ood_predictions == class_index)] for class_index in
                                   range(len(class_names))]
    similarity_thresholds_test = similarity_thresholds_for_each_TPR(ssim_per_class_per_image_test)
    # Computing precision, tpr and fpr
    precision, tpr_values, fpr_values = compute_precision_tpr_fpr_for_test_and_OoD_similarity(
        ssim_per_class_per_image_test,
        ssim_per_class_per_image_od,
        similarity_thresholds_test
    )

    # ROC Curve
    # Appending that when FPR = 1 the TPR is also 1:
    tpr_values_auroc = np.append(tpr_values, 1)
    fpr_values_auroc = np.append(fpr_values, 1)
    # Compute FPR when TPR is 95% and 80%
    fpr_95 = round(fpr_values_auroc[int(len(fpr_values_auroc) * 0.95)], 4)
    fpr_80 = round(fpr_values_auroc[int(len(fpr_values_auroc) * 0.80)], 4)
    # AUC
    auroc = round(np.trapz(tpr_values_auroc, fpr_values_auroc), 4)

    # PR Curve
    # AUC
    aupr = round(np.trapz(precision, tpr_values), 4)

    # Warning about the separator and the decimal for the .csv
    warnings.warn(f"The .csv file has the separator defined as '{CSV_SEPARATOR}' and the decimal as '{CSV_DECIMAL}'."
                  f"This is for visualization purposes when opening the file in Excel configured for Spain"
                  f"Please, be aware that this combination of separator and decimal may lead to a bad"
                  f"visualization. To change them in all the code involved, go to constants.py file.")

    # Create or update the results file
    # File path
    results_file_name = f'results_{in_dataset}_vs_{ood_dataset}_{model_arch}_' \
                        f'{args["load_or_train"]}_seed{args["seed"]}.csv'
    results_file_path = os.path.join(RESULTS_DIR_NAME, results_file_name)
    # File columns
    CSV_COLUMNS = ['InD', 'OutD', 'Average mode', 'Comparison function', 'AUROC', 'AUPR', 'FPR95', 'FPR80']
    # Check if file exist for this combination of datasets and model configuration, if yes overwrite it
    if os.path.isfile(results_file_path):
        print(f'Results file exist for the setting {in_dataset} vs {ood_dataset}, with '
              f'the model configuration: {model_arch}, {args["load_or_train"]}')
        df = pd.read_csv(results_file_path, sep=CSV_SEPARATOR, decimal=CSV_DECIMAL)
        # Check if the info to be added is already repeated
        repeated = False
        new_result = [in_dataset, ood_dataset, average_mode, comparison_function, auroc, aupr, fpr_95, fpr_80]
        for index, row in df.iterrows():
            if (df == new_result).all(1).any():
                repeated = True
                break
        if not repeated:
            df.loc[len(df)] = new_result
            # Print info
            print('')
            print('| ---------------------------------- |')
            print('| -------- New result saved -------- |')
            print('| ---------------------------------- |')
        else:
            warnings.warn("Tested setting was already in the .csv, it will not be added again")
    else:
        df = pd.DataFrame(
            data=[[in_dataset, ood_dataset, average_mode, comparison_function, auroc, aupr, fpr_95, fpr_80]],
            columns=CSV_COLUMNS
        )
    # Save or overwrite the .csv file
    df.to_csv(results_file_path, index=False, columns=CSV_COLUMNS, sep=CSV_SEPARATOR, decimal=CSV_DECIMAL)

    # Save results properly in a dict to return them
    results_dict = {
        'tpr_auroc': tpr_values_auroc,
        'fpr_auroc': fpr_values_auroc,
        'precision': precision,
        'tpr_aupr': tpr_values,
        'fpr_95': fpr_95,
        'fpr_80': fpr_80
    }

    # Create the figures
    fig_name = f'{in_dataset}_vs_{ood_dataset}_{model_arch}_{average_mode}_{comparison_function}_' \
               f'{args["load_or_train"]}_seed{args["seed"]}.pdf'
    auroc_fig_name = 'AUROC_' + fig_name
    aupr_fig_name = 'AUPR_' + fig_name
    plot_AUROC(auroc, tpr_values_auroc, fpr_values_auroc, fpr_95, fpr_80, auroc_fig_name)
    plot_AUPR(aupr, tpr_values, precision, aupr_fig_name)


def create_or_load_average_heatmaps(in_dataset: str, model_arch: str, args: dict, average_mode: str):

    # Different name depending on the average mode
    if isinstance(average_mode, float):
        percentage_threshold = average_mode
        average_mode = 'Mean'
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
