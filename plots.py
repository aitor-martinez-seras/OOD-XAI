import os.path

import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
from skimage.transform import rotate
from constants import *
from utils import search_index_of_class

def plot_average_heatmaps(average_heatmaps_per_class_and_cluster, class_names: list, fig_name: str, superimposed=False,
                          train_images_and_labels=None):
    '''
    Generates and saves the plot of the average heatmaps
    :param average_heatmaps_per_class_and_cluster: list of shape = [class, (cluster, height, width)]
    :param class_names:
    :param fig_name:
    :param superimposed:
    :param train_images_and_labels:
    :return:
    '''
    if superimposed is True:
        train_images, train_labels = train_images_and_labels
        fig_name = fig_name + '_superimposed.pdf'
    else:
        fig_name = fig_name + '.pdf'
    # Calculate the number of total columns (the max number cluster along all the classes)
    max_cols = []
    for average_heatmaps_per_class in average_heatmaps_per_class_and_cluster:
        max_cols.append(len(average_heatmaps_per_class))
    max_cols = max(max_cols)
    fontsize = 10 + 2 * max_cols
    # Plots
    fig, ax = plt.subplots(nrows=len(average_heatmaps_per_class_and_cluster), ncols=max_cols,
                           figsize=(3 * max_cols, 3 * len(average_heatmaps_per_class_and_cluster)),
                           sharex='all', sharey='all')
    for class_index in range(len(average_heatmaps_per_class_and_cluster)):
        for cluster_index, average_heatmap in enumerate(average_heatmaps_per_class_and_cluster[class_index]):
            if superimposed is False:
                im = ax[class_index, cluster_index].imshow(average_heatmap, cmap='jet', vmin=0, vmax=1)
            elif superimposed is True:
                one_image_of_class_index = search_index_of_class(class_index, 1, train_labels)[0]
                ax[class_index, cluster_index].imshow(train_images[one_image_of_class_index, :, :, 0])
                # Save AxesImage for plotting the colorbar
                im = ax[class_index, cluster_index].imshow(
                    resize(average_heatmap,
                           (train_images.shape[1:3])), alpha=0.6, cmap='jet', vmin=0, vmax=1)

            # For the first row, p
            if class_index == 0:
                ax[class_index, cluster_index].set_title('Cluster {}'.format(cluster_index), fontsize=fontsize)
                ax[class_index, cluster_index].set_xticks([])
                ax[class_index, cluster_index].set_yticks([])

            # Comment the line in the if to not plot the class names
            if cluster_index == 0:
                ax[class_index, cluster_index].set_ylabel(class_names[class_index].title(), rotation=90,
                                                          fontsize=fontsize)
                pass

        if len(average_heatmaps_per_class_and_cluster[class_index]) != max_cols:
            for empty_ax_index in range(len(average_heatmaps_per_class_and_cluster[class_index]), max_cols):
                ax[class_index][empty_ax_index].imshow(
                    np.zeros(np.shape(average_heatmap)), vmin=0, vmax=1)
                ax[class_index, empty_ax_index].plot(
                    [0, len(average_heatmap) - 1],
                    [len(average_heatmap) - 1, 0], c='w', lw=2)
                ax[class_index, empty_ax_index].plot(
                    [0, len(average_heatmap) - 1],
                    [0, len(average_heatmap) - 1], c='w', lw=2)
                if class_index == 0:
                    ax[class_index, empty_ax_index].set_title('Cluster {}'.format(empty_ax_index), fontsize=fontsize)
                    ax[class_index, empty_ax_index].set_xticks([])
                    ax[class_index, empty_ax_index].set_yticks([])

    plt.subplots_adjust(wspace=0.1, hspace=0.05)
    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.08, aspect=30)
    cbar.ax.tick_params(labelsize=12 + max_cols * 4)
    path_plot = os.path.join(FIGURES_DIR_NAME, fig_name)
    plt.savefig(path_plot, bbox_inches='tight')
