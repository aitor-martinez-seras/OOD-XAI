import os.path

import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
from skimage.transform import rotate
from constants import *
from utils import search_index_of_class, cdf_difference
from scipy import stats
from scipy.optimize import minimize_scalar


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


def plot_histograms_per_class_test_vs_ood(ssim_per_image_test, test_predictions, ssim_per_image_od, ood_predictions,
                                          class_names, fig_name):
    # Funcion que representa el % de instancias que SI activan el threshold tanto CON como SIN deriva 
    # Inicializar los rangos para los que se calcula la CDF
    x = np.linspace(-1, 1, 100)
    y = np.linspace(1, -1, 100)

    # Propiedades de la caja donde va el texto
    PROPS = dict(boxstyle='round', facecolor='azure', alpha=0.8)
    MEAN_BOX_PROPS = dict(boxstyle='round', facecolor='springgreen', alpha=0.7)
    MEAN_OUT_D_BOX_PROPS = dict(boxstyle='round', facecolor='lightcoral', alpha=0.7)
    # Creacion de figura
    fig, ax = plt.subplots(2, 5, sharey='all', figsize=(40, 14))
    green_plot = 'In-Distribution'
    red_plot = 'Out-Distribution'
    fig.suptitle('Histogram In-Distribution VS {}'.format(red_plot), fontsize=25, y=0.96)

    for class_index in range(10):
        # The count of how many od images are classified as each class
        unique, ood_counts = np.unique(ood_predictions, return_counts=True)
        # KDE fitting
        kde_sci = stats.gaussian_kde(ssim_per_image_test[np.where(test_predictions == class_index)])
        cdf = []
        for i in x:
            cdf.append(kde_sci.integrate_box_1d(-1, i))
        # KDE fitting
        try:
            if ood_counts[class_index] > 50:
                kde_sci_OOD = stats.gaussian_kde(ssim_per_image_od[np.where(ood_predictions == class_index)])
                cdf_OOD = []
                for i in y:
                    cdf_OOD.append(kde_sci_OOD.integrate_box_1d(i, 1))
        except KeyError:
            raise NameError('Error')
            pass
        # Calculo del punto de cruce minimizando diferencia entre CDFs. La solucion nos da la posicion del punto de minima 
        # diferencia dentro del array, no el valor en si
        sol = minimize_scalar(cdf_difference, args=(kde_sci, kde_sci_OOD), method='bounded', bounds=(1, 99),
                              options={'xatol': 0.001, 'maxiter': 100, 'disp': 0})
        # Punto de cruce es el valor del eje X en la posicion de la solucion 
        p_cruce = x[int(sol.x)]
        tpr_cruce = round(kde_sci.integrate_box_1d(p_cruce, 1), 3)
        fpr_cruce = round(1 - kde_sci_OOD.integrate_box_1d(-1, p_cruce), 3)

        # Definition of i and j for plotting
        i = 0
        j = class_index
        if class_index >= 5:
            i = 1
            j = class_index - 5

        # Plots de KDE
        ax[i, j].set_xlim((-0.1, 1))
        ax[i, j].set_ylim((0, 4))
        ax[i, j].set_yticks([])
        ax[i, j].hist(ssim_per_image_test[np.where(test_predictions == class_index)], density=True, bins=25,
                      color='green', label=green_plot)
        ax[i, j].hist(ssim_per_image_od[np.where(ood_predictions == class_index)], density=True, bins=25, color='red',
                      alpha=0.6, label=red_plot)
        ax[i, j].plot(x, kde_sci.pdf(x), lw=3, label='PDF {}'.format(green_plot), color='limegreen', zorder=1)
        # ax[i,j].plot(x, cdf, lw=3, label='CDF {}'.format(green_plot), color='chartreuse',zorder=2)
        ax[i, j].set_title('Represented class: {}'.format(class_names[class_index]), fontsize=16)

        meanInDistPerClass = round(ssim_per_image_test[np.where(test_predictions == class_index)].mean(), 2)
        ax[i, j].text(x=meanInDistPerClass, y=0.35, s='Mean\n' + str(meanInDistPerClass), fontsize=16,
                      fontweight='medium', bbox=MEAN_BOX_PROPS, horizontalalignment='center', zorder=4)

        # Texto a representar
        text_height = 0.71
        text_str = 'Nº of OD heatmaps: {}'.format(ood_counts[class_index])
        try:
            if ood_counts[class_index] > 50:
                # ax[i,j].plot(y, cdf_OOD, lw=3, label='CDF {}'.format(red_plot), color='orangered',zorder=3)
                # Texto
                meandOutDistPerClass = round(ssim_per_image_od[np.where(ood_predictions == class_index)].mean(), 2)
                ax[i, j].text(x=meandOutDistPerClass, y=0.35, s='Mean\n' + str(meandOutDistPerClass), fontsize=16,
                              fontweight='medium', bbox=MEAN_OUT_D_BOX_PROPS, horizontalalignment='center', zorder=4)
                ax[i, j].text(0.98, text_height, text_str, transform=ax[i, j].transAxes, fontsize=12,
                              horizontalalignment='right', verticalalignment='center', multialignment='left',
                              bbox=PROPS, zorder=4)
            else:
                try:
                    meandOutDistPerClass = round(ssim_per_image_od[np.where(ood_predictions == class_index)].mean(), 2)
                    ax[i, j].text(x=meandOutDistPerClass, y=0.35, s='Mean\n' + str(meandOutDistPerClass), fontsize=16,
                                  fontweight='medium', bbox=MEAN_OUT_D_BOX_PROPS, horizontalalignment='center', zorder=4)

                    ax[i, j].text(0.98, text_height, text_str, transform=ax[i, j].transAxes, fontsize=12,
                                  horizontalalignment='right', verticalalignment='center', multialignment='left',
                                  bbox=PROPS, zorder=4)
                except Exception as e:
                    print(f'Exception {e.__class__} because the number of OOD heatmaps is 0, ignore it.')
                    text_str = 'Nº of OD heatmaps: 0'
                ax[i, j].text(0.98, text_height, text_str, transform=ax[i, j].transAxes, fontsize=12,
                              horizontalalignment='right', verticalalignment='center', multialignment='left',
                              bbox=PROPS, zorder=4)

        except KeyError:
            text_str = 'Nº of OD heatmaps: 0'
            ax[i, j].text(0.98, text_height, text_str, transform=ax[i, j].transAxes, fontsize=12,
                          horizontalalignment='right', verticalalignment='center', multialignment='left', bbox=PROPS,
                          zorder=4)
        # ax[i,j].axvline(x=p_cruce, ymin=0, ymax=1,color='fuchsia',label='Threshold',linewidth=3,linestyle='--',zorder=5)
        ax[i, j].legend(fontsize=14, loc='upper right')
    fig_path = os.path.join(FIGURES_DIR_NAME, fig_name)
    fig.savefig(fig_path, bbox_inches='tight')


def plot_AUROC(auroc_value, tpr_values_auroc, fpr_values_auroc, fpr_95, fpr_80, fig_name):
    # ROC Curve
    # Plot
    plt.figure(figsize=(15, 12))
    plt.plot(fpr_values_auroc, tpr_values_auroc, label='ROC curve', lw=3)
    plt.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), 'k--', label='Random ROC curve')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('FPR', fontsize=20)
    plt.ylabel('TPR', fontsize=20)
    plt.title('ROC curve, AUC = %.4f' % auroc_value, fontsize=25, pad=10)
    plt.fill_between(fpr_values_auroc, tpr_values_auroc, alpha=0.3)
    plt.plot([], [], ' ', label=f'FPR at 95% TPR = {fpr_95 * 100:.2f}%')
    plt.plot([], [], ' ', label=f'FPR at 80% TPR = {fpr_80 * 100:.2f}%')
    # plt.text(0.60,0.975,'FPR at 95% TPR = {}%'.format(round(array_TPR_FPR_x_threshold[95,1]*100,2)),fontsize=20,bbox=dict(boxstyle="round",facecolor='white', alpha=0.5))
    plt.legend(fontsize=20, loc='upper left')
    fig_path = os.path.join(FIGURES_DIR_NAME, fig_name)
    plt.savefig(fig_path)


def plot_AUPR(aupr_value, tpr_values, precision, fig_name):
    # PR Curve
    # Plot
    plt.figure(figsize=(15, 12))
    plt.plot(tpr_values, precision, label='PR curve', lw=3)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('FPR', fontsize=20)
    plt.ylabel('Precision', fontsize=20)
    plt.title('PR curve, AUC = %.4f' % aupr_value, fontsize=25, pad=10)
    plt.fill_between(tpr_values, precision, alpha=0.3)
    plt.legend(fontsize=20, loc='upper left')
    fig_path = os.path.join(FIGURES_DIR_NAME, fig_name)
    plt.savefig(fig_path)





























