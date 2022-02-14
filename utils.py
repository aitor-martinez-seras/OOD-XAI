import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
# Utils
from tqdm import tqdm
# Image processing and transformations
from skimage.transform import resize
from skimage.transform import rotate
from skimage import transform as trfm
from skimage.metrics import structural_similarity as ssim
from skimage import color
# To handle files
import os
import requests
from zipfile import ZipFile
# Scipy funcitons
from scipy.io import loadmat
from scipy.cluster.hierarchy import dendrogram
# Constants
from constants import *


def create_dir(dir_name):
    '''
    Creates a directory making sure that it does not exist
    :return:
    '''
    try:
        if os.path.isdir(dir_name):
            print(f'Directory "{dir_name}" already exist')
        else:
            os.mkdir(dir_name)
        print(f'Successfully created "{dir_name}"')
    except Exception as e:
        print(f'Exception {e.__class__} occurred while creating the {dir_name} directory')


def grad_cam_plus(img, model, layer_name, label_name=None, category_id=None):
    """Get a heatmap by Grad-CAM.
    Args:
        model: A model object, build from tf.keras 2.X.
        img: An image ndarray.
        layer_name: A string, layer name in model.
        label_name: A list,
            show the label name by assign this argument,
            it should be a list of all label names.
        category_id: An integer, index of the class.
            Default is the category with the highest score in the prediction.
    Return:
        A heatmap ndarray(without color).
    """
    img_tensor = np.expand_dims(img, axis=0)
    conv_layer = model.get_layer(layer_name)
    heatmap_model = tf.keras.Model([model.inputs], [conv_layer.output, model.output])
    with tf.GradientTape() as gtape1:
        with tf.GradientTape() as gtape2:
            with tf.GradientTape() as gtape3:
                conv_output, predictions = heatmap_model(img_tensor)  # De aqui se obtiene (8,8,640)
                if category_id is None:
                    category_id = np.argmax(predictions[0])
                if label_name:
                    print(label_name[category_id])
                output = predictions[:, category_id]
                conv_first_grad = gtape3.gradient(output, conv_output)
            conv_second_grad = gtape2.gradient(conv_first_grad, conv_output)
        conv_third_grad = gtape1.gradient(conv_second_grad, conv_output)
    global_sum = np.sum(conv_output, axis=(0, 1, 2))
    # Pixel importance weight
    alpha_num = conv_second_grad[0]
    alpha_denom = conv_second_grad[0] * 2.0 + conv_third_grad[0] * global_sum
    alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, 1e-10)  # To avoid dividing by zero
    alphas = alpha_num / alpha_denom
    alpha_normalization_constant = np.sum(alphas, axis=(0, 1))
    alphas = tf.math.divide_no_nan(alphas, alpha_normalization_constant)
    weights = np.maximum(conv_first_grad[0], 0.0)  # ReLU to gradients
    # Neuron Importance Weights
    deep_linearization_weights = np.sum(weights * alphas, axis=(0, 1))
    grad_CAM_map = np.sum(deep_linearization_weights * conv_output[0], axis=2)
    # Only grab positive values
    heatmap = np.maximum(grad_CAM_map, 0)
    max_heat = np.max(heatmap)
    heatmap = tf.math.divide_no_nan(heatmap, max_heat)
    return heatmap


def grad_cam(img, model,
             layer_name="block5_conv3", label_name=None,
             category_id=None):
    """Get a heatmap by Grad-CAM.
    Args:
        model: A model object, build from tf.keras 2.X.
        img: An image ndarray.
        layer_name: A string, layer name in model.
        label_name: A list,
            show the label name by assign this argument,
            it should be a list of all label names.
        category_id: An integer, index of the class.
            Default is the category with the highest score in the prediction.
    Return:
        A heatmap ndarray(without color).
    """
    img_tensor = np.expand_dims(img, axis=0)

    conv_layer = model.get_layer(layer_name)
    heatmap_model = keras.Model([model.inputs], [conv_layer.output, model.output])

    with tf.GradientTape() as gtape:
        conv_output, predictions = heatmap_model(img_tensor)
        if category_id == None:
            category_id = np.argmax(predictions[0])
        if label_name:
            print(label_name[category_id])
        output = predictions[:, category_id]
        grads = gtape.gradient(output, conv_output)

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    max_heat = np.max(heatmap)
    if max_heat == 0:
        max_heat = 1e-10
    heatmap /= max_heat
    return np.squeeze(heatmap)


def plot_historia(history):
    '''
  Plot of the training
  '''
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1, 1)  # obtener número de epochs del eje X

    plt.plot(epochs, acc, 'r--', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and Validation Accuracy')
    plt.ylabel('acc')
    plt.xlabel('epochs')
    plt.legend()
    plt.figure()

    plt.plot(epochs, loss, 'r--', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')

    plt.legend()
    plt.figure()


def cdf_difference(x, kde_ascending, kde_descending):
    '''
  Function that calculates the difference between two kdes
  '''
    cdf_ascending = []
    x_0 = np.linspace(-1, 1, 100)
    for i in x_0:
        cdf_ascending.append(kde_ascending.integrate_box_1d(-1, i))

    y_0 = x_0[::-1]
    cdf_descending = []
    for i in y_0:
        cdf_descending.append(kde_descending.integrate_box_1d(i, 1))
    difference = cdf_ascending[int(x)] - cdf_descending[int(99 - x)]
    return abs(difference)


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


def search_index_of_class(class_index, n_times, labels, ini=0):
    '''
  Returns a list of lenght n_times with the indexes of the labels for a class.
  '''
    indexes = []
    n = 0
    i = 0
    while n < n_times:
        if class_index == np.argmax(labels[ini + i]):
            indexes.append(ini + i)
            n = n + 1
        i = i + 1
    return indexes


def creation_of_heatmaps_per_class(number_of_htmaps, instances, labels, model, class_names: list):
    '''
    Creates an array with the number of heatmaps selected per class
    '''
    last_conv_layer = extract_last_convolutional_layer_name(model)
    heatmaps = np.empty((len(class_names), number_of_htmaps, model.get_layer(last_conv_layer).output.shape[1],
                         model.get_layer(last_conv_layer).output.shape[2]))
    for class_index in tqdm(range(len(class_names))):
        # List of index of the labels for the class
        indexes_for_one_class = search_index_of_class(class_index, number_of_htmaps, labels)
        for loop_index, label_index in enumerate(indexes_for_one_class):
            # Fill the array with the heatmaps
            heatmaps[class_index, loop_index] = grad_cam_plus(instances[label_index], model, last_conv_layer,
                                                              category_id=np.argmax(labels[label_index]))
    return heatmaps


def compute_average_heatmaps_per_cluster(cluster_indexes, heatmaps_array, ssim_distance_matrix_one_class, avg_mode,
                                         thr=None):
    '''
    Computes the average per cluster of the provided heatmaps
    ::cluster_indexes: array with the cluster indexes
    ::heatmaps_array: array with the heatmaps. Shape = [number_of_htmaps, height, width]
    ::avg_mode: string that defines how to do the average
    ::thr: if we want to compute the average only on a percentage of the closest ones
            the threshold parameter should be included with the percentage in the range
            of 0 to 1.
    :Returns: array with the average heatmap per cluster
    '''
    # Extract unique numbers and how much of them are
    unique, counts = np.unique(cluster_indexes, return_counts=True)
    # Eliminate index -1 because it refers to outliers
    if -1 in unique:
        unique = np.delete(unique, 0)
        counts = np.delete(counts, 0)
    # Initialize the array where the average heatmaps are going to be stored
    avg_htmaps_per_cluster = np.empty((len(unique), heatmaps_array.shape[1], heatmaps_array.shape[2]))
    for index_unique, cluster in enumerate(unique):
        # Retrieve the indexes of the cluster
        indexes_one_cluster = np.asarray(np.asarray(cluster_indexes == cluster).nonzero()[0])
        # Either retrieve only the closest heatmaps of that cluster OR retrieve all
        if thr is not None and len(indexes_one_cluster) > 39:  # Retrieval of only the closest ones
            # Initialize the array that will contain the mean D_ssim of each heatmap against the heatmaps of the
            # cluster including itself
            mean_dist_ssim_of_htmaps_of_cluster = np.zeros((len(indexes_one_cluster)))
            for loop_index_1, index_htmap_1 in enumerate(indexes_one_cluster):
                # Initialize the array that will be used to collect the ssim values against the others
                # to then compute the mean
                dist_ssim_of_one_htmap_against_all = np.zeros((len(indexes_one_cluster)))
                for loop_index_2, index_htmap_2 in enumerate(indexes_one_cluster):
                    dist_ssim_of_one_htmap_against_all[loop_index_2] = ssim_distance_matrix_one_class[
                        index_htmap_1, index_htmap_2]
                mean_dist_ssim_of_htmaps_of_cluster[loop_index_1] = np.mean(dist_ssim_of_one_htmap_against_all)
            # Sort mean ascending
            indexes_images_sorted_per_dist_ssim = np.argsort(mean_dist_ssim_of_htmaps_of_cluster)
            # Extract the introduced percent more closer of the cluster using thr parameter
            indexes_for_average_computation_percent = indexes_images_sorted_per_dist_ssim[
                                                      :int(thr * len(mean_dist_ssim_of_htmaps_of_cluster))]
            htmaps_of_the_cluster = heatmaps_array[indexes_one_cluster][indexes_for_average_computation_percent]

        else:  # Retrieval of all the heatmaps of the cluster
            indexes_one_cluster = np.sort(indexes_one_cluster)
            htmaps_of_the_cluster = heatmaps_array[indexes_one_cluster]

        # Compute the average depending on the mode selected
        if avg_mode == 'Mean':
            # print(htmaps_of_the_cluster.shape)
            htmap_prom_un_cluster = np.mean(htmaps_of_the_cluster, axis=0)

        elif avg_mode == 'Median':
            htmap_prom_un_cluster = np.median(htmaps_of_the_cluster, axis=0)
        else:
            raise NameError('Non-existant mode introduced')
        # For every cluster we add its average heatmap
        avg_htmaps_per_cluster[index_unique] = htmap_prom_un_cluster

    return avg_htmaps_per_cluster


def extract_last_convolutional_layer_name(model) -> str:
    """ Save the name of the last convolutional layer """
    last_conv_layer = ''
    for layer in model.layers[::-1]:
        if isinstance(layer, (keras.layers.Conv2D, keras.layers.Add)):
            last_conv_layer = layer.name
            break
    if last_conv_layer == '':
        raise ValueError('No last convolutional layer encountered')
    return last_conv_layer


def generate_heatmaps(images, predictions, model):
    '''
    Generates the heatmaps of the images provided using the prediction (the label)
    '''
    last_conv_layer = extract_last_convolutional_layer_name(model)
    htmaps = np.zeros((images.shape[0], model.get_layer(last_conv_layer).output.shape[1],
                       model.get_layer(last_conv_layer).output.shape[2]))
    for i, image in tqdm(enumerate(images)):
        htmaps[i] = grad_cam_plus(image, model, last_conv_layer, category_id=predictions[i])
    return htmaps


def ssim_distance_matrix_creation(heatmaps_array):
    '''
  Creates an upper triangular array with the pairwise distances between each heatmap in the array provided.
  '''
    ssim_mat = np.zeros((len(heatmaps_array), len(heatmaps_array)))
    for i, htmap in enumerate(heatmaps_array):
        # We visit every column which index is greater than the index of the current row.
        # The diagonal is not visited as the distance between two identical images is 0
        # and the array is defined all zeros.
        for j in range(i + 1, len(heatmaps_array)):
            ssim_mat[i, j] = ssim_distance(htmap, heatmaps_array[j])
            ssim_mat[j, i] = ssim_mat[i, j]
    return ssim_mat


def ssim_distance(img1, img2):
    '''
  Computed the Dssim between two images, defined as:
  Dssim = (1 - SSIM)/2
  This way, obtained value is between 0 and 1, 0 being distance between identical images
  '''
    # Data range is from -1 to 1, therefore 2
    return (1 - ssim(img1, img2, data_range=DATA_RANGE)) / 2


def compute_ssim_against_cluster_averages(input_heatmaps, preds, avg_htmaps_per_class_and_cluster, mode='Similarity'):
    '''
      Computes the ssim of the input heatmaps against the cluster averages of the predicted class
      '''
    # Initialize the array containing the SSIM values
    ssim_per_input_heatmap = np.zeros((len(input_heatmaps)))
    for index, predicted_class in tqdm(enumerate(preds)):
        # Initialize the array containing the SSIM values against cluster averages for one input heatmap
        ssim_against_cluster_avgs = np.zeros((avg_htmaps_per_class_and_cluster[predicted_class].shape[0]))
        if mode == 'Similarity':
            for index_avg_htmap, avg_htmap in enumerate(avg_htmaps_per_class_and_cluster[predicted_class]):
                ssim_against_cluster_avgs[index_avg_htmap] = ssim(avg_htmap, input_heatmaps[index],
                                                                  data_range=DATA_RANGE)
        elif mode == 'Distance':
            for index_avg_htmap, avg_htmap in enumerate(avg_htmaps_per_class_and_cluster[predicted_class]):
                ssim_against_cluster_avgs[index_avg_htmap] = ssim_distance(avg_htmap, input_heatmaps[index])
        else:
            raise NameError('Selected mode does not exist')
        # Select the more similar average heatmap (the max SSIM value)
        ssim_per_input_heatmap[index] = np.max(ssim_against_cluster_avgs)
    return ssim_per_input_heatmap


def compute_ssim_against_all_heatmaps_of_closest_cluster(input_heatmaps, preds, avg_htmaps_per_class_and_cluster,
                                                        heatmaps_in_the_clusters_per_class,
                                                        cluster_indexes_of_heatmaps):
    '''
  Computes the ssim of the input heatmaps against all the heatmaps of the closest cluster of the predicted class.
  First computes the SSIM against cluster averages, and it selects the max similarity (max ssim).
  Then computes the SSIM against all the heatmaps belonging to that cluster
  '''
    # Initialize the array that will contain the SSIM values
    ssim_per_input_heatmap = np.zeros((len(input_heatmaps)))
    for index, predicted_class in tqdm(enumerate(preds)):
        # Initialize the array that will contain the SSIM values to the cluster averages.
        ssim_against_cluster_avgs = np.zeros((len(avg_htmaps_per_class_and_cluster[predicted_class])))
        for index_avg_htmap, avg_htmap in enumerate(avg_htmaps_per_class_and_cluster[predicted_class]):
            # Compute distances and we select the closest cluster
            ssim_against_cluster_avgs[index_avg_htmap] = ssim(input_heatmaps[index], avg_htmap, data_range=DATA_RANGE)
            closest_cluster_index = np.argmax(ssim_against_cluster_avgs)  # Max similarity == Closest cluster
        # Indexes of the heatmaps that belong to the nearest cluster.
        indexes_of_closest_cluster_heatmaps = np.asarray(
            np.asarray(cluster_indexes_of_heatmaps[predicted_class] == closest_cluster_index).nonzero()[0])
        # Compute every SSIM value to the heatmaps in the closest cluster
        # Initialze the array that will contain the SSIM values against all heatmaps in closest cluster
        ssim_against_closest_cluster_heatmaps = np.zeros((len(indexes_of_closest_cluster_heatmaps)))
        for i, htmap_indexes in enumerate(indexes_of_closest_cluster_heatmaps):
            ssim_against_closest_cluster_heatmaps[i] = ssim(input_heatmaps[index], heatmaps_in_the_clusters_per_class[
                predicted_class, htmap_indexes], data_range=DATA_RANGE)
        # Select the more similar average heatmap (the max SSIM value)
        ssim_per_input_heatmap[index] = np.max(ssim_against_closest_cluster_heatmaps)
    return ssim_per_input_heatmap


# Similarity
def similarity_thresholds_for_each_TPR(similarity_per_class):
    # Creation of the array with the thresholds for each TPR (class, dist_per_TPR)
    sorted_similarity_per_class = [np.sort(x)[::-1] for x in similarity_per_class]
    tpr_range = np.arange(0, 1, 0.005)
    tpr_range[-1] = 0.99999999  # For selecting the last item correctly
    similarity_thresholds_test = np.zeros((len(similarity_per_class), len(tpr_range)))
    for class_index in range(len(similarity_per_class)):
        for index, tpr in enumerate(tpr_range):
            similarity_thresholds_test[class_index, index] = sorted_similarity_per_class[class_index][
                int(len(sorted_similarity_per_class[class_index]) * tpr)]
    return similarity_thresholds_test


def compare_similarity_to_similarity_thrs(distances_list_per_class, thr_distances_array):
    '''
    Function that creates an array of shape (tpr, InD_or_OD), where tpr has the lenght of the number of steps of the TPR list
    and second dimensions has the total lenght of the distances_list_per_class, and cotains True if its InD and False if is OD
    :distances_list_per_class: list with each element being an array with the distances to avg clusters of one class [array(.), array(.)]
    :thr_distances_array: array of shape (class, dist_for_each_tpr), where first dimension is the class and the second is the distance for the TPR
     corresponding to that position. For example, the TPR = 0.85 corresponds to the 85th position.
    '''
    in_or_out_distribution_per_tpr = np.zeros(
        (len(np.transpose(thr_distances_array)), len(np.concatenate(distances_list_per_class))), dtype=bool)
    for tpr_index, thr_distances_per_class in enumerate(np.transpose(thr_distances_array)):
        in_or_out_distribution_per_tpr[tpr_index] = np.concatenate(
            [dist_one_class > thr_distances_per_class[cls_index] for cls_index, dist_one_class in
             enumerate(distances_list_per_class)])

    return in_or_out_distribution_per_tpr


def compute_precision_tpr_fpr_for_test_and_OoD_similarity(dist_test, dist_OoD, dist_thresholds_test):
    # Creation of the array with True if predicted InD (True) or OD (False)
    in_or_out_distribution_per_tpr_test = compare_similarity_to_similarity_thrs(dist_test, dist_thresholds_test)
    in_or_out_distribution_per_tpr_test[0] = np.zeros((in_or_out_distribution_per_tpr_test.shape[1]),
                                                      dtype=bool)  # To fix that first element is True when TPR is 0
    in_or_out_distribution_per_tpr_test[-1] = np.ones((in_or_out_distribution_per_tpr_test.shape[1]),
                                                      dtype=bool)  # To fix that last element is True when TPR is 1
    in_or_out_distribution_per_tpr_OoD = compare_similarity_to_similarity_thrs(dist_OoD, dist_thresholds_test)

    # Creation of arrays with TP, FN and FP, TN
    tp_fn_test = tp_fn_fp_tn_computation(in_or_out_distribution_per_tpr_test)
    fp_tn_OoD = tp_fn_fp_tn_computation(in_or_out_distribution_per_tpr_OoD)

    # Computing TPR, FPR and Precision
    tpr_values = tp_fn_test[:, 0] / (tp_fn_test[:, 0] + tp_fn_test[:, 1])
    fpr_values = fp_tn_OoD[:, 0] / (fp_tn_OoD[:, 0] + fp_tn_OoD[:, 1])
    precision = tp_fn_test[:, 0] / (tp_fn_test[:, 0] + fp_tn_OoD[:, 0])

    # Eliminating NaN value at TPR = 1
    precision[0] = 1
    return precision, tpr_values, fpr_values


# For both
def tp_fn_fp_tn_computation(in_or_out_distribution_per_tpr):
    '''
    Function that creates an array with the number of values of tp and fp or fn and tn, depending on if the
    passed array is InD or OD.
    :in_or_out_distribution_per_tpr: array with True if predicted InD and False if predicted OD, for each TPR
    ::return: array with shape (tpr, 2) with the 2 dimensions being tp,fn if passed array is InD, and fp and tn if the passed array is OD
    '''
    tp_fn_fp_tn = np.zeros((len(in_or_out_distribution_per_tpr), 2), dtype='uint16')
    length_array = in_or_out_distribution_per_tpr.shape[1]
    for index, element in enumerate(in_or_out_distribution_per_tpr):
        n_True = int(len(element.nonzero()[0]))
        tp_fn_fp_tn[index, 0] = n_True
        tp_fn_fp_tn[index, 1] = length_array - n_True
    return tp_fn_fp_tn


# Next three functions used to download from Gdrive
def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)
    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    save_response_content(response, destination)
    return destination


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def unzip_file(zip_file_path: str, dest_dir: str):
    '''
    Function that extracts the zip file and deletes it, returning the new file path
    :param zip_file_path: str with the desired name for the zip file downloaded
    :param dest_dir:
    '''
    # Create a ZipFile Object and load sample.zip in it
    with ZipFile(zip_file_path, 'r') as zipObj:
        # Extract all the contents of zip file in current directory
        zipObj.extractall(dest_dir)
    os.remove(zip_file_path)
    return zip_file_path[:-4]  # The path of the new dir created


def load_svhn(image_dir, image_file):
    '''

    :param image_dir:
    :param image_file:
    :return:
    '''
    print('Loading SVHN dataset.')
    image_dir = os.path.join(image_dir, image_file)
    svhn = loadmat(image_dir)
    images = np.transpose(svhn['X'], [3, 0, 1, 2]).astype('float32') / 255
    labels = svhn['y']
    labels[np.where(labels == 10)] = 0
    labels = to_categorical(labels)
    return images, labels


def download_or_load_dataset(dataset_name: str, only_test_images=False):
    '''
    Downloads the dataset, or loads it if already exist locally
    :param dataset_name: str containing the name of the dataset
    :param return_train: if True, returns the train split of the dataset
    :return: if return_train=True, then (train_images, train_labels), (test_images, test_labels), class_names,
    num_classes; else (test_images, test_labels)
    '''
    print('Only SVHN_Cropped is downloaded directly to the datasets folder, the other datasets are stored'
          'locally in ~/.keras/datasets')
    train_images, train_labels, class_names = None, None, None
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
        # Create datasets dir if it does not exist
        create_dir(DATASET_DIR)
        # Download SVHN
        svhn_zip_path = os.path.join(DATASET_DIR, SVHN_ZIP_FILE_NAME)
        svhn_dir_path = unzip_file(download_file_from_google_drive(SVHN_ID, svhn_zip_path), dest_dir=DATASET_DIR)
        # Load SVHN
        train_images, train_labels = load_svhn(svhn_dir_path, 'train_32x32.mat')
        test_images, test_labels = load_svhn(svhn_dir_path, 'test_32x32.mat')
        # Definition of the constants of the dataset
        class_names = list(np.linspace(0, 9, 10).astype('int'))
        class_names = [str(i) for i in class_names]
    elif dataset_name == 'MNIST_Color':
        (_, _), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
        test_images = test_images / 255
        test_images = test_images.reshape(10000, 28, 28, 1)
        test_images = test_images.astype('float32')
        test_images = np.tile(test_images, 3)
        test_images = resize(test_images, (10000, 32, 32, 3))
    elif dataset_name == 'Fashion_MNIST_Color':
        (_, _), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
        test_images = test_images / 255
        test_images = test_images.reshape(10000, 28, 28, 1)
        test_images = test_images.astype('float32')
        test_images = np.tile(test_images, 3)
        test_images = resize(test_images, (10000, 32, 32, 3))
    elif dataset_name == 'Cifar10_Gray':
        cifar = tf.keras.datasets.cifar10
        (_, _), (test_images, test_labels) = cifar.load_data()
        test_images = test_images.reshape(10000, 32, 32, 3)
        test_images = test_images.astype('float32') / 255
        test_images = np.expand_dims(color.rgb2gray(test_images), axis=3)
        test_images = resize(test_images, (10000, 28, 28, 1))
    else:
        raise NameError('Dataset name not found in the dataset options')
    if only_test_images is False:
        if (train_images, train_labels, class_names) is (None, None, None):
            raise Exception(f'Incompatible choices {dataset_name} with "only_test_images=False". If you want to use'
                            f' this dataset to train a model, please, include the train_images, train_labels and'
                            f'class_names attributes in the definition of the dataset in the function'
                            f'download_or_load_dataset in utils.py')
        else:
            return (train_images, train_labels), (test_images, test_labels), class_names
    elif only_test_images is True:
        return test_images
    else:
        raise NameError('Wrong option for the return_train parameter')


def load_model_weights(model: keras.Model, dataset_name: str, model_name: str):
    '''
    Loads the weights for the specified model
    :return:
    '''
    download_models_weights(PRETRAINED_WEIGHTS_DIR)
    pretrained_weights_path = os.path.join(PRETRAINED_WEIGHTS_DIR, f'{dataset_name}_{model_name}.h5')
    model.load_weights(pretrained_weights_path)


def download_models_weights(weights_dir_path):
    '''
    Downloads the pretrained weights for the models
    :return:
    '''
    weights_zip_path = os.path.join(weights_dir_path, PRETRAINED_WEIGHTS_ZIP_FILE_NAME)
    try:
        if os.path.isdir(weights_dir_path):
            print(f'{weights_dir_path} directory already exist, delete if to automatically download again the files')
        else:
            os.mkdir(weights_dir_path)
            print(f'Folder "{weights_dir_path}" created')
            weights_dir_path = unzip_file(download_file_from_google_drive(PRETRAINED_WEIGHTS_ID,
                                                                          weights_zip_path), dest_dir=weights_dir_path)
            print(f'Successfully downloaded the {weights_dir_path} directory')
    except Exception as e:
        print(f'Exception {e.__class__} occurred while creating downloading')


def load_test_sample_of_dataset(dataset_name):
    # Load the selected dataset
    if dataset_name == 'SVHN_cropped':
        # Download SVHN
        SVHN_FOLDER_PATH = unzip_file(download_SVHN())
        images, labels = load_svhn(SVHN_FOLDER_PATH, "test_32x32.mat")
        np.random.shuffle(images)
        images = images[:10000]
    elif dataset_name == 'MNIST':
        (_, _), (images, labels) = tf.keras.datasets.mnist.load_data()
        images = images / 255
        images = images.reshape(10000, 28, 28, 1)
        images = images.astype('float32')
        labels = to_categorical(labels)
    elif dataset_name == 'Fashion_MNIST':
        (_, _), (images, labels) = tf.keras.datasets.fashion_mnist.load_data()
        images = images / 255
        images = images.reshape(10000, 28, 28, 1)
        images = images.astype('float32')
    elif dataset_name == 'MNIST_color':
        (_, _), (images, labels) = tf.keras.datasets.mnist.load_data()
        images = images / 255
        images = images.reshape(10000, 28, 28, 1)
        images = images.astype('float32')
        images = np.tile(images, 3)
        images = resize(images, (10000, 32, 32, 3))
    elif dataset_name == 'Fashion_MNIST_color':
        (_, _), (images, labels) = tf.keras.datasets.fashion_mnist.load_data()
        images = images / 255
        images = images.reshape(10000, 28, 28, 1)
        images = images.astype('float32')
        images = np.tile(images, 3)
        images = resize(images, (10000, 32, 32, 3))
    elif dataset_name == 'Cifar10_grey':
        cifar = tf.keras.datasets.cifar10
        (_, _), (images, labels) = cifar.load_data()
        # Damos el formato correspondiente a las imagenes
        images = images.reshape(10000, 32, 32, 3)
        images = images.astype('float32') / 255
        images = np.expand_dims(color.rgb2gray(images), axis=3)
        images = resize(images, (10000, 28, 28, 1))
    elif dataset_name == 'Cifar10':
        cifar = tf.keras.datasets.cifar10
        (_, _), (images, labels) = cifar.load_data()
        # Damos el formato correspondiente a las imagenes
        images = images.reshape(10000, 32, 32, 3)
        images = images.astype('float32') / 255
    else:
        raise NameError()
    return images, labels


# Rotate images (17s 60.000 images)
def rotate_images(images, angle):
    for i, img in enumerate(images):
        images[i] = rotate(img, angle)
    return images


# Translation
def translate_images(images, h_trans, v_trans):
    translation = (h_trans, v_trans)
    tform = trfm.SimilarityTransform(translation=translation)
    for i, img in enumerate(images):
        images[i] = trfm.warp(img, tform)
    return images
