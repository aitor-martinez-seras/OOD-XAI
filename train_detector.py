import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import tensorflow as tf
#from tensorflow import keras
#from tensorflow.keras import layers
#from tensorflow.keras.utils import to_categorical
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
#from utils import *


def main():
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
    main()
