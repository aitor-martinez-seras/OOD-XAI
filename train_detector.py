import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import tensorflow as tf
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
#from utils import *


def train_od_detector(train_images_and_labels:tuple, model:keras.Model):
    '''
    Computes the training of the Out-of-Distribution detector by creating a clusterized space by using SSIM Distance
    :param dataset_name: str of the In-Distribution dataset name
    :param model: model instance already trained
    :return: saves the arrays in the objects directory
    '''
    train_images, train_labels = train_images_and_labels

