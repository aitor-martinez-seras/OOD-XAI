# Constants definition
DATA_RANGE = 2 # From -1 to 1
DATASET_DIR = 'datasets/'
SVHN_ID = '1Qezu-SHyjBF_fGwdFYUSioVbAu3GMfBj'
SVHN_ZIP_FILE_NAME = 'SVHN_Cropped.zip'
PRETRAINED_WEIGHTS_DIR = 'pretrained_weights/'
PRETRAINED_WEIGHTS_ID = '1DZXCm839Ht0Xcl4w-ynUepjRoxRp7xsr'
PRETRAINED_WEIGHTS_ZIP_FILE_NAME = 'pretrained_weights.zip'
OBJECTS_DIR_NAME = 'objects/'
RESULTS_DIR_NAME = 'results/'
FIGURES_DIR_NAME = 'figures/'
CSV_SEPARATOR = ';'
CSV_DECIMAL = ','
ALL_CHOICES = ['MNIST', 'Fashion_MNIST', 'Cifar10_Gray', 'SVHN_Gray',
               'Cifar10', 'SVHN_Cropped','MNIST_Color', 'Fashion_MNIST_Color']
IN_D_CHOICES = ['MNIST', 'Fashion_MNIST', 'Cifar10']
OUT_D_CHOICES = ALL_CHOICES
MNIST_LIKE_DATASETS = ['MNIST', 'Fashion_MNIST', 'Cifar10_Gray', 'SVHN_Gray']
CIFAR10_LIKE_DATASETS = ['MNIST_Color', 'Fashion_MNIST_Color', 'Cifar10', 'SVHN_Cropped']
MODELS_FOR_MNIST = ['LeNet']
MODELS_FOR_CIFAR10 = ['ResNet32']