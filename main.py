
# Utils
import warnings
import argparse
# Import created functions
from utils import *
from models import create_model
from train_detector import train_od_detector
from constants import *
from test_detector import test_od_detector, create_or_load_average_heatmaps
from plots import plot_average_heatmaps
from time import sleep


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


def check_arguments(args):

    if args['run_all'] is True:
        args = {
            'ind': ['MNIST', 'Fashion_MNIST', 'Cifar10'],
            'ood': ['MNIST', 'Fashion_MNIST', 'Cifar10', 'SVHN_Cropped'],
            'model_arch': ['LeNet', 'ResNet32'],
            'load_or_train': 'Load',
            'average_mode': ['Mean', 'Median', 10],
            'comparison_function': ['g_r', 'g_all'],
            'seed': 8,
            'n_heatmaps': 1000
        }
    else:
        # Check if some argument is missing
        keys_missing = []
        for key, value in args.items():
            if key != 'run_all':
                if bool(value) is False:
                    keys_missing.append(f'The argument "{key}" is missing')
            if not keys_missing == []:
                raise KeyError(str(keys_missing))

        # If all parameters have been introduced, we must check if they are all correct
        # 1: Average mode assertion
        for index, avg_mode in enumerate(args['average_mode']):
            try:
                avg_mode = int(avg_mode)
                args['average_mode'][index] = avg_mode
            except ValueError:
                pass  # If the argument is not convertible, it is a normal string
            if isinstance(avg_mode, str):
                if avg_mode not in ["Mean", "Median"]:
                    raise ValueError(
                        'The average mode must be either "Mean", "Median" or a positive integer between '
                        '10 and 100')
            elif isinstance(avg_mode, int):
                if not 10 <= avg_mode <= 100:
                    raise ValueError(
                        'The average mode must be either "Mean", "Median" or a positive integer between '
                        '10 and 100')
                elif 70 <= avg_mode:
                    warnings.warn(
                        'The percentage selected is above 70, take into account that this will give results '
                        'similar to the "Mean" average mode')
            else:
                print('None of the options')

        # 3: Limit the number of heatmaps
        if not MIN_N_HEATMAPS <= args['n_heatmaps'] <= MAX_N_HEATMAPS:
            raise ValueError(f'The min and max values for the "n_heatmaps" parameter are {MIN_N_HEATMAPS} and'
                             f'{MAX_N_HEATMAPS}. If you wish to execute the program not attending to those limits,go '
                             f'to the constants.py file and change the values of "MIN_N_HEATMAPS" and "MAX_N_HEATMAPS".'
                             f'Please be aware that the performance of the detector could change dramatically if the'
                             f'number of heatmaps is too low and that the memory in disk required for computing above'
                             f'the max limit may be too high.')

    return args

def main():
    # Parse the arguments of the call
    parser = argparse.ArgumentParser(description='Script that trains the detector on a specific ')
    parser.add_argument('-run_all', help='If used, it runs all the test of the paper', action='store_true')
    parser.add_argument('-ind', type=str, help='in distribution dataset', nargs='+', choices=IN_D_CHOICES)
    parser.add_argument('-ood', type=str, help='out of distribution dataset', nargs='+', choices=OUT_D_CHOICES)
    parser.add_argument('-m', '--model_arch', type=str, choices=['LeNet', 'ResNet32'], nargs='+',
                        help='model architecture, only one a each call')
    parser.add_argument('-load_or_train', type=str, choices=['Load', 'Train'], help='model architecture')
    parser.add_argument('-avg', '--average_mode', nargs='+', help='average modes to be computed: '
                        'Possible choices are Mean, Median or an integer representing the percentage')
    parser.add_argument('-comp_f', '--comparison_function', type=str, choices=['g_r', 'g_all'], nargs='+',
                        help='comparison functions to be computed')
    parser.add_argument('-s', '--seed', type=int, help='Seed for shuffling the train images and labels', default=8)
    parser.add_argument('-n_heatmaps', type=int, help='Select the number of heatmaps per class for the clustering',
                        default=1000)
    args = vars(parser.parse_args())
    args = vars(parser.parse_args())
    # Check if arguments are correct
    print(args)
    args = check_arguments(args)

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

    # Classify datasets taking into account the format of data
    mnist_like_ind_datasets = [dataset for dataset in args['ind'] if dataset in MNIST_LIKE_DATASETS]
    cifar10_like_ind_datasets = [dataset for dataset in args['ind'] if dataset in CIFAR10_LIKE_DATASETS]
    mnist_like_ood_datasets = [dataset for dataset in args['ood'] if dataset in MNIST_LIKE_DATASETS]
    cifar10_like_ood_datasets = [dataset for dataset in args['ood'] if dataset in CIFAR10_LIKE_DATASETS]

    # Initiate the main loop
    for model_arch in args["model_arch"]:
        print('-' * 50)
        print('')
        print(f'All the datasets compatible with the architecture')
        print(f'"{model_arch}" will be tested')
        print('')
        print('-' * 50)
        # Compute test only on in distributions compatible with the model architecture
        if model_arch in MODELS_FOR_MNIST:
            in_datasets = mnist_like_ind_datasets
        elif model_arch in MODELS_FOR_CIFAR10:
            in_datasets = cifar10_like_ind_datasets
        else:
            raise NameError(f'Model {model_arch} does not exist, please include it in the constants.py file')

        # If there is no dataset compatible with that model architecture, raise a warning and go to next iteration
        if in_datasets == []:
            if model_arch == 'LeNet':
                warnings.warn(f'No in distribution dataset is compatible with {model_arch} architecture, be aware that '
                              f'the following in distribution datasets '
                              f'are not going to be computed: {cifar10_like_ind_datasets}')
            elif model_arch == 'ResNet32':
                warnings.warn(f'No in distribution dataset is compatible with {model_arch} architecture, be aware that '
                              f'the following in distribution datasets '
                              f'are not going to be computed: {mnist_like_ind_datasets}')
            else:
                warnings.warn(f'No in distribution dataset is compatible with {model_arch} architecture, the program'
                              f'will continue if more model architectures have been passed as argument, otherwise'
                              f'program will finish the execution')
            continue
        else:
            print(f'Following in distribution datasets are going to be simulated for the model {model_arch}:', end=' ')
            for in_dataset in in_datasets:
                print(in_dataset, end=', ')
            print('')

        # For every In-Distribution dataset, it has to run all the tests
        for in_dataset in in_datasets:
            print('-' * 50)
            print(f'-- Executing tests for the In-Distribution dataset {in_dataset}')
            print('-' * 50)

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
            test_predictions = np.argmax(model.predict(test_images), axis=1)
            file_name_heatmaps_test = f'heatmaps_ood_{in_dataset}_{model_arch}' \
                                      f'_{args["load_or_train"]}_seed{args["seed"]}.npy'
            path_heatmaps_test = os.path.join(OBJECTS_DIR_NAME, file_name_heatmaps_test)
            if os.path.isfile(path_heatmaps_test):
                test_heatmaps = np.load(path_heatmaps_test)
                print(f'Test heatmaps of {in_dataset} exist, they have been loaded from file!')
            else:
                print('Heatmap generation:')
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
                print('-' * 50)
                print(f'---- OoD Dataset being tested is {ood_dataset}')
                print('-' * 50)

                # If in_dataset and ood_dataset are the same, go to next iteration
                if in_dataset == ood_dataset:
                    continue

                # Generate the OoD heatmaps of the ood_dataset
                ood_images = download_or_load_dataset(ood_dataset, only_test_images=True)
                ood_predictions = np.argmax(model.predict(ood_images), axis=1)
                file_name_heatmaps_ood = f'heatmaps_ood_{ood_dataset}_{model_arch}' \
                                         f'_{args["load_or_train"]}_seed{args["seed"]}.npy'
                path_heatmaps_ood = os.path.join(OBJECTS_DIR_NAME, file_name_heatmaps_ood)
                if os.path.isfile(path_heatmaps_ood):
                    ood_heatmaps = np.load(path_heatmaps_ood)
                    print(f'OoD heatmaps of {ood_dataset} exist, they have been loaded from file!')
                else:
                    print('Heatmap generation:')
                    ood_heatmaps = generate_heatmaps(ood_images, ood_predictions, model)
                    np.save(path_heatmaps_ood, ood_heatmaps, allow_pickle=False)

                # Compute all the approaches for every combination of in and out distribution dataset
                for average_mode in args['average_mode']:
                    print('-' * 50)
                    print(f'------ The average mode being used is {average_mode}')
                    print('-' * 50)

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
                        print('-' * 50)
                        print(f'-------- The comparison function is {comp_funct}')
                        print('-' * 50)
                        if isinstance(average_mode, float) and comp_funct == 'g_all':
                            pass
                        else:
                            test_od_detector(average_heatmaps_per_class_and_cluster,
                                             in_dataset,
                                             ood_dataset,
                                             test_heatmaps,
                                             test_predictions,
                                             ood_heatmaps,
                                             ood_predictions,
                                             model,
                                             model_arch,
                                             average_mode,
                                             comp_funct,
                                             class_names,
                                             args
                                             )
    print('')
    print('-'*102)
    sleep(1)
    print('')
    print(' '*45+'Program finished!')
    sleep(2)
    print('')
    sleep(0.5)
    print("Thanks for using the code, if there is any doubt don't hesitate contacting the owner of the repository")
    print('')
    sleep(1)
    print('-'*102)
    sleep(3)


if __name__ == '__main__':
    # Run main code
    main()
