
# Utils
import warnings
import argparse
# Import created functions
from utils.utils import *
from utils.models import create_model
from utils.train_detector import train_od_detector
from utils.constants import *
from utils.test_detector import test_od_detector, create_or_load_average_heatmaps
from utils.plots import plot_average_heatmaps
from time import sleep


def check_arguments(args):

    if args['run_all'] is True:
        args = {
            'ind': ['MNIST', 'Fashion_MNIST', 'Cifar10'],
            'ood': ['MNIST', 'Fashion_MNIST', 'Cifar10', 'SVHN_Cropped'],
            'model_arch': ['LeNet', 'ResNet32'],
            'load_or_train': 'Load',
            'agg_function': ['Mean', 'Median', 10],
            'ood_function': ['f_1', 'f_2'],
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
        # 1: Aggregation function assertion
        for index, f_agg in enumerate(args['agg_function']):
            try:
                f_agg = int(f_agg)
                args['agg_function'][index] = f_agg
            except ValueError:
                pass  # If the argument is not convertible, it is a normal string
            if isinstance(f_agg, str):
                if f_agg not in ["Mean", "Median"]:
                    raise ValueError(
                        'The average mode must be either "Mean", "Median" or a positive integer between '
                        '10 and 100')
            elif isinstance(f_agg, int):
                if not 10 <= f_agg <= 100:
                    raise ValueError(
                        'The average mode must be either "Mean", "Median" or a positive integer between '
                        '10 and 100')
                elif 70 <= f_agg:
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
    parser.add_argument('-f_agg', '--agg_function', nargs='+', help='average modes to be computed: '
                        'Possible choices are Mean, Median or an integer representing the percentage')
    parser.add_argument('-f_ood', '--ood_function', type=str, choices=['f_1', 'f_2'], nargs='+',
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
                for agg_function in args['agg_function']:
                    print('-' * 50)
                    print(f'------ The average mode being used is {agg_function}')
                    print('-' * 50)

                    # If the agg_function is a percentage, we must convert it to it
                    if isinstance(agg_function, int):
                        fig_name = f'percent{agg_function}_average_heatmaps_per_class_and_cluster_{in_dataset}' \
                                   f'_{model_arch}_{args["load_or_train"]}_seed{args["seed"]}'
                        agg_function = agg_function * 0.01
                    else:
                        fig_name = f'{agg_function}_average_heatmaps_per_class_and_cluster_{in_dataset}' \
                                   f'_{model_arch}_{args["load_or_train"]}_seed{args["seed"]}'

                    # Compute or load the average
                    average_heatmaps_per_class_and_cluster = create_or_load_average_heatmaps(
                        in_dataset,
                        model_arch,
                        args,
                        agg_function
                        )

                    # Create the plot for the average heatmaps
                    plot_average_heatmaps(average_heatmaps_per_class_and_cluster, class_names, fig_name,
                                          superimposed=False)
                    for f_ood in args['ood_function']:
                        print('-' * 50)
                        print(f'-------- The comparison function is {f_ood}')
                        print('-' * 50)
                        if isinstance(agg_function, float) and f_ood == 'f_2':
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
                                             agg_function,
                                             f_ood,
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
    print("Thanks for using the utils, if there is any doubt don't hesitate contacting the owner of the repository")
    print('')
    sleep(1)
    print('-'*102)
    sleep(3)


if __name__ == '__main__':
    # Run main utils
    main()
