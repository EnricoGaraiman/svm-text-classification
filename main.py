import os

import src.dataset as dataset
import src.utils as utils
import src.train as train
import argparse

"""
   MAIN FUNCTION
"""
if __name__ == '__main__':
    """
       ARGUMENTS
    """
    parser = argparse.ArgumentParser(description='SVM Text Classification')

    parser.add_argument('--dataset', type=str, help='Dataset (reuters, bbc, ag_news)')
    parser.add_argument('--dataset_path', type=str, help='Dataset path (bbc required)')
    parser.add_argument('--download_dataset', type=bool, help='Dataset will be downloaded (default False)')
    parser.add_argument('--test_split', type=float, help='Dataset test split (bbc required | default 0.2)')
    parser.add_argument('--vectorizer', type=str, help='Vectorizer type (default tfidf)')
    parser.add_argument('--kernel', type=str, help='SVM kernel (default rbf)')
    parser.add_argument('--gamma', type=str, help='SVM gamma (default auto)')
    parser.add_argument('--c', type=float, help='SVM C (default 1)')
    parser.add_argument('--features', type=int, help='Vectorizer features (default 3000)')

    args = parser.parse_args()

    ARGUMENTS = {
        'dataset': args.dataset if args.dataset is not None else '',
        'dataset_path': args.dataset_path if args.dataset_path is not None else '',
        'download_dataset': args.download_dataset if args.download_dataset is not None else False,
        'test_split': args.test_split if args.test_split is not None else 0.2,
        'vectorizer': args.vectorizer if args.vectorizer is not None else 'tfidf',
        'kernel': args.kernel if args.kernel is not None else 'linear',
        'gamma': args.gamma if args.gamma is not None else 'auto',
        'c': args.c if args.c is not None else 1,
        'features': args.features if args.features is not None else 3000
    }

    dir_name = utils.get_next_run_director_name()
    if not os.path.isdir('runs/' + dir_name):
        os.mkdir('runs/' + dir_name)

    """
       SVM
    """
    tuned_parameters = {
        'kernel': ARGUMENTS['kernel'],
        'gamma': ARGUMENTS['gamma'],
        'C': ARGUMENTS['c']
    }

    [texts_train, texts_test, labels_train, labels_test, categories] = dataset.prepare_dataset(ARGUMENTS)

    model = train.training_stage(ARGUMENTS, tuned_parameters, texts_train, labels_train)
    train.testing_stage(ARGUMENTS, model, texts_test, labels_test, categories)
    utils.save_parameters(ARGUMENTS)