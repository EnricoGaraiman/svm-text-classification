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
    parser.add_argument('--k_grid', type=int, help='K Cross validation (default 5)')
    parser.add_argument('--vectorizer', type=str, help='Vectorizer type (default tfidf)')

    args = parser.parse_args()

    ARGUMENTS = {
        'dataset': args.dataset if args.dataset is not None else '',
        'dataset_path': args.dataset_path if args.dataset_path is not None else '',
        'download_dataset': args.download_dataset if args.download_dataset is not None else False,
        'test_split': args.test_split if args.test_split is not None else 0.2,
        'k_grid': args.k_grid if args.k_grid is not None else 10,
        'vectorizer': args.vectorizer if args.vectorizer is not None else 'tfidf',
    }

    utils.save_parameters(ARGUMENTS)

    """
       SVM
    """
    concat = 'estimator__' if ARGUMENTS['dataset'] == 'reuters' else ''
    tuned_parameters = [
        # {
        #     concat + 'kernel': ['poly'],
        #     concat + 'degree': [2, 3, 4],
        #     concat + 'C': [0.1, 1, 2, 3, 5, 10, 50]
        # },
        # {
        #     concat + 'kernel': ['rbf', 'poly', 'sigmoid'],
        #     concat + 'gamma': ['scale', 'auto', 1e-2, 1e-3, 1e-4],
        #     concat + 'C': [0.1, 1, 2, 3, 5, 10, 50]
        # },
        # {
        #     concat + 'kernel': ['linear'],
        #     concat + 'C': [0.1, 1, 2, 3, 5, 10, 50]
        # }
        {
            concat + 'kernel': ['rbf'],
            concat + 'gamma': ['auto'],
            concat + 'C': [0.1, 1, 2, 10]
        },
    ]

    [texts_train, texts_test, labels_train, labels_test, categories] = dataset.prepare_dataset(ARGUMENTS)

    clf = train.training_stage(ARGUMENTS, tuned_parameters, texts_train, labels_train)
    train.testing_stage(ARGUMENTS, clf, texts_test, labels_test)
