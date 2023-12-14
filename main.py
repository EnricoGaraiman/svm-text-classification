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

    parser.add_argument('--dataset_path', type=str, help='Dataset path')
    parser.add_argument('--test_split', type=float, help='Dataset test split (default 0.2)')
    parser.add_argument('--k_grid', type=int, help='K Cross validation (default 5)')

    args = parser.parse_args()

    ARGUMENTS = {
        'dataset_path': args.dataset_path if args.dataset_path is not None else '',
        'test_split': args.test_split if args.test_split is not None else 0.2,
        'k_grid': args.k_grid if args.k_grid is not None else 10,
    }

    utils.save_parameters(ARGUMENTS)

    """
       SVM
    """
    tuned_parameters = [
        {
            'kernel': ['rbf'],
            'gamma': [1e-3, 1e-4, 1e-5],
            'C': [1, 10, 100]
        },
        {
            'kernel': ['linear'],
            'C': [1, 10, 100]
        }
    ]

    [texts_train, texts_test, labels_train, labels_test, categories] = dataset.prepare_dataset(ARGUMENTS)

    clf = train.training_stage(ARGUMENTS, tuned_parameters, texts_train, labels_train)
    train.testing_stage(clf, texts_test, labels_test)
