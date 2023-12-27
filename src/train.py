import pickle
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import time
import src.utils as utils

np.random.seed(101)

def training_stage(ARGUMENTS, tuned_parameters, texts_train, labels_train):
    """
    SVM Training Stage
    :param ARGUMENTS: arguments dict   :type ARGUMENTS: dict
    :param tuned_parameters: parameters for cross validation   :type tuned_parameters: dict
    :param texts_train: data for train   :type texts_train: list
    :param labels_train: labels for train   :type labels_train: list
    :return: SVM model    :rtype:  SVM object
    """
    clf = SVC(
        C=tuned_parameters['C'],
        kernel=tuned_parameters['kernel'],
        gamma=tuned_parameters['gamma'],
        tol=1e-5,
        random_state=101,
    )

    if ARGUMENTS['dataset'] == 'reuters':
        model = OneVsRestClassifier(clf)
    else:
        model = clf

    start_time = time.time()
    model.fit(texts_train, labels_train)
    print('Training time: {}'.format(time.time() - start_time))

    return model


def testing_stage(ARGUMENTS, model, text_test, labels_test, categories):
    """
    SVM Testing Stage

    :param ARGUMENTS: arguments dict   :type ARGUMENTS: dict
    :param model: SVM model   :type model: SVM object
    :param text_test: data for test   :type text_test: list
    :param labels_test: labels for test   :type labels_test: list
    :param categories: categories   :type categories: list
    :return labels_pred: labels from prediction   :type labels_test: list
    """
    # calculate accuracy
    start_time = time.time()
    labels_pred = model.predict(text_test)
    print('Testing time: {}'.format(time.time() - start_time))

    print('\nClassification report:')
    print(classification_report(labels_test, labels_pred, digits=4))

    # save model
    save_model(model)

    # save confusion matrix
    if ARGUMENTS['dataset'] != 'reuters':
        plot_confusion_matrix(text_test, labels_test, model, categories)

    return labels_pred


def save_model(model):
    """
    Save SVM model

    :param model: SVM model   :type model: SVM object
    """
    with open('runs/' + utils.get_next_run_director_name() + '/svm.pickle', 'wb') as fp:
        pickle.dump(model, fp, protocol=pickle.HIGHEST_PROTOCOL)


def plot_confusion_matrix(text_test, labels_test, model, categories):
    """
    Plot & save confusion matrix

    :param text_test: data for test   :type text_test: list
    :param labels_test: labels for test   :type labels_test: list
    :param categories: categories   :type categories: list
    :param model: SVM model   :type model: SVM object
    """
    labels_pred = model.predict(text_test)
    labels = list(range(0, len(categories)))

    label_encoder = LabelEncoder()
    labels_test = label_encoder.fit_transform(labels_test)
    labels_pred = label_encoder.transform(labels_pred)

    cmx = confusion_matrix(labels_test, labels_pred, labels=labels, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cmx, display_labels=categories)

    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
    fig.suptitle('Confusion Matrix', fontsize=20)
    plt.xlabel('True label', fontsize=16)
    plt.ylabel('Predicted label', fontsize=16)
    disp.plot(ax=ax, xticks_rotation=0, colorbar=True)
    plt.savefig('runs/' + utils.get_next_run_director_name() + '/confusion-matrix.jpg', dpi=fig.dpi)
