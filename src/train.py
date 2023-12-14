import pickle
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import time
import pandas as pd
import src.utils as utils
import seaborn as sns


def training_stage(PARAMETERS, tuned_parameters, texts_train, labels_train):
    """
    SVM Training Stage
    :param PARAMETERS: arguments dict   :type PARAMETERS: dict
    :param tuned_parameters: parameters for cross validation   :type tuned_parameters: dict
    :param texts_train: data for train   :type texts_train: list
    :param labels_train: labels for train   :type labels_train: list
    :return: SVM model    :rtype:  SVM object
    """
    # K Cross validation
    A = time.time()
    clf = GridSearchCV(
        SVC(),
        tuned_parameters,
        cv=PARAMETERS['k_grid'],
        scoring='precision_weighted',
        verbose=2,
        n_jobs=-1
    )
    clf.fit(texts_train, labels_train)

    print('Cross validation time: {}'.format(time.time() - A))
    print("Best parameters set found on development set:")
    print(clf.best_params_)

    # save results for K Cross validation
    cv_result = pd.DataFrame.from_dict(clf.cv_results_)
    with open('runs/' + utils.get_next_run_director_name() + '/cross_validation_result.csv', 'w') as f:
        cv_result.to_csv(f)

    return clf


def testing_stage(model, text_test, labels_test):
    """
    SVM Testing Stage

    :param model: SVM model   :type model: SVM object
    :param text_test: data for test   :type text_test: list
    :param labels_test: labels for test   :type labels_test: list
    """
    # calculate accuracy
    labels_pred = model.predict(text_test)
    accuracy = get_predict_accuracy(labels_pred=labels_pred, labels_test=labels_test)
    print("\nAccuracy: {}".format(round(accuracy * 100, 2)))

    print('\nClassification report:')
    print(classification_report(labels_test, labels_pred))

    # save model
    save_model(model)

    # save confusion matrix
    plot_confusion_matrix(text_test, labels_test, model)


def save_model(model):
    """
    Save SVM model

    :param model: SVM model   :type model: SVM object
    """
    with open('runs/' + utils.get_next_run_director_name() + '/svm.pickle', 'wb') as fp:
        pickle.dump(model, fp, protocol=pickle.HIGHEST_PROTOCOL)


def get_predict_accuracy(labels_pred, labels_test):
    """
    Calculate prediction accuracy

    :param labels_pred: predicted labels   :type labels_pred: list
    :param labels_test: labels for test   :type labels_test: list
    :return: accuracy   :rtype: float
    """
    sum_acc = 0

    for j, pre in enumerate(labels_pred):
        if pre in labels_test[j]:
            sum_acc += 1

    return sum_acc / len(labels_pred)


def plot_confusion_matrix(text_test, labels_test, model):
    """
    Plot & save confusion matrix

    :param text_test: data for test   :type text_test: list
    :param labels_test: labels for test   :type labels_test: list
    :param model: SVM model   :type model: SVM object
    """
    Y_pred = model.predict(text_test)

    con_mat = confusion_matrix(labels_test, Y_pred)
    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)

    label_names = list(range(len(con_mat_norm)))
    con_mat_df = pd.DataFrame(con_mat_norm,
                              index=label_names,
                              columns=label_names)

    fig = plt.figure(figsize=(10, 10), dpi=300)
    sns.heatmap(con_mat_df, cmap=plt.cm.Blues, annot=True)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('runs/' + utils.get_next_run_director_name() + '/confusion-matrix.jpg', dpi=fig.dpi)
