import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import src.utils as utils


def prepare_dataset(ARGUMENTS):
    """
    Prepare dataset

    :param ARGUMENTS: arguments dict   :type ARGUMENTS: dict
    :return: train & test data and labels   :rtype: list
    """
    df = pd.read_csv(ARGUMENTS['dataset_path'])
    # print(df.head())

    texts = df[['text']].to_numpy().reshape(-1)
    labels = df[['category']].to_numpy().reshape(-1)

    n_texts = len(texts)
    print('Texts in dataset: %d' % n_texts)

    categories = utils.get_categories(df)
    print('Number of categories: %d' % len(categories))

    texts_train, texts_test, labels_train, labels_test = train_test_split(texts, labels,
                                                                          test_size=ARGUMENTS['test_split'])

    # vectorizer to convert the text into numerical features
    vectorizer = CountVectorizer(strip_accents=None, lowercase=False)
    texts_train = vectorizer.fit_transform(texts_train)
    texts_test = vectorizer.transform(texts_test)

    plot_dataset_distribution(df['category'])

    return [texts_train, texts_test, labels_train, labels_test, categories]


def plot_dataset_distribution(categories):
    """
    Plot & save dataset distribution

    :param categories: classes    :type categories: list
    """
    counter = collections.Counter(categories)

    distribution = []
    data = []

    for key, count in sorted(counter.items()):
        data.append(key)
        distribution.append(count)

    distribution = distribution / np.sum(distribution) * 100

    fig = plt.figure(figsize=(10, 5), dpi=300)
    plt.bar(data, distribution, color='royalblue', width=0.5)
    plt.xlabel("Class", fontsize=16)
    plt.ylabel("Texts [%]", fontsize=16)
    plt.title('Dataset classes distribution', fontsize=20)
    ax = plt.gca()
    ax.tick_params(axis='both', labelsize=14)
    plt.tight_layout()
    utils.add_labels(data, distribution)
    plt.savefig('results/dataset_class_distribution.jpg', dpi=fig.dpi)
