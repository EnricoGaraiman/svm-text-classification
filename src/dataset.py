import collections
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import src.utils as utils
from nltk.corpus import reuters
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer


def prepare_dataset(ARGUMENTS):
    """
    Prepare dataset

    :param ARGUMENTS: arguments dict   :type ARGUMENTS: dict
    :return: train & test data and labels   :rtype: list
    """
    if ARGUMENTS['dataset'] == 'bbc':
        return prepare_dataset_bbc(ARGUMENTS)
    elif ARGUMENTS['dataset'] == 'reuters':
        return prepare_dataset_reuters(ARGUMENTS)
    elif ARGUMENTS['dataset'] == 'ag_news':
        return prepare_dataset_ag_news(ARGUMENTS)

    raise Exception('Not supporting dataset')


def prepare_dataset_bbc(ARGUMENTS):
    """
    Prepare dataset (BBC)

    :param ARGUMENTS: arguments dict   :type ARGUMENTS: dict
    :return: train & test data and labels   :rtype: list
    """
    df = pd.read_csv(ARGUMENTS['dataset_path'])
    # print(df.head())

    texts = df[['text']].to_numpy().reshape(-1)
    labels = df[['category']].to_numpy().reshape(-1)

    n_texts = len(texts)
    print('Texts in dataset: %d' % n_texts)

    categories = utils.get_categories(df, 'category')
    print('Number of categories: %d' % len(categories))

    texts_train, texts_test, labels_train, labels_test = train_test_split(texts,
                                                                          labels,
                                                                          test_size=ARGUMENTS['test_split'])

    # vectorizer to convert the text into numerical features
    [texts_train, texts_test] = fit_vectorizer(ARGUMENTS, texts_train, texts_test)

    plot_dataset_distribution(df['category'])

    print('Train size:', np.shape(texts_train))
    print('Test size:', np.shape(texts_test))

    return [texts_train, texts_test, labels_train, labels_test, categories]


def prepare_dataset_reuters(ARGUMENTS):
    """
    Prepare dataset (Reuters)

    :param ARGUMENTS: arguments dict   :type ARGUMENTS: dict
    :return: train & test data and labels   :rtype: list
    """
    if ARGUMENTS['download_dataset']:
        nltk.download('stopwords')
        nltk.download('reuters')

    # get list with documents ids
    documents = reuters.fileids()

    train_docs_id = list(filter(lambda doc: doc.startswith("train"),
                                documents))
    test_docs_id = list(filter(lambda doc: doc.startswith("test"),
                               documents))

    # get labels and plot distribution
    labels_train_raw = [reuters.categories(doc_id) for doc_id in train_docs_id]
    labels_test_raw = [reuters.categories(doc_id) for doc_id in test_docs_id]

    labels_train_raw_flattened = [label for sublist in labels_train_raw for label in sublist]
    labels_test_raw_flattened = [label for sublist in labels_test_raw for label in sublist]

    plot_dataset_distribution(labels_train_raw_flattened + labels_test_raw_flattened)

    # transform multilabel labels
    mlb = MultiLabelBinarizer()
    labels_train = mlb.fit_transform(labels_train_raw)
    labels_test = mlb.transform(labels_test_raw)

    train_docs = [reuters.raw(doc_id) for doc_id in train_docs_id]
    test_docs = [reuters.raw(doc_id) for doc_id in test_docs_id]

    # vectorizer to convert the text into numerical features
    [texts_train, texts_test] = fit_vectorizer(ARGUMENTS, train_docs, test_docs)

    print('Train size:', np.shape(texts_train))
    print('Test size:', np.shape(texts_test))

    return [texts_train, texts_test, labels_train, labels_test,
            set(labels_train_raw_flattened + labels_test_raw_flattened)]


def prepare_dataset_ag_news(ARGUMENTS):
    """
    Prepare dataset (AG News)

    :param ARGUMENTS: arguments dict   :type ARGUMENTS: dict
    :return: train & test data and labels   :rtype: list
    """
    df_train = pd.read_csv(ARGUMENTS['dataset_path'] + '/train.csv')
    df_test = pd.read_csv(ARGUMENTS['dataset_path'] + '/test.csv')

    df_train['text'] = (df_train['Title'] + df_train['Description'])
    df_test['text'] = (df_test['Title'] + df_test['Description'])

    texts_train = df_train[['text']].to_numpy().reshape(-1)
    labels_train = df_train[['Class Index']].to_numpy().reshape(-1)

    texts_test = df_test[['text']].to_numpy().reshape(-1)
    labels_test = df_test[['Class Index']].to_numpy().reshape(-1)

    n_texts = len(texts_train) + len(texts_test)
    print('Texts in dataset: %d' % n_texts)

    categories = utils.get_categories(df_train, 'Class Index')
    print('Number of categories: %d' % len(categories))

    # vectorizer to convert the text into numerical features
    [texts_train, texts_test] = fit_vectorizer(ARGUMENTS, texts_train, texts_test)

    plot_dataset_distribution(pd.concat([df_train, df_test], ignore_index=True)['Class Index'])

    print('Train size:', np.shape(texts_train))
    print('Test size:', np.shape(texts_test))

    return [texts_train, texts_test, labels_train, labels_test, categories]


def fit_vectorizer(ARGUMENTS, train_docs, test_docs):
    """
    Fit & transform dataset train and test

    :param ARGUMENTS: arguments dict   :type ARGUMENTS: dict
    :param train_docs: list of documents for training   :type ARGUMENTS: list
    :param test_docs: list of documents for testing   :type ARGUMENTS: list
    :return: train & test data   :rtype: list
    """
    vectorizer = None
    if ARGUMENTS['vectorizer'] == 'count':
        vectorizer = CountVectorizer(strip_accents=None, lowercase=False)
    elif ARGUMENTS['vectorizer'] == 'tfidf':
        vectorizer = TfidfVectorizer(tokenizer=utils.tokenize, token_pattern=None, min_df=3, max_df=0.90,
                                     max_features=1000, use_idf=True, sublinear_tf=True)

    if vectorizer is None:
        raise Exception('Not supporting vectorizer')

    start_time = time.time()
    texts_train = vectorizer.fit_transform(train_docs)
    texts_test = vectorizer.transform(test_docs)
    print('Vectorizer processing time: {}'.format(time.time() - start_time))

    return [texts_train, texts_test]


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
    plt.savefig('runs/' + utils.get_next_run_director_name() + '/dataset_class_distribution.jpg', dpi=fig.dpi)
