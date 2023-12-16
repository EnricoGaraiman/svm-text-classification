import glob
import json
import os
import pprint
import re
from matplotlib import pyplot as plt
import spacy
from nltk import word_tokenize, PorterStemmer
from nltk.corpus import stopwords

nlp = spacy.load("en_core_web_md")
cachedStopWords = stopwords.words("english")


def get_categories(df, key):
    """
    Get all categories

    :param df: df   :type df: dict
    :param key: key   :type key: string
    :return: categories    :rtype: set
    """
    return df[key].unique()


def get_next_run_director_name():
    """
    Get director name

    :return: director name    :rtype: str
    """
    dirs = sorted(glob.glob('runs/run-*'), key=os.path.getmtime)
    if len(dirs) == 0:
        return 'run-1'
    else:
        if len(glob.glob(dirs[-1] + '/confusion-matrix.jpg')) == 0:
            return 'run-' + str(int(dirs[-1].split('\\')[-1].split('-')[-1]))
        return 'run-' + str(int(dirs[-1].split('\\')[-1].split('-')[-1]) + 1)


def save_parameters(ARGUMENTS):
    """
    Save training ARGUMENTS

    :param ARGUMENTS: arguments dict    :type ARGUMENTS: dict
    """
    print('\n')
    print('ARGUMENTS: ')
    pprint.pprint(ARGUMENTS)
    print('\n')

    dir_name = get_next_run_director_name()
    if not os.path.isdir('runs/' + dir_name):
        os.mkdir('runs/' + dir_name)

    with open('runs/' + get_next_run_director_name() + '/ARGUMENTS.pkl', 'w') as convert_file:
        convert_file.write(json.dumps(ARGUMENTS))


def add_labels(x, y):
    """
    Add labels for charts

    :param x: data    :type x: list
    :param y: distribution     :type y: list
    """
    for i in range(len(x)):
        plt.text(i, y[i] + 0.2, "{:.2f}".format(y[i]) + ' %', ha='center', fontsize=16)


def tokenize(text):
    """
    Tokenize text

    :param text: text    :type x: string
    :return y: filtered tokens     :rtype y: list
    """
    min_length = 3
    words = map(lambda word: word.lower(), word_tokenize(text))
    words = [word for word in words if word not in cachedStopWords]
    tokens = (list(map(lambda token: PorterStemmer().stem(token), words)))
    p = re.compile('[a-zA-Z]+')
    filtered_tokens = list(filter(lambda token:
                                  p.match(token) and len(token) >= min_length,
                                  tokens))
    return filtered_tokens
