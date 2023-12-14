import glob
import json
import os
import pprint

from matplotlib import pyplot as plt


def get_categories(df):
    """
    Get all categories

    :param df: df   :type df: dict
    :return: categories    :rtype: set
    """
    return df['category'].unique()


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


def save_parameters(PARAMETERS):
    """
    Save training parameters

    :param PARAMETERS: arguments dict    :type PARAMETERS: dict
    """
    print('\n')
    print('PARAMETERS: ')
    pprint.pprint(PARAMETERS)
    print('\n')

    dir_name = get_next_run_director_name()
    if not os.path.isdir('runs/' + dir_name):
        os.mkdir('runs/' + dir_name)

    with open('runs/' + get_next_run_director_name() + '/PARAMETERS.pkl', 'w') as convert_file:
        convert_file.write(json.dumps(PARAMETERS))


def add_labels(x, y):
    """
    Add labels for charts

    :param x: data    :type x: list
    :param y: distribution     :type y: list
    """
    for i in range(len(x)):
        plt.text(i, y[i] + 0.2, "{:.2f}".format(y[i]) + ' %', ha='center', fontsize=16)