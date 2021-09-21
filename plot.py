"""Plot evaluation scores"""
from collections import deque
from itertools import cycle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _main():
    curdir = Path(__file__).parent.resolve()
    resultsdir = curdir / 'output' / '2096804712593481934'
    label_to_dir = {
        'train': resultsdir / 'train_eval',
        'dev': resultsdir / 'dev_eval',
        'test': resultsdir / 'test_eval',
    }
    plot_spearman(label_to_dir)
    plot_mse(label_to_dir)


def plot_spearman(label_to_dir):
    colorcycler = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    markercycler = cycle(['o', 's', '^'])
    for label, evaldir in label_to_dir.items():
        datapoints = read_record(evaldir)
        win_avg = windowed_average(datapoints, window_size=10)
        cur_color = next(colorcycler)
        cur_marker = next(markercycler)
        plt.scatter(np.arange(len(datapoints)),
                    datapoints,
                    label=label,
                    alpha=0.5,
                    marker=cur_marker,
                    color=cur_color)
        plt.plot(win_avg, color=cur_color)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Spearman\'s Rank Correlation')
    plt.title('Rank Correlation of Predictions to Labels')
    plt.savefig('eval_plot.svg')
    plt.close()


def plot_mse(label_to_dir):
    colorcycler = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    markercycler = cycle(['o', 's', '^'])
    for label, evaldir in label_to_dir.items():
        datapoints = read_mse(evaldir)
        win_avg = windowed_average(datapoints, window_size=10)
        cur_color = next(colorcycler)
        cur_marker = next(markercycler)
        plt.scatter(np.arange(len(datapoints)),
                    datapoints,
                    label=label,
                    alpha=0.5,
                    marker=cur_marker,
                    color=cur_color)
        plt.plot(win_avg, color=cur_color)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.title('Mean Squared Error of Predictions to Labels')
    plt.savefig('loss_plot.svg')
    plt.close()


def read_record(evaldir):
    recordpath = evaldir / 'record.txt'
    with recordpath.open('r', encoding='utf-8') as ifh:
        # skip header
        next(ifh)
        return [float(line.strip().split('\t')[2]) for line in ifh]


def read_mse(evaldir):
    recordpath = evaldir / 'record.txt'
    with recordpath.open('r', encoding='utf-8') as ifh:
        # skip header
        next(ifh)
        return [float(line.strip().split('\t')[3]) for line in ifh]


def windowed_average(datapoints, window_size=5):
    window = deque(maxlen=window_size)
    result = []
    for point in datapoints:
        window.append(point)
        result.append(np.mean(window))
    return result


if __name__ == '__main__':
    _main()
