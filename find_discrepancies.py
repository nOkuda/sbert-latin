"""Analyze test set output for score discrepancies

Latin SBERT predicts a similarity score for each pair of passages, and the
Lucan-Vergil benchmark provides an intertext quality rating that can be
transformed to a similarity score as a target label for Latin SBERT to train
against. This script takes the difference between the predicted score and the
label score to see which parallels have the smallest and greatest discrepancies
between those scores.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _main():
    cur_dir = Path(__file__).parent
    output_dir = cur_dir / 'output_cross_val' / 'fold'
    parallels = []
    predictions = []
    for i in range(5):
        test_dir = output_dir / str(i) / 'test_eval'
        parallels.extend(read_parallels(test_dir / 'parallels.txt'))
        predictions.extend(read_predictions(test_dir / 'predictions_0039.txt'))
    diffs = np.abs(np.diff(predictions).reshape(-1))
    label_to_rating = {
        0.0: 1,
        0.25: 2,
        0.5: 3,
        0.75: 4,
        1.0: 5,
    }
    plot_diffs_by_rating_boxplots(diffs,
                                  [label_to_rating[a[1]] for a in predictions])
    sorted_inds = np.argsort(diffs)
    with open('discrepancies.txt', 'w') as ofh:
        for i, ind in enumerate(sorted_inds):
            ofh.write(f'==== {i}')
            ofh.write('\n\n')
            entry = parallels[ind]
            ofh.write(entry.source_snippet)
            ofh.write('\n\n')
            ofh.write(entry.target_snippet)
            ofh.write('\n\n')
            pred, label = predictions[ind]
            ofh.write(f'{diffs[ind]}\t({pred}\t{label})\n\n')


@dataclass
class Entry:
    source_snippet: str
    target_snippet: str
    label: float


def read_parallels(filepath: Path) -> List[Entry]:
    result = []
    with filepath.open() as ifh:
        source_lines = []
        target_lines = []
        for line in ifh:
            if '\t' in line:
                sections = line.split('\t')
                if len(sections) == 2:
                    earlier, later = sections[0], sections[1]
                    if not target_lines:
                        source_lines.append(earlier)
                        target_lines.append(later)
                    else:
                        target_lines.append(earlier)
                        result.append(
                            Entry(source_snippet=''.join(source_lines),
                                  target_snippet=''.join(target_lines),
                                  label=float(later)))
                        source_lines = []
                        target_lines = []
                elif len(sections) == 3:
                    result.append(
                        Entry(source_snippet=sections[0],
                              target_snippet=sections[1],
                              label=sections[2]))
                    source_lines = []
                    target_lines = []
                else:
                    raise Exception(f'Unexpected number of tabs: {line}')
            else:
                if not target_lines:
                    source_lines.append(line)
                else:
                    target_lines.append(line)
    return result


def read_predictions(filepath: Path) -> List[List[float]]:
    result = []
    with filepath.open() as ifh:
        for line in ifh:
            line = line.strip()
            if line:
                pred, label = line.split('\t')
                result.append([float(pred), float(label)])
    return result


def plot_diffs_by_rating_boxplots(diffs: np.array, ratings: List[int]) -> None:
    fig, ax = plt.subplots()
    agg = {
        1: [],
        2: [],
        3: [],
        4: [],
        5: [],
    }
    for d, r in zip(diffs, ratings):
        agg[r].append(d)
    ax.boxplot([agg[i] for i in [1, 2, 3, 4, 5]])
    ax.set_title('Score Discrepancies by Rating')
    ax.set_ylabel('Discrepancy')
    ax.set_xlabel('Rating')
    plt.tight_layout()
    plt.savefig('score_discrepancies_by_rating.svg')
    plt.close()


def plot_diffs_by_rating_violinplot(diffs: np.array,
                                    ratings: List[int]) -> None:
    data = pd.DataFrame({'Discrepancy': diffs, 'Rating': ratings})
    ax = sns.violinplot(x='Rating', y='Discrepancy', data=data)
    ax.set_title('Score Discrepancies by Rating')
    ax.set_ylabel('Discrepancy')
    ax.set_xlabel('Rating')
    plt.tight_layout()
    plt.savefig('score_discrepancies_by_rating.svg')
    plt.close()


if __name__ == '__main__':
    _main()
