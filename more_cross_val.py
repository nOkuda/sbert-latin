"""More machine learning on Latin SBERT predictions"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sklearn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             matthews_corrcoef)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 12345


def _main():
    outdir = Path(__file__).parent.resolve() / 'output_cross_val'
    params = [
        ('normalized_neural_network', build_norm_nn, RatingsTask()),
        ('normalized_neural_network', build_norm_nn, FivesTask()),
        ('normalized_neural_network', build_norm_nn, MeaningfulsTask()),
        ('oversampled_neural_network', build_bal_nn, RatingsTask()),
        ('oversampled_neural_network', build_bal_nn, FivesTask()),
        ('oversampled_neural_network', build_bal_nn, MeaningfulsTask()),
        ('weighted_logistic_regression', build_bal_lr, RatingsTask()),
        ('weighted_logistic_regression', build_bal_lr, FivesTask()),
        ('weighted_logistic_regression', build_bal_lr, MeaningfulsTask()),
    ]
    for model_name, model_builder, task in params:
        print('#', model_name)
        results_pred, results_true = run_model(model_builder, task, outdir)
        save_results(results_pred, results_true, model_name, task, outdir)


def build_norm_nn():
    return make_pipeline(
        SimpleImputer(missing_values=np.nan, strategy='constant'),
        StandardScaler(with_mean=True),
        MLPClassifier(hidden_layer_sizes=(50, ),
                      max_iter=200000,
                      random_state=RANDOM_SEED))


def build_bal_nn():
    return make_pipeline(
        SimpleImputer(missing_values=np.nan, strategy='constant'),
        SMOTE(random_state=RANDOM_SEED),
        MLPClassifier(hidden_layer_sizes=(50, ),
                      max_iter=200000,
                      random_state=RANDOM_SEED))


def build_bal_lr():
    return LogisticRegression(dual=False,
                              class_weight='balanced',
                              random_state=RANDOM_SEED)


class RatingsTask:

    def __init__(self):
        self.task_name = 'ratings'
        self.categories = [str(a) for a in range(1, 6)]

    def convert(self, y):
        """Convert range [0, 1] to labels [0, 4]"""
        y = y * 4
        return y.astype(int)


class FivesTask:

    def __init__(self):
        self.task_name = 'fives'
        self.categories = ['non-5', '5']

    def convert(self, y):
        """Convert range [0, 1] to labels [0, 4]"""
        y = y == 1
        return y.astype(int)


class MeaningfulsTask:

    def __init__(self):
        self.task_name = 'meaningfuls'
        self.categories = ['meaningless', 'meaningful']

    def convert(self, y):
        """Convert range [0, 1] to labels {0, 1}"""
        y = y > 0.25
        return y.astype(int)


def run_model(model_builder, task, outdir):
    results_pred = []
    results_true = []
    for k in range(5):
        train_X, train_y = extract_values(outdir, k, 'train_eval')
        test_X, test_y = extract_values(outdir, k, 'test_eval')
        model = model_builder()
        train_y = task.convert(train_y)
        test_y = task.convert(test_y)
        model.fit(train_X, train_y)
        results_pred.extend(model.predict(test_X))
        results_true.extend(test_y)
    return results_pred, results_true


def extract_values(outdir, k, dirname):
    pred_vals = []
    true_vals = []
    dirpath = outdir / 'fold' / f'{k}' / dirname
    predpath = dirpath / 'predictions_0039.txt'
    with predpath.open('r', encoding='utf-8') as ifh:
        for line in ifh:
            line = line.strip()
            values = line.split('\t')
            pred_vals.append(float(values[0]))
            true_vals.append(float(values[1]))
    return np.array(pred_vals).reshape(-1, 1), np.array(true_vals)


def save_results(results_pred, results_true, model_name, task, outdir):
    task_name = task.task_name
    categories = task.categories
    conf_mat = confusion_matrix(results_true, results_pred)
    save_confusion_matrix(conf_mat, categories, outdir, task_name, model_name)
    PRECISION = 3
    acc = np.format_float_positional(accuracy_score(results_true,
                                                    results_pred),
                                     precision=PRECISION)
    f1 = np.format_float_positional(f1_score(results_true,
                                             results_pred,
                                             average='macro'),
                                    precision=PRECISION)
    mcc = np.format_float_positional(matthews_corrcoef(results_true,
                                                       results_pred),
                                     precision=PRECISION)
    outpath = outdir / 'metrics.txt'
    if not outpath.exists():
        with outpath.open('w', encoding='utf-8') as ofh:
            ofh.write('task\tmodel\tacc\tf1\tmcc\n')
    with outpath.open('a', encoding='utf-8') as ofh:
        ofh.write(f'{task_name}\t{model_name}\t{acc}\t{f1}\t{mcc}\n')


def save_confusion_matrix(conf_mat, categories, data_dir, identifier,
                          modelname):
    # plotting with axis=1 tells me what the model learned;
    normalized_conf_mat = sklearn.preprocessing.normalize(conf_mat,
                                                          axis=1,
                                                          norm='l1')
    learnname = f'learnplot.{identifier}.confusion.{modelname}.svg'
    outplotpath = data_dir / learnname
    plot_confusion_matrix(conf_mat, normalized_conf_mat, categories,
                          outplotpath)
    # plotting with axis=0 tells me how reliable the model's predictions are
    normalized_conf_mat = sklearn.preprocessing.normalize(conf_mat,
                                                          axis=0,
                                                          norm='l1')
    relianame = f'reliabilityplot.{identifier}.confusion.{modelname}.svg'
    outplotpath = data_dir / relianame
    plot_confusion_matrix(conf_mat, normalized_conf_mat, categories,
                          outplotpath)


def plot_confusion_matrix(conf_mat, normalized_conf_mat, categories,
                          outplotpath):
    fig, ax = plt.subplots()
    ax.imshow(normalized_conf_mat)
    ax.set_xticks(np.arange(len(categories)))
    ax.set_yticks(np.arange(len(categories)))
    ax.set_xticklabels(categories)
    ax.set_yticklabels(categories)
    plt.setp(ax.get_xticklabels(),
             rotation=45,
             ha="right",
             rotation_mode="anchor")
    for i in range(len(categories)):
        for j in range(len(categories)):
            ax.text(j, i, conf_mat[i, j], ha="center", va="center", color="w")
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('prediction')
    ax.set_ylabel('benchmark label')
    fig.tight_layout()
    plt.savefig(str(outplotpath))
    fig.clear
    plt.close(fig)


if __name__ == '__main__':
    _main()
