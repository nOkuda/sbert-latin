import math
from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
import transformers
from fuzzywuzzy import fuzz
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, matthews_corrcoef,
                             mean_squared_error)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm.autonotebook import trange

from sbert_latin.data import get_aen_luc_benchmark
from sbert_latin.reader import build_tagkeeper
from sbert_latin.sbertlatin import CosineSimilarityLoss, SBertLatin
from sbert_latin.tf_text_encoder import SubwordTextEncoder
from sbert_latin.tokenize import LatinTokenizer, LatinWordTokenizer


def _main():
    RANDOM_SEED = 12345
    outdir = Path(__file__).parent.resolve() / 'output_cross_val'
    labelled_examples = get_labelled_examples()
    X = np.arange(len(labelled_examples))
    y = np.array([0 if a.label <= 0.25 else 1 for a in labelled_examples])
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits,
                          shuffle=True,
                          random_state=RANDOM_SEED)
    skf_iter = iter(skf.split(X, y))
    results_pred = []
    results_true = []
    for k in trange(n_splits, desc='Fold'):
        train_inds, test_inds = next(skf_iter)
        train_data = [labelled_examples[ti] for ti in train_inds]
        test_data = [labelled_examples[ti] for ti in test_inds]
        folddir = outdir / 'fold' / f'{k}'
        train_pred, test_pred = train_model(RANDOM_SEED,
                                            train_data,
                                            test_data,
                                            folddir,
                                            epochs=40)
        train_true = [0 if a.label <= 0.25 else 1 for a in train_data]
        results_true.extend([0 if a.label <= 0.25 else 1 for a in test_data])
        lr = LogisticRegression(dual=False, class_weight='balanced')
        lr.fit(np.array(train_pred).reshape(-1, 1), train_true)
        results_pred.extend(lr.predict(np.array(test_pred).reshape(-1, 1)))
    resultspath = outdir / 'results.txt'
    with resultspath.open('w', encoding='utf-8') as ofh:
        for pred, val in zip(results_pred, results_true):
            ofh.write(f'{pred}\t{val}\n')
    results_pred = np.array(results_pred).reshape(-1, 1)
    mcc = matthews_corrcoef(results_pred, results_true)
    print(mcc)
    with open(str(outdir / 'mcc.txt'), 'w', encoding='utf-8') as ofh:
        ofh.write(f'{mcc}\n')
    conf_mat = confusion_matrix(results_true, results_pred)
    categories = ['meaningless', 'meaningful']
    identifier = 'sbert_latin'
    modelname = 'weighted_logistic_regression'
    save_confusion_matrix(conf_mat, categories, outdir, identifier, modelname)


@dataclass
class LabelledExample:
    sentences: List[str]
    # input_ids[0] corresponds with sentences[0] and is a 1d array containing
    # an integer for every subtoken in sentences[0]; likewise for input_ids[1]
    input_ids: List[np.array]
    # transforms[0] corresponds with input_ids[0] and is a 2d array with shape
    # (len(sentences), len(input_ids[0])); the idea is to transform LatinBERT
    # embeddings from subtoken space to token space; likewise for transforms[1]
    transforms: List[np.array]
    label: float


def get_labelled_examples():
    tess_texts_dir = Path('~/Code/tesserae/texts/la/').expanduser().resolve()
    name_to_filepath = {
        'aeneid': tess_texts_dir / 'vergil.aeneid.tess',
        'lucan': tess_texts_dir / 'lucan.bellum_civile.tess',
    }
    name_to_tagkeeper = {
        name: build_tagkeeper(filepath)
        for name, filepath in name_to_filepath.items()
    }
    benchmark = get_aen_luc_benchmark()
    # write_found_sentences(name_to_tagkeeper, benchmark)
    word_tokenizer = LatinWordTokenizer()
    subword_encoder = LatinTokenizer(
        SubwordTextEncoder(
            '/home/okuda/Code/latin-bert/models/subword_tokenizer_latin/'
            'latin.subword.encoder'))
    results = []
    for (aen_tag, luc_tag), values in benchmark.items():
        for ((aen_snip, luc_snip), label) in values:
            aen_sent = name_to_tagkeeper['aeneid'].get_sentence(
                aen_tag, aen_snip)
            aen_inputs, aen_trans = extract_model_inputs(
                aen_sent, word_tokenizer, subword_encoder)
            luc_sent = name_to_tagkeeper['lucan'].get_sentence(
                luc_tag, luc_snip)
            luc_inputs, luc_trans = extract_model_inputs(
                luc_sent, word_tokenizer, subword_encoder)
            results.append(
                LabelledExample(
                    sentences=[aen_sent, luc_sent],
                    input_ids=[aen_inputs, luc_inputs],
                    transforms=[aen_trans, luc_trans],
                    # force label to be between 0 and 1
                    label=(label - 1) / 4))
    return results


def train_model(random_seed,
                train_data,
                test_data,
                folddir,
                epochs=5,
                batch_size=8):
    train_eval_dir = folddir / 'train_eval'
    test_eval_dir = folddir / 'test_eval'
    write_data(train_data, train_eval_dir)
    write_data(test_data, test_eval_dir)
    torch.manual_seed(int(random_seed))
    generator = torch.Generator()
    generator = generator.manual_seed(int(random_seed))
    dataloader = DataLoader(
        train_data,
        # need to keep batch sizes small enough to fit into GPU
        batch_size=batch_size,
        sampler=RandomSampler(train_data, generator=generator),
        collate_fn=collate)
    steps_per_epoch = math.ceil(len(train_data) / batch_size)
    device = torch.device('cuda')
    model = SBertLatin(
        bertPath='/home/okuda/Code/latin-bert/models/latin_bert/')
    model.to(device)
    loss_model = CosineSimilarityLoss(model)
    loss_model.to(device)
    optimizer = get_optimizer(loss_model)
    max_grad_norm = 1
    for epoch in trange(epochs, desc='Epoch'):
        loss_model.zero_grad()
        loss_model.train()
        data_iter = iter(dataloader)
        training_log = trange(steps_per_epoch, desc='Training', smoothing=0.05)
        training_log.write(f'#### {epoch}')
        for _ in training_log:
            try:
                features, labels = next(data_iter)
            except StopIteration:
                continue
            optimizer.zero_grad()
            loss_value = loss_model(features, labels)
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(loss_model.parameters(),
                                           max_grad_norm)
            optimizer.step()
        train_values = evaluate(model, train_data, batch_size, epoch,
                                train_eval_dir)
        train_score = train_values['spearman']
        training_log.write(f'Training data score: {train_score}')
        test_values = evaluate(model, test_data, batch_size, epoch,
                               test_eval_dir)
        test_score = test_values['spearman']
        training_log.write(f'Test data score: {test_score}')
        if epoch % 10 == 0:
            training_log.write(f'Saving current model (epoch {epoch})')
            model_path = folddir / 'checkpoint'
            model.save_embedder(str(model_path))
    model_path = folddir / 'final_model'
    model.save_embedder(str(model_path))
    train_pred, _ = predict(model, train_data, batch_size, 'final train')
    test_pred, _ = predict(model, test_data, batch_size, 'final test')
    return train_pred, test_pred


def write_data(data: List[LabelledExample], outdir: Path):
    if not outdir.exists():
        outdir.mkdir(parents=True)
    outpath = outdir / 'parallels.txt'
    with outpath.open('w', encoding='utf-8') as ofh:
        for example in data:
            s0 = example.sentences[0]
            s1 = example.sentences[1]
            ofh.write(f'{s0}\t{s1}\t{example.label}\n')


def get_optimizer(loss_model):
    weight_decay = 0.01
    optimizer_params = {'lr': 2e-5}
    optimizer_class = transformers.AdamW
    param_optimizer = list(loss_model.named_parameters())

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [{
        'params':
        [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay':
        weight_decay
    }, {
        'params':
        [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay':
        0.0
    }]

    return optimizer_class(optimizer_grouped_parameters, **optimizer_params)


def collate(batch: List[LabelledExample]):
    device = torch.device('cuda')
    results = []
    for i in range(len(batch[0].input_ids)):
        input_ids, attention_mask, transforms = collate_helper(batch, i)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        transforms = transforms.to(device)
        results.append({
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'transforms': transforms
        })
    return results, torch.tensor([b.label for b in batch]).to(device)


def collate_helper(batch: List[LabelledExample], ind: int):
    max_len = np.array([len(a.input_ids[ind]) for a in batch]).max()
    max_sent_len = max([a.transforms[ind].shape[0] for a in batch])
    batch_input_ids = torch.zeros((len(batch), max_len), dtype=torch.long)
    batch_attention_masks = torch.zeros((len(batch), max_len),
                                        dtype=torch.float)
    batch_transforms = torch.zeros((len(batch), max_sent_len, max_len),
                                   dtype=torch.float)
    for i, example in enumerate(batch):
        input_ids = example.input_ids[ind]
        batch_input_ids[i, :len(input_ids)] = torch.from_numpy(input_ids)
        batch_attention_masks[i, :len(input_ids)] = 1
        sent_len = example.transforms[ind].shape[0]
        batch_transforms[i, :sent_len, :len(input_ids)] = torch.from_numpy(
            example.transforms[ind])
    return batch_input_ids, batch_attention_masks, batch_transforms


def evaluate(model, data, batch_size, epoch, outdir):
    results = {}
    if not outdir.exists():
        outdir.mkdir(parents=True)
    predictions, true_labels = predict(model, data, batch_size, str(outdir))
    pearson_value, _ = pearsonr(predictions, true_labels)
    spearman_value, _ = spearmanr(predictions, true_labels)
    mse_value = mean_squared_error(predictions, true_labels)
    prevalues = [epoch, pearson_value, spearman_value, mse_value]
    values = '\t'.join([str(a) for a in prevalues])
    results['pearson'] = pearson_value
    results['spearman'] = spearman_value
    results['mse'] = mse_value
    recordpath = outdir / 'record.txt'
    if recordpath.exists():
        with recordpath.open('a', encoding='utf-8') as ofh:
            ofh.write(f'{values}\n')
    else:
        with recordpath.open('w', encoding='utf-8') as ofh:
            ofh.write('Epoch\tPearson\tSpearman\tMSE\n')
            ofh.write(f'{values}\n')
    predictionspath = outdir / f'predictions_{epoch:04}.txt'
    with predictionspath.open('w', encoding='utf=8') as ofh:
        for pred, val in zip(predictions, true_labels):
            ofh.write(f'{pred}\t{val}\n')
    return results


def predict(model, data, batch_size, msg):
    expected_iters = math.ceil(len(data) / batch_size)
    dataloader = DataLoader(data,
                            batch_size=batch_size,
                            sampler=SequentialSampler(data),
                            collate_fn=collate)
    data_iter = iter(dataloader)
    true_labels = []
    predictions = []
    with torch.no_grad():
        for _ in trange(expected_iters,
                        desc=f'Evaluation ({msg})',
                        smoothing=0.05):
            try:
                features, labels = next(data_iter)
            except StopIteration:
                continue
            true_labels.extend(labels.tolist())
            embeddings = [
                model(sentence_feature)['sentence_embedding']
                for sentence_feature in features
            ]
            predictions.extend(
                torch.cosine_similarity(embeddings[0], embeddings[1]).tolist())
    return predictions, true_labels


def extract_model_inputs(sentence: str, word_tokenizer: LatinWordTokenizer,
                         subword_encoder: SubwordTextEncoder):
    tokens = word_tokenizer.tokenize_with_spans(sentence)[0]
    filt_toks = ['[CLS]']
    for tok in tokens:
        if tok != "":
            filt_toks.append(tok)
    filt_toks.append('[SEP]')

    input_ids = []
    transform = []
    all_toks = []
    n = 0
    for idx, word in enumerate(filt_toks):
        toks = subword_encoder.tokenize(word)
        all_toks.append(toks)
        n += len(toks)
    cur = 0
    for idx, word in enumerate(filt_toks):
        toks = all_toks[idx]
        ind = list(np.zeros(n))
        for j in range(cur, cur + len(toks)):
            ind[j] = 1. / len(toks)
        cur += len(toks)
        transform.append(ind)
        input_ids.extend(subword_encoder.convert_tokens_to_ids(toks))
    return np.array(input_ids), np.array(transform)


def write_found_sentences(name_to_tagkeeper, benchmark):
    aen_founds = []
    luc_founds = []
    for (aen_tag, luc_tag), values in benchmark.items():
        for ((aen_snip, luc_snip), label) in values:
            aen_sent = name_to_tagkeeper['aeneid'].get_sentence(
                aen_tag, aen_snip)
            aen_founds.append(
                (fuzz.ratio(aen_snip, aen_sent), aen_tag, aen_snip, aen_sent))
            luc_sent = name_to_tagkeeper['lucan'].get_sentence(
                luc_tag, luc_snip)
            luc_founds.append(
                (fuzz.ratio(luc_snip, luc_sent), luc_tag, luc_snip, luc_sent))
    aen_founds.sort()
    luc_founds.sort()
    with open('aen_founds.txt', 'w') as ofh:
        for found in aen_founds:
            ofh.write(f'{found}\n')
    with open('luc_founds.txt', 'w') as ofh:
        for found in luc_founds:
            ofh.write(f'{found}\n')


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
