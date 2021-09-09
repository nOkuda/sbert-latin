import math
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
import transformers
from fuzzywuzzy import fuzz
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm.autonotebook import trange

from sbert_latin.data import get_aen_luc_benchmark
from sbert_latin.reader import build_tagkeeper
from sbert_latin.sbertlatin import CosineSimilarityLoss, SBertLatin
from sbert_latin.tf_text_encoder import SubwordTextEncoder
from sbert_latin.tokenize import LatinTokenizer, LatinWordTokenizer


def _main():
    RANDOM_SEED = 12345
    rng = np.random.default_rng(RANDOM_SEED)
    initial_seeds = rng.integers(np.iinfo(np.int64).max, size=10)
    outdir = Path(__file__).parent.resolve() / 'output'
    labelled_examples = get_labelled_examples()
    train_model(initial_seeds[0], labelled_examples, outdir)


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


def train_model(initial_seed,
                labelled_examples,
                outdir,
                epochs=150,
                batch_size=8):
    train_data, dev_data, test_data = split_data(labelled_examples,
                                                 random_seed=initial_seed)
    train_eval_dir = outdir / f'{initial_seed}' / 'train_eval'
    dev_eval_dir = outdir / f'{initial_seed}' / 'dev_eval'
    test_eval_dir = outdir / f'{initial_seed}' / 'test_eval'
    write_data(train_data, train_eval_dir)
    write_data(dev_data, dev_eval_dir)
    write_data(test_data, test_eval_dir)
    torch.manual_seed(int(initial_seed))
    generator = torch.Generator()
    generator = generator.manual_seed(int(initial_seed))
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
    best_score = -999999999
    best_epoch = -1
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
        dev_values = evaluate(model, dev_data, batch_size, epoch, dev_eval_dir)
        dev_score = dev_values['spearman']
        training_log.write(f'Development data score: {dev_score}')
        test_values = evaluate(model, test_data, batch_size, epoch,
                               test_eval_dir)
        test_score = test_values['spearman']
        training_log.write(f'Test data score: {test_score}')
        if best_score < dev_score:
            best_score = dev_score
            best_epoch = epoch
            training_log.write(f'Best epoch so far: {best_epoch}')
            model_path = outdir / f'{initial_seed}' / 'best_model'
            model.save_embedder(str(model_path))
        if epoch % 10 == 0:
            training_log.write(f'Saving current model (epoch {epoch})')
            model_path = outdir / f'{initial_seed}' / 'checkpoint'
            model.save_embedder(str(model_path))
    model_path = outdir / f'{initial_seed}' / 'final_model'
    model.save_embedder(str(model_path))
    print(f'Best epoch: {best_epoch}')
    return model


def split_data(examples,
               random_seed=None,
               train_portion=0.8,
               dev_portion=0.1,
               test_portion=0.1):
    portion_sum = train_portion + dev_portion + test_portion
    train_portion = train_portion / portion_sum
    dev_portion = dev_portion / portion_sum
    test_portion = test_portion / portion_sum
    labels = np.array([ex.label for ex in examples])
    sort_idxs = np.argsort(labels)
    # labels are spaced out 0.25 apart
    label_change_idxs = np.nonzero(np.diff(labels[sort_idxs]) >= 0.2)[0] + 1
    rng = np.random.default_rng(random_seed)
    train_data_idxs = []
    dev_data_idxs = []
    test_data_idxs = []
    prev_start = 0
    for idx in label_change_idxs:
        rng.shuffle(sort_idxs[prev_start:idx])
        cur_class_size = idx - prev_start
        train_end = prev_start + int(cur_class_size * train_portion)
        train_data_idxs.extend([a for a in sort_idxs[prev_start:train_end]])
        dev_end = train_end + int(cur_class_size * dev_portion)
        dev_data_idxs.extend([a for a in sort_idxs[train_end:dev_end]])
        test_data_idxs.extend([a for a in sort_idxs[dev_end:idx]])
        prev_start = idx
    # don't forget about the last class
    rng.shuffle(sort_idxs[prev_start:])
    cur_class_size = sort_idxs.size - prev_start
    train_end = prev_start + int(cur_class_size * train_portion)
    train_data_idxs.extend([a for a in sort_idxs[prev_start:train_end]])
    dev_end = train_end + int(cur_class_size * dev_portion)
    dev_data_idxs.extend([a for a in sort_idxs[train_end:dev_end]])
    test_data_idxs.extend([a for a in sort_idxs[dev_end:sort_idxs.size]])
    train_data = [examples[i] for i in train_data_idxs]
    dev_data = [examples[i] for i in dev_data_idxs]
    test_data = [examples[i] for i in test_data_idxs]
    return train_data, dev_data, test_data


def write_data(data: List[LabelledExample], outdir: Path):
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
                        desc=f'Evaluation ({str(outdir)})',
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


if __name__ == '__main__':
    _main()
