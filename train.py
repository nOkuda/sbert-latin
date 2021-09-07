from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
import transformers
from fuzzywuzzy import fuzz
from torch.utils.data import DataLoader, RandomSampler
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
    labelled_examples = get_labelled_examples()
    generator = torch.Generator()
    generator = generator.manual_seed(int(initial_seeds[0]))
    dataloader = DataLoader(
        labelled_examples,
        # need to keep batch sizes small enough to fit into GPU
        batch_size=8,
        sampler=RandomSampler(labelled_examples, generator=generator),
        collate_fn=collate)
    device = torch.device('cuda')
    model = SBertLatin(
        bertPath='/home/okuda/Code/latin-bert/models/latin_bert/')
    model.to(device)
    loss_model = CosineSimilarityLoss(model)
    loss_model.to(device)
    optimizer = get_optimizer(loss_model)
    max_grad_norm = 1
    loss_model.zero_grad()
    loss_model.train()
    for features, labels in dataloader:
        optimizer.zero_grad()
        loss_value = loss_model(features, labels)
        loss_value.backward()
        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
        optimizer.step()


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
