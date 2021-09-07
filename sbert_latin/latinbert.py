"""LatinBERT

Modified from https://github.com/dbamman/latin-bert/blob/
d6bea9f7ff84ff4b18c172f3d5719d1d3198e69/scripts/gen_berts.py to work with
transformers v4.x
"""

import numpy as np
import torch
from torch import nn
from transformers import BertModel

from .tf_text_encoder import SubwordTextEncoder
from .tokenize import LatinTokenizer, LatinWordTokenizer


class LatinBERT():

    def __init__(self, tokenizerPath=None, bertPath=None):
        encoder = SubwordTextEncoder(tokenizerPath)
        self.wp_tokenizer = LatinTokenizer(encoder)
        self.model = BertLatin(bertPath=bertPath)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def get_berts_and_spans(self, raw_sents):
        """Generates BERT embeddings with associated spans

        Assumes that ``raw_sents`` has already been sentence tokenized
        """
        sents, spans = convert_to_toks_and_spans(raw_sents)
        batch_size = 32
        batched_data, batched_mask, batched_transforms, ordering = \
            self.get_batches(sents, batch_size, self.wp_tokenizer)

        ordered_preds = []
        for b in range(len(batched_data)):
            size = batched_transforms[b].shape
            b_size = size[0]
            berts = self.model.forward(batched_data[b],
                                       attention_mask=batched_mask[b],
                                       transforms=batched_transforms[b])
            berts = berts.detach()
            berts = berts.cpu()
            for row in range(b_size):
                ordered_preds.append([np.array(r) for r in berts[row]])

        preds_in_order = [None for i in range(len(sents))]

        for i, ind in enumerate(ordering):
            preds_in_order[ind] = ordered_preds[i]

        bert_sents = []

        for idx, sentence in enumerate(sents):
            bert_sent = []

            bert_sent.append(("[CLS]", preds_in_order[idx][0]))

            for t_idx in range(1, len(sentence) - 1):
                token = sentence[t_idx]

                pred = preds_in_order[idx][t_idx]
                bert_sent.append((token, pred))

            bert_sent.append(("[SEP]", preds_in_order[idx][len(sentence) - 1]))

            bert_sents.append(bert_sent)

        return bert_sents, spans

    def get_batches(self, sentences, max_batch, tokenizer):

        all_data = []
        all_masks = []
        all_labels = []
        all_transforms = []

        for sentence in sentences:
            tok_ids = []
            input_mask = []
            labels = []
            transform = []

            all_toks = []
            n = 0
            for idx, word in enumerate(sentence):
                toks = tokenizer.tokenize(word)
                all_toks.append(toks)
                n += len(toks)

            cur = 0
            for idx, word in enumerate(sentence):
                toks = all_toks[idx]
                ind = list(np.zeros(n))
                for j in range(cur, cur + len(toks)):
                    ind[j] = 1. / len(toks)
                cur += len(toks)
                transform.append(ind)

                tok_ids.extend(tokenizer.convert_tokens_to_ids(toks))

                input_mask.extend(np.ones(len(toks)))
                labels.append(1)

            all_data.append(tok_ids)
            all_masks.append(input_mask)
            all_labels.append(labels)
            all_transforms.append(transform)

        lengths = np.array([len(d) for d in all_data])

        # Note sequence must be ordered from shortest to longest so
        # current_batch will work
        ordering = np.argsort(lengths)

        ordered_data = [None for i in range(len(all_data))]
        ordered_masks = [None for i in range(len(all_data))]
        ordered_labels = [None for i in range(len(all_data))]
        ordered_transforms = [None for i in range(len(all_data))]

        for i, ind in enumerate(ordering):
            ordered_data[i] = all_data[ind]
            ordered_masks[i] = all_masks[ind]
            ordered_labels[i] = all_labels[ind]
            ordered_transforms[i] = all_transforms[ind]

        batched_data = []
        batched_mask = []
        batched_labels = []
        batched_transforms = []

        i = 0
        current_batch = max_batch

        while i < len(ordered_data):

            batch_data = ordered_data[i:i + current_batch]
            batch_mask = ordered_masks[i:i + current_batch]
            batch_labels = ordered_labels[i:i + current_batch]
            batch_transforms = ordered_transforms[i:i + current_batch]

            max_len = max([len(sent) for sent in batch_data])
            max_label = max([len(label) for label in batch_labels])

            for j in range(len(batch_data)):

                blen = len(batch_data[j])
                blab = len(batch_labels[j])

                for k in range(blen, max_len):
                    batch_data[j].append(0)
                    batch_mask[j].append(0)
                    for z in range(len(batch_transforms[j])):
                        batch_transforms[j][z].append(0)

                for k in range(blab, max_label):
                    batch_labels[j].append(-100)

                for k in range(len(batch_transforms[j]), max_label):
                    batch_transforms[j].append(np.zeros(max_len))

            batched_data.append(torch.LongTensor(batch_data))
            batched_mask.append(torch.FloatTensor(batch_mask))
            batched_labels.append(torch.LongTensor(batch_labels))
            batched_transforms.append(torch.FloatTensor(batch_transforms))

            i += current_batch

            # adjust batch size; sentences are ordered from shortest to longest
            # so decrease as they get longer
            if max_len > 100:
                current_batch = 12
            if max_len > 200:
                current_batch = 6

        return batched_data, batched_mask, batched_transforms, ordering


def convert_to_toks_and_spans(sents):

    word_tokenizer = LatinWordTokenizer()

    all_sents = []
    all_spans = []

    for data in sents:
        text = data.lower()
        tokens, spans = word_tokenizer.tokenize_with_spans(text)
        filt_toks = []
        filt_toks.append("[CLS]")
        filt_spans = [(0, 0)]
        for tok, span in zip(tokens, spans):
            if tok != "":
                filt_toks.append(tok)
                filt_spans.append(span)
        filt_toks.append("[SEP]")
        max_pos = max(max(a[0] for a in filt_spans),
                      max(a[1] for a in filt_spans))
        filt_spans.append((max_pos, max_pos))

        all_sents.append(filt_toks)
        all_spans.append(filt_spans)

    return all_sents, all_spans


class BertLatin(nn.Module):
    """LatinBERT model

    On a forward pass, the model expects a batch of subtoken input ids, an
    attention mask indicating which of the subtoken input ids are relevant,
    and a transforms matrix that converts the subtoken embeddings into token
    embeddings. As output, the forward pass yields a dictionary containing the
    following entries:
        last_hidden_state: the subtoken embeddings
        attention_mask: mask for subtoken inputs
        token_embeddings: the token embeddings
        token_attention_mask: mask for token embeddings
        pooler_output: unknown
    """

    def __init__(self, bertPath=None):
        super(BertLatin, self).__init__()

        self.bert = BertModel.from_pretrained(bertPath)
        self.bert.eval()
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, features):
        input_ids = features['input_ids']
        attention_mask = features['attention_mask']
        transforms = features['transforms']

        bert_outs = self.bert.forward(input_ids,
                                      token_type_ids=None,
                                      attention_mask=attention_mask)
        bert_outs['attention_mask'] = attention_mask
        bert_outs['token_embeddings'] = torch.matmul(
            transforms, bert_outs['last_hidden_state'])
        mask_shape = attention_mask.shape
        bert_outs['token_attention_mask'] = torch.bmm(
            transforms,
            attention_mask.reshape(
                (mask_shape[0], mask_shape[1], 1))).squeeze()
        return bert_outs

    def save_embedder(self, outpath):
        self.bert.save_pretrained(outpath)
