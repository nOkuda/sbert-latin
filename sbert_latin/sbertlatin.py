from collections import OrderedDict
from typing import Dict, Iterable

import torch
from torch import Tensor, nn

from sbert_latin.latinbert import BertLatin


class Pooling(nn.Module):
    """Mean pooling on token or subtoken embeddings

    This layer is expected to come directly after a BertLatin layer. Outputs
    from the BertLatin layer are preserved, and a 'sentence_embedding' entry
    will be added.

    Adapted from https://github.com/UKPLab/sentence-transformers/blob/
    c6d13637a773e6dc5654f7ceec3ce34f48d460a3/sentence_transformers/models/
    Pooling.py
    """

    def __init__(self, mode: str = 'token'):
        super().__init__()
        self.mode = mode
        self.device = torch.device('cuda')

    def forward(self, bert_outs):
        output_vectors = []
        if self.mode == 'token':
            embeddings = bert_outs['token_embeddings']
            mask = bert_outs['token_attention_mask']
        elif self.mode == 'subtoken':
            embeddings = bert_outs['last_hidden_state']
            mask = bert_outs['attention_mask']
        else:
            raise ValueError(f'Unknown mode: {self.mode}')
        input_mask_expanded = mask.unsqueeze(-1).expand(
            embeddings.size()).float()
        sum_embeddings = torch.sum(embeddings * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        output_vectors.append(sum_embeddings / sum_mask)
        output_vector = torch.cat(output_vectors, 1)
        bert_outs['sentence_embedding'] = output_vector.to(self.device)
        return bert_outs


class SBertLatin(nn.Sequential):
    """Encapsulates BertLatin and pooling to create sentence embedding"""

    def __init__(self, bertPath, mode='token'):
        self.modules = OrderedDict([
            (str(idx), module)
            for idx, module in enumerate([BertLatin(bertPath),
                                          Pooling(mode)])
        ])
        super().__init__(self.modules)
        self.mode = mode

    def save_embedder(self, outpath):
        """Save Bert model to specified outpath"""
        self.modules['0'].save_embedder(outpath)


class ReinitSBertLatin(nn.Sequential):
    """Encapsulates BertLatin and pooling to create sentence embedding

    Also re-intializes some of the final layers of BertLatin
    """

    def __init__(self, bertPath, reinit, mode='token'):
        bl = BertLatin(bertPath)
        with torch.no_grad():
            torch.nn.init.normal_(bl.bert.pooler.dense.weight, std=0.02**2)
            for layer in bl.bert.encoder.layer[-reinit:]:
                for component in layer.modules():
                    if isinstance(component, torch.nn.Linear):
                        torch.nn.init.normal_(component.weight, std=0.02**2)
        self.modules = OrderedDict([
            (str(idx), module)
            for idx, module in enumerate([bl, Pooling(mode)])
        ])
        super().__init__(self.modules)
        self.mode = mode

    def save_embedder(self, outpath):
        """Save Bert model to specified outpath"""
        self.modules['0'].save_embedder(outpath)


class CLSExtractor(nn.Module):
    """Uses CLS as sentence embedding

    This layer is expected to come directly after a BertLatin layer. Outputs
    from the BertLatin layer are preserved, and a 'sentence_embedding' entry
    will be added.
    """

    def __init__(self, mode: str = 'token'):
        super().__init__()
        self.mode = mode
        self.device = torch.device('cuda')

    def forward(self, bert_outs):
        output_vectors = []
        if self.mode == 'token':
            embeddings = bert_outs['token_embeddings']
        elif self.mode == 'subtoken':
            embeddings = bert_outs['last_hidden_state']
        else:
            raise ValueError(f'Unknown mode: {self.mode}')
        mask = torch.zeros_like(embeddings)
        mask[:, 0, :] = 1
        sum_embeddings = torch.sum(embeddings * mask, 1)
        output_vectors.append(sum_embeddings)
        output_vector = torch.cat(output_vectors, 1)
        bert_outs['sentence_embedding'] = output_vector.to(self.device)
        return bert_outs


class CLSBertLatin(nn.Sequential):
    """Encapsulates BertLatin and CLSExtractor to create sentence embedding"""

    def __init__(self, bertPath, mode='token'):
        self.modules = OrderedDict([
            (str(idx), module) for idx, module in enumerate(
                [BertLatin(bertPath), CLSExtractor(mode)])
        ])
        super().__init__(self.modules)
        self.mode = mode

    def save_embedder(self, outpath):
        """Save Bert model to specified outpath"""
        self.modules['0'].save_embedder(outpath)


class CosineSimilarityLoss(nn.Module):
    """Train model to optimize cosine similarity

    Adapted from https://github.com/UKPLab/sentence-transformers/blob/
    14b3561193a059ebe9438af3c24944e3b22f6b84/sentence_transformers/losses/
    CosineSimilarityLoss.py
    """

    def __init__(self,
                 model: SBertLatin,
                 loss_fct=nn.MSELoss(),
                 cos_score_transformation=nn.Identity()):
        super(CosineSimilarityLoss, self).__init__()
        self.model = model
        self.loss_fct = loss_fct
        self.cos_score_transformation = cos_score_transformation

    def forward(self, sentences_features: Iterable[Dict[str, Tensor]],
                labels: Tensor):
        embeddings = [
            self.model(sentence_feature)['sentence_embedding']
            for sentence_feature in sentences_features
        ]
        output = self.cos_score_transformation(
            torch.cosine_similarity(embeddings[0], embeddings[1]))
        return self.loss_fct(output, labels.view(-1))
