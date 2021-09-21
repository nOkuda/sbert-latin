"""Latin tokenization with span information"""
import re
from collections import namedtuple
from typing import List, Tuple

from cltk.tokenize.latin.params import latin_exceptions
from cltk.tokenize.latin.sentence import SentenceTokenizer


class LatinTokenizer():

    def __init__(self, encoder):
        self.vocab = {}
        self.reverseVocab = {}
        self.encoder = encoder

        self.vocab["[PAD]"] = 0
        self.vocab["[UNK]"] = 1
        self.vocab["[CLS]"] = 2
        self.vocab["[SEP]"] = 3
        self.vocab["[MASK]"] = 4

        for key in self.encoder._subtoken_string_to_id:
            self.vocab[key] = self.encoder._subtoken_string_to_id[key] + 5
            self.reverseVocab[self.encoder._subtoken_string_to_id[key] +
                              5] = key

    def convert_tokens_to_ids(self, tokens):
        wp_tokens = []
        for token in tokens:
            if token == "[PAD]":
                wp_tokens.append(0)
            elif token == "[UNK]":
                wp_tokens.append(1)
            elif token == "[CLS]":
                wp_tokens.append(2)
            elif token == "[SEP]":
                wp_tokens.append(3)
            elif token == "[MASK]":
                wp_tokens.append(4)

            else:
                wp_tokens.append(self.vocab[token])

        return wp_tokens

    def tokenize(self, text):
        tokens = text.split(" ")
        wp_tokens = []
        for token in tokens:

            if token in {"[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"}:
                wp_tokens.append(token)
            else:

                wp_toks = self.encoder.encode(token)

                for wp in wp_toks:
                    wp_tokens.append(self.reverseVocab[wp + 5])

        return wp_tokens


class LatinSentenceTokenizer:
    """Tokenizes Latin text into sentences

    Relies on the CLTK Latin SentenceTokenizer giving access to a trained
    instance of nltk.tokenize.punkt.PunktSentenceTokenizer. If they change
    their implementation, this will break.
    """

    def __init__(self):
        self._toker = SentenceTokenizer().model

    def tokenize(self, text: str) -> List[str]:
        """Tokenizes text into sentences"""
        return self._toker.tokenize(text)

    def tokenize_with_spans(
            self, text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
        """Tokenizes text into sentences with associated spans"""
        spans = [span for span in self._toker.span_tokenize(text)]
        sentences = [text[span[0]:span[1]] for span in spans]
        return sentences, spans


class LatinWordTokenizer:
    """Tokenizes Latin sentences into words

    Heavily based off of ctlk.tokenize.latin.word.WordTokenizer and
    nltk.tokenize.punkt.PunktLanguageVars

    CLTK license: MIT License, (c) 2013 Classical Language Toolkit
    NLTK license: Apache License 2.0, (c) 2001--2021 NLTK Project
    """

    ENCLITICS = ['que', 'n', 'ne', 'ue', 've', 'st']

    EXCEPTIONS = list(set(ENCLITICS + latin_exceptions))

    def __init__(self):
        self.language = 'latin'

    def tokenize_with_spans(
            self, sentence: str) -> Tuple[List[str], List[Tuple[int, int]]]:
        """Tokenizes sentence into substrings with associated spans"""
        final_tokens = []
        final_spans = []
        for match in self._word_tokenizer_re().finditer(sentence):
            if match:
                for i, token in enumerate(match.groups()):
                    if token:
                        # group 0 is the entire match; subsequent groups are
                        # matched subgroups
                        cur_span = match.span(i + 1)
                        tokens, spans = self._handle_special_tokens(
                            token, cur_span)
                        final_tokens += tokens
                        final_spans += spans
        # handle period at end of sentence, if it exists
        if final_tokens[-1].endswith('.'):
            final_word = final_tokens[-1][:-1]
            final_word_start = final_spans[-1][0]
            final_word_end = final_spans[-1][1] - 1
            del final_tokens[-1]
            del final_spans[-1]
            tokens, spans = self._handle_special_tokens(
                final_word, (final_word_start, final_word_end))
            final_tokens += tokens + ['.']
            final_spans += spans + [(final_word_end, final_word_end + 1)]
        return final_tokens, final_spans

    _re_word_start = r"[^\(\"\`{\[:;&\#\*@\)}\]\-,]"
    """Excludes some characters from starting word tokens"""

    _re_non_word_chars = r"(?:[?!)\";}\]\*:@\'\({\[])".replace("'", '')
    """Characters that cannot appear within words"""

    _re_multi_char_punct = r"(?:\-{2,}|\.{2,}|(?:\.\s){2,}\.)"
    """Hyphen and ellipsis are multi-character punctuation"""

    _word_tokenize_fmt = r"""(
        %(MultiChar)s
        |
        (?=%(WordStart)s)\S+?  # Accept word characters until end is found
        (?= # Sequences marking a word's end
            \s|                                 # White-space
            $|                                  # End-of-string
            %(NonWord)s|%(MultiChar)s|          # Punctuation
            ,(?=$|\s|%(NonWord)s|%(MultiChar)s) # Comma if at end of word
        )
        |
        \S
    )"""
    """Format of a regular expression to split punctuation from words,
    excluding period."""

    def _word_tokenizer_re(self):
        """Compiles and returns a regular expression for word tokenization"""
        try:
            return self._re_word_tokenizer
        except AttributeError:
            self._re_word_tokenizer = re.compile(
                self._word_tokenize_fmt % {
                    "NonWord": self._re_non_word_chars,
                    "MultiChar": self._re_multi_char_punct,
                    "WordStart": self._re_word_start,
                },
                re.UNICODE | re.VERBOSE,
            )
            return self._re_word_tokenizer

    def _handle_special_tokens(self, token, cur_span):
        span_start, span_end = cur_span
        if token in latin_replacements:
            tokens, span_offsets = latin_replacements[token]
            new_spans = [(span_start + offset[0], span_start + offset[1])
                         for offset in span_offsets]
            return tokens, new_spans
        if token.lower() not in self.EXCEPTIONS:
            for enclitic in self.ENCLITICS:
                if token.endswith(enclitic):
                    if enclitic == 'n':
                        final_word = token[:-len(enclitic)]
                        final_word_end = span_start + len(final_word)
                        return [final_word,
                                '-ne'], [(span_start, final_word_end),
                                         (final_word_end, span_end)]
                    elif enclitic == 'st':
                        if token.endswith('ust'):
                            final_word = token[:-len(enclitic) + 1]
                            final_word_end = span_start + len(final_word)
                            return [final_word,
                                    'est'], [(span_start, final_word_end),
                                             (final_word_end, span_end)]
                        else:
                            final_word = token[:-len(enclitic)]
                            final_word_end = span_start + len(final_word)
                            return [final_word,
                                    'est'], [(span_start, final_word_end),
                                             (final_word_end, span_end)]
                    else:
                        final_word = token[:-len(enclitic)]
                        final_word_end = span_start + len(final_word)
                        return [final_word,
                                '-' + enclitic], [(span_start, final_word_end),
                                                  (final_word_end, span_end)]
        # otherwise, no need to do anything special to this token and span
        return [token], [cur_span]


ReplacementInfo = namedtuple('ReplacementInfo', ['tokens', 'span_offsets'])

latin_replacements = {
    'mecum':
    ReplacementInfo(tokens=['cum', 'me'], span_offsets=[(2, 5), (0, 2)]),
    'tecum':
    ReplacementInfo(tokens=['cum', 'te'], span_offsets=[(2, 5), (0, 2)]),
    'secum':
    ReplacementInfo(tokens=['cum', 'se'], span_offsets=[(2, 5), (0, 2)]),
    'nobiscum':
    ReplacementInfo(tokens=['cum', 'nobis'], span_offsets=[(5, 8), (0, 5)]),
    'vobiscum':
    ReplacementInfo(tokens=['cum', 'vobis'], span_offsets=[(5, 8), (0, 5)]),
    'uobiscum':
    ReplacementInfo(tokens=['cum', 'uobis'], span_offsets=[(5, 8), (0, 5)]),
    'quocum':
    ReplacementInfo(tokens=['cum', 'quo'], span_offsets=[(3, 6), (0, 3)]),
    'quacum':
    ReplacementInfo(tokens=['cum', 'qua'], span_offsets=[(3, 6), (0, 3)]),
    'quicum':
    ReplacementInfo(tokens=['cum', 'qui'], span_offsets=[(3, 6), (0, 3)]),
    'quibuscum':
    ReplacementInfo(tokens=['cum', 'quibus'], span_offsets=[(6, 9), (0, 6)]),
    'sodes':
    ReplacementInfo(tokens=['si', 'audes'], span_offsets=[(0, 1), (1, 5)]),
    'satin':
    ReplacementInfo(tokens=['satis', 'ne'], span_offsets=[(0, 4), (4, 5)]),
    'scin':
    ReplacementInfo(tokens=['scis', 'ne'], span_offsets=[(0, 3), (3, 4)]),
    'sultis':
    ReplacementInfo(tokens=['si', 'vultis'], span_offsets=[(0, 1), (1, 6)]),
    'similist':
    ReplacementInfo(tokens=['similis', 'est'], span_offsets=[(0, 6), (6, 8)]),
    'qualist':
    ReplacementInfo(tokens=['qualis', 'est'], span_offsets=[(0, 5), (5, 7)])
}
