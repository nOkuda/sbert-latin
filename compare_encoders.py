import re
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, TextIO, Tuple

import numpy as np
from cltk.tokenize.latin.params import latin_exceptions
from cltk.tokenize.latin.sentence import SentenceTokenizer
from tensor2tensor.data_generators import text_encoder
from transformers import BertTokenizer

VALID_LINE_RE = re.compile(r'^<(?P<tag>.+)>\s+(?P<text>.+)$')


def _main():
    lat_sent_toker = LatinSentenceTokenizer()
    subword = text_encoder.SubwordTextEncoder(
        '/home/okuda/Code/latin-bert/models/subword_tokenizer_latin/'
        'latin.subword.encoder')
    orig = LatinTokenizer(subword)
    other = BertTokenizer(
        '/home/okuda/Code/latin-bert/models/latin_bert/vocab.txt',
        do_basic_tokenize=False,
        # additional_special_tokens=['---', '--', ".'", "!'", "?'"],
    )
    other.wordpiece_tokenizer = orig
    tess_texts_dir = Path('~/Code/tesserae/texts/la/').expanduser().resolve()
    filepaths = [
        tess_texts_dir / 'vergil.aeneid.tess',
        tess_texts_dir / 'lucan.bellum_civile.tess',
        tess_texts_dir / 'ovid.metamorphoses.tess',
        tess_texts_dir / 'statius.thebaid.tess',
        tess_texts_dir / 'valerius_flaccus.argonautica.tess',
    ]
    for fpath in filepaths:
        with fpath.open() as ifh:
            lines = _extract_tessfile_lines(ifh)
        sentences = extract_sentences(lat_sent_toker, lines)
        sents_toks, _ = convert_to_toks_and_spans(
            [s.sentence for s in sentences])
        for sent, sent_toks in zip(sentences, sents_toks):
            # TODO figure out how to use the tokenizer/encoder
            orig_tokens = []
            for word_tok in sent_toks:
                orig_tokens.extend(orig.tokenize(word_tok))
            other_tokens = []
            for word_tok in sent_toks:
                other_tokens.extend(other.tokenize(word_tok))
            if not np.all(orig_tokens == other_tokens):
                print('####')
                print(lines[sent.line_span[0]].tag)
                print(sent.sentence)
                print(orig_tokens)
                print(other_tokens)


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


class TessFileLine:
    """Representation of a single line in a .tess file"""

    def __init__(self, file_line_index: int, raw_text: str, tag: str,
                 text: str, tag_span: Tuple[int, int], text_span: Tuple[int,
                                                                        int]):
        # line_index refers to original file's line index
        self.file_line_index = file_line_index
        self.raw_text = raw_text
        self.tag = tag
        self.text = text
        # spans index into raw_text
        self.tag_span = tag_span
        self.text_span = text_span


def _extract_tessfile_lines(ifh: TextIO) -> List[TessFileLine]:
    """Subroutine for easier unit testing"""
    results = []
    for i, raw_text in enumerate(ifh):
        match = VALID_LINE_RE.match(raw_text)
        if match:
            results.append(
                TessFileLine(file_line_index=i,
                             raw_text=raw_text,
                             tag=match.group('tag'),
                             text=match.group('text'),
                             tag_span=match.span('tag'),
                             text_span=match.span('text')))
    return results


NEWLINE_RE = re.compile('\n')


@dataclass
class TessFileSentence:
    """Sentence of text from a .tess file

    The span refers to the list of TessFileLines used to create a given
    instance of a TessFileSentence
    """
    # text of sentence
    sentence: str
    # start and one past end indices into array of lines
    line_span: Tuple[int, int]
    # position on the first line in which this sentence begins
    initial_pos: int

    def get_line_index_and_span(
            self,
            span: Tuple[int, int],
            newline_positions=None) -> Tuple[int, Tuple[int, int]]:
        """Finds line index and span associated with ``span``

        Can optimize by precomputing the newline positions for a given sentence
        and passing that in

        Returns
        -------
        line_index : int
            Index into the original list of TessFileLines used to compute this
            instance of TessFileSentence
        span_in_line : Tuple[int, int]
            Slice information of ``span`` with respect to the TessFileLine
            associated with ``line_index``
        """
        if newline_positions is None:
            newline_positions = self.get_newline_positions()
        if not newline_positions:
            # no newlines in sentence
            line_index = self.line_span[0]
            span_in_line = (span[0] + self.initial_pos,
                            span[1] + self.initial_pos)
            return line_index, span_in_line
        prev_newline_pos = 0
        for i, cur_newline_pos in enumerate(newline_positions):
            if span[0] < cur_newline_pos:
                line_index = self.line_span[0] + i
                if i > 0:
                    # the extra 1 subtracted accounts for the fact that the
                    # newline position is 1 before the first text position in
                    # the line
                    span_in_line = (span[0] - prev_newline_pos - 1,
                                    span[1] - prev_newline_pos - 1)
                    return line_index, span_in_line
                else:
                    # ``span`` is on the first line, so account for initial
                    # position of sentence in line
                    span_in_line = (span[0] + self.initial_pos,
                                    span[1] + self.initial_pos)
                    return line_index, span_in_line
            prev_newline_pos = cur_newline_pos
        # if we get here, span must be on the last line
        line_index = self.line_span[1] - 1
        span_in_line = (span[0] - newline_positions[-1] - 1,
                        span[1] - newline_positions[-1] - 1)
        return line_index, span_in_line

    def get_newline_positions(self) -> List[int]:
        return [m.start() for m in NEWLINE_RE.finditer(self.sentence)]


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


def extract_sentences(sent_toker: LatinSentenceTokenizer,
                      lines: List[TessFileLine]) -> List[TessFileSentence]:
    full_text = '\n'.join([line.text for line in lines])
    line_starts = _compute_line_starts(lines)
    sentences, spans = sent_toker.tokenize_with_spans(full_text)
    result = []
    line_span_start = 0
    for sent, (sent_start, _) in zip(sentences, spans):
        assert line_span_start < len(lines)
        newline_count = sent.count('\n')
        line_span_end = line_span_start + newline_count + 1
        if line_span_end > len(lines):
            # don't go past the end
            line_span_end = len(lines)
        initial_pos = sent_start - line_starts[line_span_start]
        result.append(
            TessFileSentence(sentence=sent,
                             line_span=(line_span_start, line_span_end),
                             initial_pos=initial_pos))
        line_span_start += newline_count
        if line_span_start < len(lines) and _sentence_ends_line(
                result[-1], lines[line_span_start]):
            # account for sentence ending at end of a line (which resulted in a
            # newline being taken away by the sentence tokenizer)
            line_span_start += 1
    return result


def _compute_line_starts(lines: List[TessFileLine]) -> List[int]:
    """Indices into a string containing newline-separated texts of ``lines``"""
    # the first line always starts at index 0
    line_starts = [0]
    for prev_line in lines[:-1]:
        # the next line's text starts after where the previous line's text
        # ends, plus 1 to account for the newline that separates them
        line_starts.append(line_starts[-1] + len(prev_line.text) + 1)
    return line_starts


def _sentence_ends_line(sentence: TessFileSentence,
                        line: TessFileLine) -> bool:
    if sentence.sentence.endswith(line.text):
        # sentence and line end at the same place
        return True
    if line.text.endswith(sentence.sentence):
        # line ends with what is in sentence, so we know that sentence must be
        # completely within the line;
        # now we need to make sure that the sentence ends where the line ends
        return len(line.text) == sentence.initial_pos + len(sentence.sentence)

    return False


def _get_index_of_earliest_sentence_in_line(
        sentences_so_far: List[TessFileSentence], line: TessFileLine) -> int:
    last_index = len(sentences_so_far) - 1
    for i in range(last_index, 0, -1):
        if '\n' in sentences_so_far[i].sentence:
            return i
        if i < last_index:
            cur_sent_span_end = sentences_so_far[i].line_span[1]
            following_sent_span_start = sentences_so_far[i + 1].line_span[0]
            if cur_sent_span_end == following_sent_span_start:
                # these two sentences were separated by a newline
                return i
    return 0


def _last_sentences_endwith_line(sentences: List[TessFileSentence], start: int,
                                 line: TessFileLine) -> bool:
    combined = ' '.join([s.sentence for s in sentences[start:]])
    print(combined)
    print(line.text)
    return combined.endswith(line.text)


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


if __name__ == '__main__':
    _main()
