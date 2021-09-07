"""Mapping between strings and loci, according to .tess files

The main class exposed by this module is `TessFile`. The main function exposed
by this module is `read_tessfile`, which constructs an instance of `TessFile`.
"""
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, TextIO, Tuple

import fuzzywuzzy.process
import numpy as np

from sbert_latin.latinbert import LatinBERT
from sbert_latin.tokenize import LatinSentenceTokenizer


class TessFileToken:
    """Representation of a token in a .tess file

    Attributes
    ----------
    token : str
    line_index : int
        index associated with TessFileLine from which this token comes
    sentence_index : int
        index associated with TessFileSentence from which this token comes
    line_span : Tuple[int, int]
        span of indices into line from where this token comes
    sentence_span : Tuple[int, int]
        span of indices into sentence from where this token comes
    """

    def __init__(self, token: str, line_index: int, sentence_index: int,
                 line_span: Tuple[int, int], sentence_span: Tuple[int, int]):
        self.token = token
        self.line_index = line_index
        self.sentence_index = sentence_index
        self.line_span = line_span
        self.sentence_span = sentence_span


class TessFileTokenData:
    """Token database for a .tess file

    Attributes
    ----------
    tokens : List[str]
        Tokens extracted from the .tess file that were used to compute the
        LatinBERT embeddings
    embeddings : 2d np.ndarray
        The LatinBERT embedding of ``tokens[x]`` is ``embeddings[x]``
    line_indices : 1d np.ndarray
        ``line_indices[x]`` is the index to an instance of TessFileLine from
        which ``tokens[x]`` comes
    sentence_indices : 1d np.ndarray
        ``sentence_indices[x]`` is the index to an instance of TessFileSentence
        from which ``tokens[x]`` comes
    line_spans : 2d np.ndarray
        ``line_spans[x][0]`` is the starting position and ``line_spans[x][1]``
        is one past the ending position of ``tokens[x]`` of the text portion of
        the TessFileLine associated with the index at ``line_indices[x]``; the
        starting and ending positions described match the convention of using
        spans for slices
    sentence_spans : 2d np.ndarray
        ``sentence_spans[x][0]`` is the starting position and
        ``sentence_spans[x][1]`` is one past the ending position of
        ``tokens[x]`` of the text portion of the TessFileLine associated with
        the index at ``sentence_indices[x]``; the starting and ending positions
        described match the convention of using spans for slices
    """

    def __init__(self, tokens: List[str], embeddings: np.ndarray,
                 line_indices: np.ndarray, sentence_indices: np.ndarray,
                 line_spans: np.ndarray, sentence_spans: np.ndarray):
        self.tokens = tokens
        self.embeddings = embeddings
        self.line_indices = line_indices
        self.sentence_indices = sentence_indices
        self.line_spans = line_spans
        self.sentence_spans = sentence_spans

    def get_token(self, index: int) -> TessFileToken:
        line_span = self.line_spans[index]
        sentence_span = self.sentence_spans[index]
        return TessFileToken(token=self.tokens[index],
                             line_index=self.line_indices[index],
                             sentence_index=self.sentence_indices[index],
                             line_span=(int(line_span[0]), int(line_span[1])),
                             sentence_span=(int(sentence_span[0]),
                                            int(sentence_span[1])))

    def get_token_frequencies(self) -> np.array:
        """

        Frequencies are calculated by work alone, not by corpus.

        The ith item of the returned np.array is the number of times the word
        corresponding to ith token in self.tokens appears in the work, divided
        by the number of tokens in self.tokens.
        """
        counter = Counter(self.tokens)
        total = len(self.tokens)
        inv_freqs = {token: count / total for token, count in counter.items()}
        return np.array([inv_freqs[a] for a in self.tokens])


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

    def get_highlighted_with(self, token: TessFileToken):
        text = self.text
        span = token.line_span
        parts = (text[:span[0]], text[span[0]:span[1]], text[span[1]:])
        highlighted_text = '***'.join(parts)
        return self.tag + '\t' + highlighted_text


def read_tessfile_lines(filepath: Path) -> List[TessFileLine]:
    """Produces TessFileLines from the .tess file specified at filepath

    Invalid lines are ignored
    """
    with filepath.expanduser().resolve().open() as ifh:
        return _extract_tessfile_lines(ifh)


VALID_LINE_RE = re.compile(r'^<(?P<tag>.+)>\s+(?P<text>.+)$')


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

    def get_highlighted_with(self, token: TessFileToken):
        text = self.sentence
        span = token.sentence_span
        parts = (text[:span[0]], text[span[0]:span[1]], text[span[1]:])
        highlighted_text = '***'.join(parts)
        return highlighted_text

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


class TessFile:
    """Representation of a .tess file

    Attributes
    ----------
    filepath : Path
        .tess file location
    lines : List[TessFileLine]
    sentences : List[TessFileSentence]
    tokendata : TessFileTokenData
    embeddings : 2d np.array
    """

    def __init__(self, filepath: Path, lines: List[TessFileLine],
                 sentences: List[TessFileSentence],
                 tokendata: TessFileTokenData):
        self.filepath = filepath
        self.lines = lines
        self.sentences = sentences
        self.tokendata = tokendata

    @property
    def embeddings(self):
        return self.tokendata.embeddings

    def get_token(self, index: int) -> TessFileToken:
        return self.tokendata.get_token(index)

    def get_line_for(self, token: TessFileToken) -> TessFileLine:
        return self.lines[token.line_index]

    def get_sentence_for(self, token: TessFileToken) -> TessFileSentence:
        return self.sentences[token.sentence_index]

    def tokens_iter(self) -> Iterable[TessFileToken]:
        for i in range(len(self.tokendata.tokens)):
            yield self.get_token(i)

    def get_token_frequencies(self) -> np.array:
        return self.tokendata.get_token_frequencies()


def read_tessfile(filepath: Path, sent_toker: LatinSentenceTokenizer,
                  bert_model: LatinBERT) -> TessFile:
    """Get LatinBERT emeddings and loci for each token in a .tess file"""
    lines = read_tessfile_lines(filepath)
    sentences = extract_sentences(sent_toker, lines)
    return _build_tessfile(filepath, lines, sentences, bert_model)


def _build_tessfile(filepath: Path, lines: List[TessFileLine],
                    sentences: List[TessFileSentence], bert_model: LatinBERT):
    berts, spans_in_sents = bert_model.get_berts_and_spans(
        [sent.sentence for sent in sentences])
    tokens = []
    embeddings = []
    line_indices = []
    sentence_indices = []
    line_spans = []
    sentence_spans = []
    for sent_i, (bert, sent, spans_in_sent) in enumerate(
            zip(berts, sentences, spans_in_sents)):
        newline_positions = sent.get_newline_positions()
        for (token, embedding), span_in_sent in zip(bert, spans_in_sent):
            tokens.append(token)
            embeddings.append(embedding)
            line_index, span_in_line = sent.get_line_index_and_span(
                span_in_sent, newline_positions)
            line_indices.append(line_index)
            sentence_indices.append(sent_i)
            line_spans.append(span_in_line)
            sentence_spans.append(span_in_sent)
    tokendata = TessFileTokenData(tokens=tokens,
                                  embeddings=np.array(embeddings),
                                  line_indices=np.array(line_indices),
                                  sentence_indices=np.array(sentence_indices),
                                  line_spans=np.array(line_spans),
                                  sentence_spans=np.array(sentence_spans))
    return TessFile(filepath=filepath,
                    lines=lines,
                    sentences=sentences,
                    tokendata=tokendata)


def read_lines_and_sentences(
        filepath: Path) -> Tuple[List[TessFileLine], List[TessFileSentence]]:
    """Gets lines and sentences from a .tess file"""
    lines = read_tessfile_lines(filepath)
    sentences = extract_sentences(LatinSentenceTokenizer(), lines)
    return lines, sentences


class TagKeeper:
    """Keeps track of tag associations with sentences"""

    def __init__(self, lines: List[TessFileLine],
                 sentences: List[TessFileSentence]):
        self.lines = lines
        self.sentences = sentences
        self.line_idx_to_sentences = {}
        for sent in sentences:
            start, end = sent.line_span
            for line_idx in range(start, end):
                if line_idx not in self.line_idx_to_sentences:
                    self.line_idx_to_sentences[line_idx] = [sent]
                else:
                    self.line_idx_to_sentences[line_idx].append(sent)
        # assumed that tag searching will be by numbering only
        self.tag_to_line_idx = {
            line.tag.strip().split()[-1]: i
            for i, line in enumerate(lines)
        }

    def get_sentence(self, tag: str, snippet: str) -> str:
        """Get sentence string associated with the given tag"""
        if tag not in self.tag_to_line_idx:
            raise ValueError(f'Cannot find {tag}')
        sents_str = [
            s.sentence
            for s in self.line_idx_to_sentences[self.tag_to_line_idx[tag]]
        ]
        if len(sents_str) == 1:
            return sents_str[0]
        return fuzzywuzzy.process.extractOne(snippet, sents_str)[0]


def build_tagkeeper(filepath: Path) -> TagKeeper:
    lines, sentences = read_lines_and_sentences(filepath)
    return TagKeeper(lines, sentences)
