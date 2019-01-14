import dataclasses
import torch
from typing import List, Optional

from dataclasses import dataclass
from torch import Tensor

from bilm.load_vocab import BiLMVocabLoader
from coli.basic_tools.common_utils import add_slots
from coli.data_utils.dataset import SentenceFeaturesBase, PAD, UNKNOWN, START_OF_WORD, \
    END_OF_WORD
from coli.data_utils.vocab_utils import Dictionary
from coli.torch_extra.dataset import lookup_list, lookup_characters


def get_chunk_type(tok, tag_transform=lambda x: x):
    """
    Args:

    Returns:
        tuple: "B", "PER"

    """
    tag_name = tag_transform(tok)
    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[-1]
    return tag_class, tag_type


def get_chunks(seq, tag_dict=None, none_tag="O"):
    """Given a sequence of tags, group entities and their position

    Args:
        seq: [4, 4, 0, 0, ...] sequence of labels
        tags: dict["O"] = 4

    Returns:
        list of (chunk_type, chunk_start, chunk_end)

    Example:
        seq = [4, 5, 0, 3]
        tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
        result = [("PER", 0, 2), ("LOC", 3, 4)]

    """
    if tag_dict is not None:
        default = tag_dict[none_tag]
        idx_to_tag = {idx: tag for tag, idx in tag_dict.items()}
        tag_lookup = lambda x: idx_to_tag[x]
    else:
        default = "O"
        tag_lookup = lambda x: x

    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != default:
            tok_chunk_class, tok_chunk_type = get_chunk_type(
                tok, tag_lookup)
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass

    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)

    return chunks


@add_slots
@dataclass
class SentenceWithTags(object):
    words: List[str]
    labels: List[str]
    postags: List[str]

    @classmethod
    def from_file(cls, path, keep_tag=True):
        results = []
        with open(path) as f:
            sent = cls([], [], [])
            for line in f:
                line_s = line.rstrip()
                if not line_s:
                    results.append(sent)
                    sent = cls([], [], [])
                else:
                    fields = line_s.rsplit(maxsplit=2)
                    if len(fields) == 3:
                        word, tag, postag = fields
                    elif len(fields) == 2:
                        word, tag = fields
                        postag = "X"
                    else:
                        assert len(fields) == 1
                        word = " "
                        tag = fields[0]
                        postag = "X"
                    sent.words.append(word)
                    sent.labels.append(tag)
                    sent.postags.append(postag)
            if sent:
                results.append(sent)
            return results

    def to_string(self):
        return "\n".join(["\t".join([word, label, postag])
                          for word, label, postag in zip(
                self.words, self.labels, self.postags)]) + "\n\n"

    def replaced_labels(self, new_labels):
        return dataclasses.replace(self, labels=new_labels)

    @classmethod
    def internal_evaluate(cls, gold_sents: List["SentenceWithTags"],
                          system_sents: List["SentenceWithTags"],
                          log_file: str = None
                          ):
        correct_preds, total_correct, total_preds = 0., 0., 0.
        for gold_sent, system_sent in zip(gold_sents, system_sents):
            correct_chunks = set(get_chunks(gold_sent.labels))
            pred_chunks = set(get_chunks(system_sent.labels))
            correct_preds += len(correct_chunks & pred_chunks)
            total_preds += len(pred_chunks)
            total_correct += len(correct_chunks)
        p = correct_preds / total_preds if correct_preds > 0 else 0
        r = correct_preds / total_correct if correct_preds > 0 else 0
        f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
        log_str = "P: {:.2f}, R: {:.2f}, F: {:.2f}".format(p * 100, r * 100, f1 * 100)
        print(log_str)
        if log_file is not None:
            with open(log_file, "w") as f:
                print(log_str, file=f)
        return f1 * 100

    @classmethod
    def evaluate_with_external_program(cls, gold_file, system_file):
        p, r, f1 = cls.internal_evaluate(cls.from_file(gold_file),
                                         cls.from_file((system_file)))
        print("P: {:.2f}, R: {:.2f}, F: {:.2f}".format(p, r, f1))
        with open(gold_file + ".txt", "w") as f:
            print("P: {:.2f}, R: {:.2f}, F: {:.2f}".format(p, r, f1),
                  file=f)

    def __len__(self):
        return len(self.words)


class SentenceWithTagsSimple(SentenceWithTags):
    @classmethod
    def internal_evaluate(cls, gold_sents: List["SentenceWithTags"],
                          system_sents: List["SentenceWithTags"],
                          log_file: str = None
                          ):
        correct_tags = 0
        total_tags = 0
        for gold_sent, system_sent in zip(gold_sents, system_sents):
            for gold_label, system_label in zip(gold_sent.labels, system_sent.labels):
                total_tags += 1
                if gold_label == system_label:
                    correct_tags += 1
        accuracy = correct_tags / total_tags * 100
        if log_file is not None:
            with open(log_file, "w") as f:
                print("Accuracy: {:.2f}".format(accuracy), file=f)
        return accuracy

    @classmethod
    def evaluate_with_external_program(cls, gold_file, system_file):
        accuracy = cls.internal_evaluate(cls.from_file(gold_file),
                                         cls.from_file((system_file)))
        print("Accuracy: {:.2f}".format(accuracy))
        with open(gold_file + ".txt", "w") as f:
            print("Accuracy: {:.2f}".format(accuracy), file=f)


class DTMLexicalType(SentenceWithTagsSimple):
    @classmethod
    def from_file(cls, path, keep_tag=True):
        results = []
        with open(path) as f:
            sent = None
            for line in f:
                line_s = line.rstrip()
                if not line_s or line_s.startswith("#"):
                    if sent is not None:
                        results.append(sent)
                    sent = None
                else:
                    if sent is None:
                        # noinspection PyArgumentList
                        sent = cls([], [], [])
                    fields = line_s.split()
                    sent.words.append(fields[1])
                    sent.labels.append(fields[4])
                    sent.postags.append(fields[3])
        if sent is not None:
            results.append(sent)
        return results


@add_slots
@dataclass
class Statistics(object):
    words: Dictionary
    labels: Dictionary
    postags: Dictionary
    characters: Dictionary
    words_lite: Dictionary = None

    @classmethod
    def from_sentences(cls, sents: List[SentenceWithTags],
                       word_threshold: int):
        # noinspection PyArgumentList
        result = cls(
            Dictionary(),
            Dictionary(initial=(PAD, "O",)),
            Dictionary(),
            Dictionary(initial=(PAD, UNKNOWN, START_OF_WORD, END_OF_WORD))
        )
        for sent in sents:
            result.words.update(sent.words)
            result.labels.update(sent.labels)
            result.postags.update(sent.postags)
            for word in sent.words:
                result.characters.update(word)

        if word_threshold > 0:
            result.words_lite = result.words.strip_low_freq(
                min_count=word_threshold,
                ensure=("___PAD___", "___UNKNOWN___")
            )
        return result


@add_slots
@dataclass
class SentenceFeatures(SentenceFeaturesBase):
    original_idx: int
    original_obj: SentenceWithTags
    words: Tensor  # [int] (pad_length, )
    words_pretrained: Tensor  # [int] (pad_length, )
    characters: Tensor  # [int] (pad_length, word_pad_length)
    bilm_characters: Tensor
    labels: Tensor  # [int] (pad_length, )
    postags: Tensor  # [int] (pad_length, )
    sent_length: int
    char_lengths: Tensor  # [int] (pad_length, )

    int_type = torch.int64
    bilm_boundaries = False

    @classmethod
    def from_sentence_obj(cls, original_idx, sent: SentenceWithTags,
                          statistics: Statistics,
                          external_lookup,
                          padded_length, lower=True,
                          bilm_loader: Optional[BiLMVocabLoader] = None
                          ):
        sent_length = len(sent.words)
        lower_func = lambda x: x.lower() if lower else lambda x: x
        words_int = lookup_list(
            (lower_func(i) for i in sent.words), statistics.words_lite.word_to_int,
            padded_length=padded_length, default=1, dtype=cls.int_type)
        words_pretrained_int = lookup_list(
            (lower_func(i) for i in sent.words), external_lookup,
            padded_length=padded_length, default=0, dtype=cls.int_type) \
            if external_lookup is not None else None
        labels_int = lookup_list(sent.labels, statistics.labels.word_to_int,
                                 padded_length=padded_length, default=0, dtype=cls.int_type,
                                 tensor_factory=lambda shape, *, dtype: torch.full(shape, -100, dtype=dtype)
                                 )
        postags_int = lookup_list(sent.postags, statistics.postags.word_to_int,
                                  padded_length=padded_length, default=1, dtype=cls.int_type)
        char_lengths, chars_int = lookup_characters(
            sent.words, statistics.characters.word_to_int,
            padded_length, 1, dtype=cls.int_type, return_lengths=True)

        bilm_chars_padded = torch.from_numpy(bilm_loader.get_chars_input(
            sent.words, padded_length, boundaries=cls.bilm_boundaries)) \
            if bilm_loader is not None else None

        # noinspection PyArgumentList
        return cls(original_idx, sent, words_int, words_pretrained_int, chars_int, bilm_chars_padded,
                   labels_int, postags_int, sent_length, char_lengths)

    @classmethod
    def get_feed_dict(cls, pls, batch_sentences):
        # noinspection PyCallingNonCallable
        ret = {
            pls.words: torch.stack([i.words for i in batch_sentences]),
            pls.postags: torch.stack([i.postags for i in batch_sentences]),
            pls.chars: torch.stack([i.characters for i in batch_sentences]),
            pls.labels: torch.stack([i.labels for i in batch_sentences]),
            pls.sent_lengths: torch.tensor([i.sent_length for i in batch_sentences],
                                           dtype=torch.int64),
            pls.word_lengths: torch.stack([i.char_lengths for i in batch_sentences])
        }

        if batch_sentences[0].words_pretrained is not None:
            ret[pls.words_pretrained] = torch.stack([i.words_pretrained for i in batch_sentences])

        if batch_sentences[0].bilm_characters is not None:
            # noinspection PyCallingNonCallable
            ret[pls.bilm_chars] = torch.stack([i.bilm_characters for i in batch_sentences])
        return ret
