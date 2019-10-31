from dataclasses import dataclass, field

from coli.basic_tools.dataclass_argparse import argfield
from coli.torch_extra.layers import ContextualUnits
from coli.torch_extra.parser_base import SimpleParser
from coli.torch_extra.sentence import SentenceEmbeddings


@dataclass
class TaggerHParams(SimpleParser.HParams):
    train_iters: "Count of training step" = 10000
    evaluate_every: int = 100

    dims_hidden: "dims of hidden layers" = argfield(default_factory=lambda: [100],
                                                    nargs="+")
    mlp_dropout: float = 0.2

    # NOTE: if both chars and crf, only 1.6x slower on GPU
    use_crf: "if crf, training is 1.7x slower on CPU" = True
    word_threshold: int = 30

    sentence_embedding: SentenceEmbeddings.Options = field(
        default_factory=SentenceEmbeddings.Options)
    contextual: ContextualUnits.Options = field(
        default_factory=ContextualUnits.Options)

    @classmethod
    def get_default(cls):
        default_tagger_hparams = cls()
        default_tagger_hparams.contextual.lstm_options.num_layers = 1
        default_tagger_hparams.contextual.lstm_options.recurrent_keep_prob = 1.0
        default_tagger_hparams.sentence_embedding.dim_postag = 0
        return default_tagger_hparams
