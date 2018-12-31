from dataclasses import dataclass, field

from coli.basic_tools.dataclass_argparse import argfield
from coli.data_utils.dataset import HParamsBase
from coli.torch_extra.layers import CharacterEmbedding, ContextualUnits, AdvancedLearningOptions


@dataclass
class TaggerOptions(HParamsBase):
    train_iters: "Count of training step" = 10000
    evaluate_every: int = 100

    # embeddings
    dim_word: "Word dims" = 100
    dim_char: "Word dims" = 100
    dim_postag: "Postag dims" = 50
    dims_hidden: "dims of hidden layers" = argfield(default_factory=lambda: [100],
                                                    nargs="+")

    # NOTE: if both chars and crf, only 1.6x slower on GPU
    use_crf: "if crf, training is 1.7x slower on CPU" = True
    word_threshold: int = 30
    input_layer_norm: bool = True

    character_embedding: CharacterEmbedding.Options = field(
        default_factory=CharacterEmbedding.Options)
    contextual: ContextualUnits.Options = field(
        default_factory=ContextualUnits.Options)
    learning: AdvancedLearningOptions = field(
        default_factory=AdvancedLearningOptions)


def get_default_tagger_options(factory=TaggerOptions):
    default_tagger_options = factory()
    default_tagger_options.contextual.lstm_options.num_layers = 1
    default_tagger_options.contextual.lstm_options.recurrent_keep_prob = 1.0
    return default_tagger_options
