from dataclasses import dataclass

from coli.data_utils.dataset import TensorflowHParamsBase
from coli.torch_extra.layers import CharacterEmbedding, ContextualUnitsOptions


@dataclass
class TaggerOptions(TensorflowHParamsBase,
                    CharacterEmbedding.Options,
                    ContextualUnitsOptions
                    ):
    train_iters: "Count of training step" = 10000
    evaluate_every: int = 100

    # embeddings
    dim_word: "Word dims" = 100
    dim_postag: "Postag dims" = 50
    dim_hidden: "dim of hidden layer" = 100

    # NOTE: if both chars and crf, only 1.6x slower on GPU
    use_crf: "if crf, training is 1.7x slower on CPU" = True

    recurrent_keep_prob: float = 1.0

    word_threshold: int = 30

    learning_rate: float = 1e-3
    learning_rate_warmup_steps: int = 160
    step_decay_factor: float = 0.5
    step_decay_patience: int = 5

    lstm_layers: int = 1
