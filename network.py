import torch
from torch.nn import Module, LeakyReLU

from coli.basic_tools.common_utils import AttrDict
from coli.torch_extra.sentence import SentenceEmbeddings
from coli.torch_extra.utils import cross_entropy_nd
from coli.torch_extra.layers import ContextualUnits, create_mlp
from coli.torch_tagger.config import TaggerHParams
from .crf import CRF


class TaggerNetwork(Module):
    """"""

    def __init__(self, hparams: TaggerHParams,
                 statistics, plugins=None,
                 target="labels"):
        super().__init__()
        self.hparams = hparams
        self.target = target
        self.label_count = len(getattr(statistics, target))
        self.embeddings = SentenceEmbeddings(hparams.sentence_embedding, statistics, plugins)

        # RNN
        self.rnn = ContextualUnits.get(hparams.contextual,
                                       input_size=self.embeddings.output_dim)

        self.projection = create_mlp(self.rnn.output_dim,
                                     self.label_count,
                                     hidden_dims=self.hparams.dims_hidden,
                                     dropout=hparams.mlp_dropout,
                                     layer_norm=True,
                                     last_bias=True,
                                     activation=lambda: LeakyReLU(0.1))

        if self.hparams.use_crf:
            self.crf_unit = CRF(self.label_count)

        self.reset_parameters()

    def reset_parameters(self):
        self.embeddings.reset_parameters()

    def forward(self, batch_sentences, inputs):
        total_input_embeded = self.embeddings(inputs)
        sent_lengths = inputs.sent_lengths
        contextual_output = self.rnn(total_input_embeded, sent_lengths)

        logits_3d = self.projection(contextual_output)

        total_count = sent_lengths.sum()
        answer = getattr(inputs, self.target)
        mask_2d = inputs.words.gt(0)

        if self.hparams.use_crf:
            norm_scores = self.crf_unit(logits_3d, sent_lengths)
            # strip "-100"
            # noinspection PyCallingNonCallable
            answer = torch.max(answer, torch.tensor(0, device=answer.device))
            lstm_scores = torch.gather(logits_3d, 2, answer.unsqueeze(-1)).squeeze(-1)
            lstm_scores_masked = (lstm_scores * mask_2d.float()).sum(-1)
            transition_scores = self.crf_unit.transition_score(answer, sent_lengths)
            sequence_scores = transition_scores + lstm_scores_masked
            losses = norm_scores - sequence_scores
            loss = losses.mean()
        else:
            loss = cross_entropy_nd(logits_3d, answer, reduction="mean")

        if self.training or not self.hparams.use_crf:
            labels_pred = logits_3d.argmax(dim=-1)
        else:
            _, labels_pred = self.crf_unit.viterbi_decode(logits_3d, sent_lengths)

        correct_count = torch.eq(answer, labels_pred).sum()

        return AttrDict({"labels_pred": labels_pred,
                         "contextual_output": contextual_output,
                         "answer": answer,
                         "loss": loss,
                         "word_mask": mask_2d,
                         "logits": logits_3d,
                         "total_count": total_count, "correct_count": correct_count,
                         "lengths": sent_lengths,
                         "sent_count": inputs.words.shape[0],
                         "total_input_embeded": total_input_embeded
                         })
