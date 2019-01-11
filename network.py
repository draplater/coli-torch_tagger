import os

import torch
from torch.nn import Module, Embedding, LayerNorm, LeakyReLU

from coli.basic_tools.common_utils import AttrDict, NoPickle
from coli.torch_extra.elmo_manager import get_elmo
from coli.torch_extra.utils import cross_entropy_nd
from .config import TaggerOptions
from coli.torch_extra.layers import get_external_embedding, CharacterEmbedding, ContextualUnits, create_mlp
from .crf import CRF


class SentenceEmbeddings(Module):
    def __init__(self,
                 hparams,
                 statistics,
                 external_word_embedding_loader=None,
                 bilm_path=None):

        super().__init__()
        self.hparams = hparams

        # embedding
        self.word_embeddings = Embedding(
            len(statistics.words_lite), hparams.dim_word, padding_idx=0)
        if hparams.dim_postag != 0:
            self.pos_embeddings = Embedding(
                len(statistics.postags), hparams.dim_postag, padding_idx=0)
        if external_word_embedding_loader is not None:
            self.pretrained_embeddings = NoPickle(get_external_embedding(external_word_embedding_loader))
        else:
            self.pretrained_embeddings = None
        total_input_dim = hparams.dim_word + hparams.dim_postag

        if bilm_path is not None:
            self.load_bilm(bilm_path, False)
            total_input_dim += self.bilm.get_output_dim()
            self.character_lookup = None
        elif hparams.dim_char > 0:
            self.bilm = None
            self.character_lookup = Embedding(len(statistics.characters), hparams.dim_char)
            self.char_embeded = CharacterEmbedding.get(hparams.character_embedding, input_size=hparams.dim_char)
            total_input_dim += hparams.dim_char
        else:
            self.bilm = None
            self.character_lookup = self.char_embeded = None

        self.total_input_dim = total_input_dim

        self.input_layer_norm = LayerNorm(total_input_dim) \
            if hparams.input_layer_norm else None

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.word_embeddings.weight.data)
        if self.hparams.dim_postag != 0:
            torch.nn.init.xavier_normal_(self.pos_embeddings.weight.data)
        if self.character_lookup is not None:
            torch.nn.init.xavier_normal_(self.character_lookup.weight.data)

    def load_bilm(self, bilm_path, gpu):
        self.bilm = NoPickle(get_elmo(os.path.join(bilm_path, 'options.json'),
                                      os.path.join(bilm_path, 'lm_weights.hdf5'),
                                      num_output_representations=1,
                                      dropout=0.0,
                                      # TODO: fix vacab cache (how to bypass UNK??)
                                      # vocab_to_cache=statistics.words_lite.int_to_word
                                      ))
        if gpu:
            self.bilm.cuda()
        if not getattr(self, "scalar_mix", None):
            self.scalar_mix = self.bilm.scalar_mix_0
        else:
            self.bilm.scalar_mix_0 = self.scalar_mix

    def forward(self, inputs):
        # input embedding
        # word
        words = inputs.words
        word_embeded = self.word_embeddings(words)

        if self.pretrained_embeddings is not None:
            pretrained_word_embeded = self.pretrained_embeddings(inputs.words_pretrained.to(device))
            word_embeded += pretrained_word_embeded

        # pos
        if self.hparams.dim_postag != 0:
            pos_embeded = self.pos_embeddings(inputs.postags)
        else:
            pos_embeded = None

        # character
        # batch_size, bucket_size, word_length, embedding_dims
        if self.bilm is not None:
            word_embeded_by_char = self.bilm(inputs.bilm_chars,
                                             # inputs.words
                                             )["elmo_representations"][0]
        elif self.hparams.dim_char:
            # use character embedding instead
            # batch_size, bucket_size, word_length, embedding_dims
            char_embeded_4d = self.character_lookup(inputs.chars)
            word_embeded_by_char = self.char_embeded(inputs.word_lengths,
                                                     char_embeded_4d)
        else:
            word_embeded_by_char = None

        total_input_embeded = torch.cat(
            list(filter(lambda x: x is not None,
                        [word_embeded, pos_embeded, word_embeded_by_char])), -1)
        if self.input_layer_norm is not None:
            total_input_embeded = self.input_layer_norm(total_input_embeded)

        return total_input_embeded


class TaggerNetwork(Module):
    """"""

    def __init__(self, hparams: TaggerOptions,
                 statistics, external_word_embedding_loader=None,
                 bilm_path=None,
                 target="labels"):
        super().__init__()
        self.hparams = hparams
        self.target = target
        self.label_count = len(getattr(statistics, target))
        self.embeddings = SentenceEmbeddings(hparams, statistics, external_word_embedding_loader, bilm_path)

        # RNN
        self.rnn = ContextualUnits.get(hparams.contextual,
                                       input_size=self.embeddings.total_input_dim)

        self.projection = create_mlp(self.rnn.output_dim,
                                     self.label_count,
                                     hidden_dims=self.hparams.dims_hidden,
                                     layer_norm=True,
                                     last_bias=True,
                                     activation=lambda: LeakyReLU(0.1))

        if self.hparams.use_crf:
            self.crf_unit = CRF(self.label_count)

        self.reset_parameters()

    def load_bilm(self, bilm_path, gpu):
        self.embeddings.load_bilm(bilm_path, gpu)

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
