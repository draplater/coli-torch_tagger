import torch

from torch import nn


def sequence_mask(lens, max_len=None):
    batch_size = lens.size(0)

    if max_len is None:
        max_len = lens.max().data[0]

    ranges = torch.arange(0, max_len).long()
    ranges = ranges.unsqueeze(0).expand(batch_size, max_len)

    if lens.data.is_cuda:
        ranges = ranges.cuda()

    lens_exp = lens.unsqueeze(1).expand_as(ranges)
    mask = ranges < lens_exp

    return mask


class CRF(nn.Module):
    """
    from https://github.com/kaniblu/pytorch-bilstmcrf/blob/master/model.py
    """
    def __init__(self, vocab_size):
        super(CRF, self).__init__()

        self.vocab_size = vocab_size
        self.n_labels = n_labels = vocab_size + 2
        self.start_idx = n_labels - 2
        self.stop_idx = n_labels - 1
        self.transitions = nn.Parameter(torch.randn(n_labels, n_labels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.transitions.data)

    def pad_logits(self, logits):
        """
        :param logits: [batch_size, seq_len, vocab_size] FloatTensor
        :return: [batch_size, seq_len, vocab_size + 2] FloatTensor
        """
        return torch.nn.functional.pad(
                        logits,
                        [0, 2,
                         0, 0,
                         0, 0
                         ])

    def forward(self, logits, lens):
        """
        Arguments:
            logits: [batch_size, seq_len, vocab_size] FloatTensor
            lens: [batch_size] LongTensor
        """
        logits = self.pad_logits(logits)
        batch_size, seq_len, n_labels = logits.size()
        alpha = logits.data.new(batch_size, self.n_labels).fill_(-10000)
        alpha[:, self.start_idx] = 0
        c_lens = lens.clone()

        logits_t = logits.transpose(1, 0)
        for logit in logits_t:
            logit_exp = logit.unsqueeze(-1).expand(batch_size,
                                                   *self.transitions.size())
            alpha_exp = alpha.unsqueeze(1).expand(batch_size,
                                                  *self.transitions.size())
            trans_exp = self.transitions.unsqueeze(0).expand_as(alpha_exp)
            mat = trans_exp + alpha_exp + logit_exp
            alpha_nxt = torch.logsumexp(mat, 2).squeeze(-1)

            mask = (c_lens > 0).float().unsqueeze(-1).expand_as(alpha)
            alpha = mask * alpha_nxt + (1 - mask) * alpha
            c_lens = c_lens - 1

        alpha = alpha + self.transitions[self.stop_idx].unsqueeze(0).expand_as(alpha)
        norm = torch.logsumexp(alpha, 1).squeeze(-1)

        return norm

    def viterbi_decode(self, logits, lens):
        """Borrowed from pytorch tutorial
        Arguments:
            logits: [batch_size, seq_len, vocab_size] FloatTensor
            lens: [batch_size] LongTensor
        """
        logits = self.pad_logits(logits)
        batch_size, seq_len, n_labels = logits.size()
        vit = logits.data.new(batch_size, self.n_labels).fill_(-10000)
        vit[:, self.start_idx] = 0
        c_lens = lens.clone()

        # [seq_len, batch_size, vocab_size]
        logits_t = logits.transpose(1, 0)
        pointers = []

        for logit in logits_t:
            # logit: [batch_size, vocab_size]
            vit_exp = vit.unsqueeze(1).expand(batch_size, n_labels, n_labels)
            trn_exp = self.transitions.unsqueeze(0).expand_as(vit_exp)
            vit_trn_sum = (vit_exp + trn_exp)[:, :, :-2]  # excluding start and end
            vt_max, vt_argmax = vit_trn_sum.max(2)

            vt_max = vt_max.squeeze(-1)
            vit_nxt = vt_max + logit
            pointers.append(vt_argmax.squeeze(-1).unsqueeze(0))

            mask = (c_lens > 0).float().unsqueeze(-1).expand_as(vit_nxt)
            vit = mask * vit_nxt + (1 - mask) * vit

            mask = (c_lens == 1).float().unsqueeze(-1).expand_as(vit_nxt)
            vit += mask * self.transitions[ self.stop_idx ].unsqueeze(0).expand_as(vit_nxt)

            c_lens = c_lens - 1

        pointers = torch.cat(pointers)
        scores, idx = vit.max(1)
        idx = idx.squeeze(-1)
        paths = [idx.unsqueeze(1)]

        for argmax in reversed(pointers):
            idx_exp = idx.unsqueeze(-1)
            idx = torch.gather(argmax, 1, idx_exp)
            idx = idx.squeeze(-1)

            paths.insert(0, idx.unsqueeze(1))

        paths = torch.cat(paths[1:], 1)
        scores = scores.squeeze(-1)

        return scores, paths

    def transition_score(self, labels, lens):
        """
        Arguments:
             labels: [batch_size, seq_len] LongTensor
             lens: [batch_size] LongTensor
        """
        batch_size, seq_len = labels.size()

        # pad labels with <start> and <stop> indices
        labels_ext = labels.data.new(batch_size, seq_len + 2)
        labels_ext[:, 0] = self.start_idx
        labels_ext[:, 1:-1] = labels
        mask = sequence_mask(lens + 1, max_len=seq_len + 2).long()
        pad_stop = labels.data.new(1).fill_(self.stop_idx)
        pad_stop = pad_stop.unsqueeze(-1).expand(batch_size, seq_len + 2)
        labels_ext = (1 - mask) * pad_stop + mask * labels_ext
        labels = labels_ext

        trn = self.transitions

        # obtain transition vector for each label in batch and timestep
        # (except the last ones)

        #  transition score table for each sentence
        # (sent_count, tag_count, tag_count)
        trn_exp = trn.unsqueeze(0).expand(batch_size, *trn.size())
        # select right side label
        lbl_r = labels[:, 1:]
        lbl_rexp = lbl_r.unsqueeze(-1).expand(*lbl_r.size(), trn.size(0))
        trn_row = torch.gather(trn_exp, 1, lbl_rexp)

        # select left side label
        # obtain transition score from the transition vector for each label
        # in batch and timestep (except the first ones)
        lbl_lexp = labels[:, :-1].unsqueeze(-1)
        trn_scr = torch.gather(trn_row, 2, lbl_lexp)
        trn_scr = trn_scr.squeeze(-1)

        mask = sequence_mask(lens + 1, max_len=seq_len + 1).float()
        trn_scr = trn_scr * mask
        score = trn_scr.sum(1).squeeze(-1)

        return score

