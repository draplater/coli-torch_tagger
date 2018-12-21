import sys
import time
from argparse import ArgumentParser
from typing import Any, List

import dataclasses
import torch

from bilm.load_vocab import BiLMVocabLoader
from coli.basic_tools.common_utils import DictionarySubParser, AttrDict, Progbar, NoPickle, try_cache_keeper, \
    cache_result
from coli.data_utils.embedding import ExternalEmbeddingLoader
from coli.torch_extra.layers import get_external_embedding
from coli.torch_tagger.data_loader import SentenceWithTags, SentenceFeatures, Statistics, DTMLexicalType
from coli.torch_extra.parser_base import PyTorchParserBase
from .config import TaggerOptions
from .network import TaggerNetwork


class Tagger(PyTorchParserBase):
    """"""
    available_data_formats = {"default": SentenceWithTags, "dtm_lexical_type": DTMLexicalType}
    sentence_feature_class = SentenceFeatures

    @classmethod
    def add_parser_arguments(cls, arg_parser: ArgumentParser):
        super(Tagger, cls).add_parser_arguments(arg_parser)
        sub_parser = DictionarySubParser("hparams", arg_parser,
                                         choices={"default": TaggerOptions()},
                                         title=Tagger.__name__)

    @classmethod
    def add_common_arguments(cls, arg_parser):
        super(Tagger, cls).add_common_arguments(arg_parser)
        arg_parser.add_argument("--embed-file", metavar="FILE")
        arg_parser.add_argument("--gpu", action="store_true", default=False)

    def __init__(self, args: Any, data_train):
        super(Tagger, self).__init__(args, data_train)

        self.args = args
        self.hparams: TaggerOptions = args.hparams

        if self.args.bilm_path is not None:
            self.bilm_vocab = BiLMVocabLoader(self.args.bilm_path)
        else:
            self.bilm_vocab = None

        @cache_result(self.options.output + "/statistics.pkl",
                      enable=self.options.debug_cache)
        def load_statistics():
            return Statistics.from_sentences(data_train, self.hparams.word_threshold)

        self.statistics = load_statistics()

        if args.embed_file is not None:
            self.external_embedding_loader = NoPickle(ExternalEmbeddingLoader(args.embed_file))
        else:
            self.external_embedding_loader = None

        self.global_step = 0
        self.global_epoch = 1
        self.best_score = 0

        self.network = TaggerNetwork(self.hparams, self.statistics,
                                     self.external_embedding_loader,
                                     bilm_path=args.bilm_path
                                     )

        trainable_parameters = [param for param in self.network.parameters()
                                if param.requires_grad]

        self.optimizer = torch.optim.Adam(
            trainable_parameters
            # betas=(0.9, 0.98), eps=1e-9
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'max',
            factor=self.hparams.step_decay_factor,
            patience=self.hparams.step_decay_patience,
            verbose=True,
        )

        if self.options.gpu:
            self.network.cuda()

        self.progbar = NoPickle(Progbar(self.hparams.train_iters, log_func=self.file_logger.info))

    @property
    def file_logger(self):
        if getattr(self, "_file_logger", None) is None:
            self._file_logger = NoPickle(
                self.get_logger(self.options, False))
        return self._file_logger

    def set_lr(self, new_lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

    def schedule_lr(self, iteration):
        iteration = iteration + 1
        warmup_coeff = self.hparams.learning_rate / self.hparams.learning_rate_warmup_steps
        if iteration <= self.hparams.learning_rate_warmup_steps:
            self.set_lr(iteration * warmup_coeff)

    def train(self, train_data, dev_args_list=None):
        self.logger.info("Epoch: {}".format(self.global_epoch))
        total_loss = 0.0
        sent_count = 0
        start_time = time.time()
        self.network.train()
        for batch_sents, feed_dict in train_data.generate_batches(
                self.hparams.train_batch_size,
                shuffle=True, original=True,
                # sort_key_func=lambda x: x.sent_length
        ):
            self.schedule_lr(self.global_step)
            self.optimizer.zero_grad()
            output = self.network(batch_sents, AttrDict(feed_dict))
            total_loss += output.loss
            sent_count += output.sent_count
            output.loss.backward()
            self.optimizer.step()
            if self.global_step != 0 and \
                    self.global_step % self.hparams.print_every == 0:
                end_time = time.time()
                speed = sent_count / (end_time - start_time)
                start_time = end_time
                self.progbar.update(
                    self.global_step,
                    exact=[("Loss", total_loss), ("Speed", speed)]
                )
                sent_count = 0
                total_loss = 0.0
                if self.global_step % self.hparams.evaluate_every == 0:
                    if dev_args_list is not None:
                        for filename, data, buckets in dev_args_list:
                            output_file = self.get_output_name(
                                self.args.output, filename, self.global_step)
                            sys.stdout.write("\n")
                            outputs = self.predict_bucket(buckets)
                            self.evaluate_and_update_best_score(
                                data, outputs, log_file=output_file + ".txt")
                            with open(output_file, "w") as f:
                                for output in outputs:
                                    f.write(output.to_string())
                    self.network.train()
            self.global_step += 1

        self.global_epoch += 1
        self.progbar.finish()
        if self.global_step > self.hparams.learning_rate_warmup_steps:
            self.scheduler.step(self.best_score)

    def get_parsed(self, bucket, return_original=True):
        self.network.eval()
        with torch.no_grad():
            for batch_sent, feed_dict in bucket.generate_batches(
                    self.hparams.test_batch_size,
                    original=True,
                    # sort_key_func=lambda x: x.sent_length
            ):
                results = self.network(batch_sent, AttrDict(feed_dict))
                for original, parsed in zip(batch_sent, results.labels_pred.cpu().numpy()):
                    if return_original:
                        yield original, parsed
                    else:
                        yield parsed

    def evaluate_and_update_best_score(self, gold, outputs, log_file=None):
        p, r, f1 = SentenceWithTags.internal_evaluate(
            gold, outputs, log_file=log_file)
        last_best_score = self.best_score
        if f1 > last_best_score:
            self.logger.info("New best score: {:.2f} > {:.2f}, saving model...".format(
                f1, last_best_score))
            self.best_score = f1
            self.save(self.args.output + "/model")
        else:
            self.logger.info("No best score: {:.2f} <= {:.2f}".format(f1, last_best_score))

    def predict_bucket(self, bucket):
        outputs: List[Any] = [None for _ in range(len(bucket))]
        for sent_feature, labels_pred in self.get_parsed(bucket):
            sent = sent_feature.original_obj
            new_labels = list(self.statistics.labels.int_to_word[i]
                              for i in labels_pred[:len(sent.words)])
            outputs[sent_feature.original_idx] = dataclasses.replace(sent, labels=new_labels)
        return outputs

    def post_load(self, new_options):
        # TODO: without external embedding?
        self.options.__dict__.update(new_options.__dict__)

        if self.options.embed_file:
            assert new_options.embed_file, "Embedding file is required"
            self.external_embedding_loader = NoPickle(ExternalEmbeddingLoader(new_options.embed_file))
            self.network.pretrained_embeddings = NoPickle(get_external_embedding(self.external_embedding_loader))
            if self.options.gpu:
                self.network.pretrained_embeddings.cuda()

        if self.options.bilm_path:
            self.network.load_bilm(self.options.bilm_path, self.options.gpu)
