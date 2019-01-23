from dataclasses import dataclass

from coli.basic_tools.common_utils import cache_result
from coli.torch_tagger.config import TaggerHParams
from coli.torch_tagger.data_loader import SentenceWithTags, SentenceFeatures, Statistics, DTMLexicalType
from coli.torch_extra.parser_base import SimpleParser
from .network import TaggerNetwork


class Tagger(SimpleParser):
    """"""
    available_data_formats = {"default": SentenceWithTags, "dtm_lexical_type": DTMLexicalType}
    sentence_feature_class = SentenceFeatures
    target = "labels"

    @dataclass
    class Options(SimpleParser.Options):
        hparams: "TaggerHParams" = TaggerHParams.get_default()

    def __init__(self, args, data_train):
        super(Tagger, self).__init__(args, data_train)
        self.args: "Tagger.Options"
        self.hparams: "TaggerHParams"

        @cache_result(self.options.output + "/statistics.pkl",
                      enable=self.options.debug_cache)
        def load_statistics():
            return Statistics.from_sentences(data_train, self.hparams.word_threshold)

        self.statistics = load_statistics()

        self.network = TaggerNetwork(self.hparams, self.statistics, self.plugins)
        if self.args.gpu:
            self.network.cuda()

        self.trainable_parameters = [param for param in self.network.parameters()
                                if param.requires_grad]

        self.optimizer, self.scheduler = self.get_optimizer_and_scheduler(self.trainable_parameters)

    def split_batch_result(self, batch_result):
        yield from batch_result.labels_pred.cpu().numpy()

    def merge_answer(self, sent_feature, answer):
        sent = sent_feature.original_obj
        new_labels = list(self.statistics.labels.int_to_word[i]
                          for i in answer[:len(sent.words)])
        return sent.replaced_labels(new_labels)
