import dataclasses
import torch

from graph_embedding.padding import pad_2d_values, sequence_mask, pad_3d_values


class Batch:
    def init(self, options, instances):
        self.size = len(instances)
        # shape: [batch_size, num_entities]
        self.entity_labels = pad_2d_values([_['entities'] for _ in instances])
        # shape: [batch_size, num_entities]
        entities_mask = sequence_mask(torch.tensor([len(_['entities'])
                                                    for _ in instances]))
        self.entities_mask = entities_mask.float().unsqueeze(-1)

        if options.use_property_embeddings:
            # list of shape: [batch_size, num_entities]
            self.entity_properties_list = [
                pad_2d_values([_['properties'][prop_idx] for _ in instances])
                for prop_idx in range(len(instances[0]["properties"]))
            ]

        if options.use_char_embedding:
            # shape: [batch_size, num_entities, chars_per_word]
            self.entity_chars = pad_3d_values([_['entity_chars'] for _ in instances])
            # shape: [batch_size, num_entities]
            self.entity_chars_nums = pad_2d_values([[len(chars) for chars in _['entity_chars']]
                                                    for _ in instances])

        # shape: [batch_size, num_entities, incoming_degree]
        conn_mask = sequence_mask(pad_2d_values([
            [len(indices) for indices in _['conn_indices']]
            for _ in instances
        ]))
        self.conn_mask = conn_mask.float().unsqueeze(-1)

        # shape: [batch_size, num_entities, incoming_degree]
        self.conn_indices = pad_3d_values([_['conn_indices'] for _ in instances])
        # shape: [batch_size, num_entities, incoming_degree]
        self.conn_labels = pad_3d_values([_['conn_labels'] for _ in instances])

        if "out_conn_indices" in instances[0]:
            # shape: [batch_size, num_entities, out_degree]
            out_conn_mask = sequence_mask(pad_2d_values([
                [len(indices) for indices in _['out_conn_indices']]
                for _ in instances
            ]))
            self.out_conn_mask = out_conn_mask.float().unsqueeze(-1)

            # shape: [batch_size, num_entities, out_degree]
            self.out_conn_indices = pad_3d_values([_['out_conn_indices'] for _ in instances])
            # shape: [batch_size, num_entities, out_degree]
            self.out_conn_labels = pad_3d_values([_['out_conn_labels'] for _ in instances])

    def cuda(self):
        batch = Batch()
        batch.size = self.size
        batch.entity_labels = self.entity_labels.cuda()
        batch.entities_mask = self.entities_mask.cuda()

        if hasattr(self, 'entity_properties_list'):
            batch.entity_properties_list = [
                properties.cuda()
                for properties in self.entity_properties_list
            ]

        if hasattr(self, 'entity_chars'):
            batch.entity_chars = self.entity_chars.cuda()
            batch.entity_chars_nums = self.entity_chars_nums.cuda()

        batch.conn_mask = self.conn_mask.cuda()
        batch.conn_indices = self.conn_indices.cuda()
        batch.conn_labels = self.conn_labels.cuda()

        if hasattr(self, 'out_conn_mask'):
            batch.out_conn_mask = self.out_conn_mask.cuda()
            batch.out_conn_indices = self.out_conn_indices.cuda()
            batch.out_conn_labels = self.out_conn_labels.cuda()
        return batch
