import torch

from graph_embedding.padding import pad_2d_values, sequence_mask, pad_3d_values

PROPERTY_NAMES = ['postags', 'senses', "external_index"]


class Batch:
    def init(self, options, instances):
        self.size = len(instances)
        # shape: [batch_size, num_nodes]
        self.node_labels = pad_2d_values([_['nodes'] for _ in instances])
        # shape: [batch_size, num_nodes]
        nodes_mask = sequence_mask(torch.tensor([len(_['nodes'])
                                                 for _ in instances]))
        self.nodes_mask = nodes_mask.float().unsqueeze(-1)

        if options.use_property_embeddings:
            # list of shape: [batch_size, num_nodes]
            self.node_properties_list = [
                pad_2d_values([_['properties'][prop_idx] for _ in instances])
                for prop_idx in range(len(PROPERTY_NAMES))
            ]

        if options.use_char_embedding:
            # shape: [batch_size, num_nodes, chars_per_word]
            self.node_chars = pad_3d_values([_['node_chars'] for _ in instances])
            # shape: [batch_size, num_nodes]
            self.node_chars_nums = pad_2d_values([[len(chars) for chars in _['node_chars']]
                                                  for _ in instances])

        # shape: [batch_size, num_nodes, incoming_degree]
        in_nodes_mask = sequence_mask(pad_2d_values([
            [len(indices) for indices in _['in_node_indices']]
            for _ in instances
        ]))
        self.in_nodes_mask = in_nodes_mask.float().unsqueeze(-1)

        # shape: [batch_size, num_nodes, incoming_degree]
        self.in_node_indices = pad_3d_values([_['in_node_indices'] for _ in instances])
        # shape: [batch_size, num_nodes, incoming_degree]
        self.in_edge_labels = pad_3d_values([_['in_edge_labels'] for _ in instances])

        # shape: [batch_size, num_nodes, outgoing_degree]
        try:
            out_nodes_mask = sequence_mask(pad_2d_values([
                [len(indices) for indices in _['out_node_indices']]
                for _ in instances
            ]))
            self.out_nodes_mask = out_nodes_mask.float().unsqueeze(-1)
            # shape: [batch_size, num_nodes, outgoing_degree]
            self.out_node_indices = pad_3d_values([_['out_node_indices'] for _ in instances])
            # shape: [batch_size, num_nodes, outgoing_degree]
            self.out_edge_labels = pad_3d_values([_['out_edge_labels'] for _ in instances])
        except KeyError:
            self.out_nodes_mask = self.out_node_indices = self.out_edge_labels = None

    def cuda(self):
        batch = Batch()
        batch.size = self.size
        batch.node_labels = self.node_labels.cuda()
        batch.nodes_mask = self.nodes_mask.cuda()

        if hasattr(self, 'node_properties_list'):
            batch.node_properties_list = [
                properties.cuda()
                for properties in self.node_properties_list
            ]

        if hasattr(self, 'node_chars'):
            batch.node_chars = self.node_chars.cuda()
            batch.node_chars_nums = self.node_chars_nums.cuda()

        batch.in_nodes_mask = self.in_nodes_mask.cuda()
        batch.in_node_indices = self.in_node_indices.cuda()
        batch.in_edge_labels = self.in_edge_labels.cuda()

        if hasattr(self, "out_nodes_mask"):
            batch.out_nodes_mask = self.out_nodes_mask.cuda()
            batch.out_node_indices = self.out_node_indices.cuda()
            batch.out_edge_labels = self.out_edge_labels.cuda()

        return batch