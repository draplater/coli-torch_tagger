import torch
from torch import nn
from torch.nn import Embedding

from graph_embedding.dataset import PROPERTY_NAMES
from graph_embedding.reader import GraphEmbeddingStatistics


def collect_neighbor_nodes(node_embedding, indices):
    """
    node_embedding: [batch_size, num_nodes, feature_dim]
    indices: [batch_size, num_nodes, num_neighbors]
    """
    batch_size, num_nodes, feature_dim = node_embedding.size()
    # shape: [batch_size, num_nodes * num_neighbors, feature_dim]
    indices_ = indices.view(batch_size, -1, 1).expand(-1, -1, feature_dim)
    # `indices_` represents new indices tensor:
    #     indices_[i][j * num_neighbors + k][l] = indices[i][j][k]
    #
    #     output[i][j][k][l] = \
    #       node_embedding[i][indices_[i][j * num_neighbors + k][l]][l] = \
    #         node_embedding[i][indices[i][j][k]][l]
    return node_embedding.gather(1, indices_) \
        .view(batch_size, num_nodes, -1, feature_dim)


class Highway(nn.Module):
    def __init__(self, size, num_layers, f):
        super().__init__()
        self.nonlinears = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.linears = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.f = f

    def forward(self, input):
        """
        :param input: tensor with shape of [batch_size, size]
        :return: tensor with shape of [batch_size, size]
        """
        for gate, linear, nonlinear in zip(self.gates, self.linears,
                                           self.nonlinears):
            gate = torch.sigmoid(gate(input))
            nonlinear = self.f(nonlinear(input))
            linear = linear(input)

            input = gate * nonlinear + (1 - gate) * linear
        return input


class GraphRNNGate(nn.Module):
    def __init__(self, hidden_size, f):
        super().__init__()
        self.w_in = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.u_in = nn.Linear(hidden_size, hidden_size, bias=False)
        self.u_out = nn.Linear(hidden_size, hidden_size, bias=True)
        self.f = f

    def forward(self, in_neigh_embeddings, out_neigh_embeddings,
                in_neigh_prev_hidden, out_neigh_prev_hidden):
        return self.f(self.w_in(in_neigh_embeddings)
                      + self.w_out(out_neigh_embeddings)
                      + self.u_in(in_neigh_prev_hidden)
                      + self.u_out(out_neigh_prev_hidden))


class GraphRNNEncoder(nn.Module):
    def __init__(self, options, statistics: GraphEmbeddingStatistics):
        super().__init__()

        self.use_char_embedding = options.use_char_embedding
        self.use_property_embeddings = options.use_property_embeddings
        self.use_highway = options.use_highway
        self.compress_node_embedding = options.compress_node_embedding
        self.num_rnn_layers = options.num_rnn_layers

        self.dropout = nn.Dropout(p=options.dropout_rate)

        self.word_embedding = Embedding(len(statistics.words),
                                        options.embedding_dim,
                                        padding_idx=0
                                        )
        self.edge_embedding = Embedding(len(statistics.edges),
                                        options.embedding_dim,
                                        padding_idx=0
                                        )
        self.embeddings = {}
        for vocab_name in PROPERTY_NAMES:
            vocab = getattr(statistics, vocab_name)
            embedding = Embedding(len(vocab),
                                  options.embedding_dim,
                                  padding_idx=0
                                  )
            embedding_name = '{}_embedding'.format(vocab_name)
            self.embeddings[vocab_name] = embedding
            self.add_module(embedding_name, embedding)

        if not options.word_vector_trainable:
            # Do not train embeddings
            self.word_embedding.weight.requires_grad_(False)

        node_embedding_dim = self.word_embedding.embedding_dim

        if self.use_char_embedding:
            char_dim = self.char_embedding.embedding_dim
            node_embedding_dim += options.char_lstm_hidden_size
            # TODO: change data format to avoid batch_fisrt=True
            self.char_lstm = nn.LSTM(char_dim,
                                     options.char_lstm_hidden_size,
                                     options.num_char_lstm_layers,
                                     batch_first=True)

        if self.use_property_embeddings:
            for prop_name in PROPERTY_NAMES:
                node_embedding_dim += self.embeddings[prop_name].embedding_dim

        if self.compress_node_embedding:
            self.compress_linear = nn.Linear(node_embedding_dim,
                                             options.compressed_embedding_dim)
            node_embedding_dim = options.compressed_embedding_dim

        if self.use_highway:
            self.multi_highway = Highway(node_embedding_dim,
                                         options.num_highway_layers,
                                         f=torch.tanh)

        edge_dim = self.edge_embedding.embedding_dim
        hidden_size = options.model_hidden_size
        self.hidden_size = hidden_size
        self.node_embedding_dim = node_embedding_dim
        self.neighbor_linear = nn.Linear(node_embedding_dim + edge_dim,
                                         hidden_size)

        self.input_gate = GraphRNNGate(hidden_size, torch.sigmoid)
        self.output_gate = GraphRNNGate(hidden_size, torch.sigmoid)
        self.forget_gate = GraphRNNGate(hidden_size, torch.sigmoid)
        self.cell = GraphRNNGate(hidden_size, torch.tanh)

    def compute_node_embedding(self, batch):
        batch_size, max_num_nodes = batch.node_labels.size()

        # shape: [batch_size, max_num_nodes, word_dim]
        # (batch * max_num_nodes) vectors represents nodes
        node_embedding = self.word_embedding(batch.node_labels)
        node_embedding_list = [node_embedding]

        # CharLSTM ########################################################
        if self.use_char_embedding:
            char_batch_size = batch_size * max_num_nodes
            # shape: [batch_size * max_num_nodes, max_chars_per_word]
            node_chars = batch.node_chars.view(char_batch_size, -1)
            # shape: [batch_size * max_num_nodes, max_chars_per_word, char_dim]
            node_chars_embedding = self.char_embedding(node_chars)
            # shape: [batch_size * max_num_nodes, max_chars_per_word,
            #         char_lstm_dim]
            chars_output, _ = self.char_lstm(node_chars_embedding)
            # shape: [batch_size * max_num_nodes]
            # The indices of last hidden states of the char lstm outputs
            char_lstm_last_indices = batch.node_chars_nums.view(-1) - 1
            batch_indices = torch.arange(0, char_batch_size,
                                         dtype=torch.long,
                                         device=node_embedding.device)
            # shape: [batch_size * max_num_nodes, char_lstm_dim]
            # We donot need bidirectional RNN, so we use padding+masking
            chars_output = chars_output[batch_indices, char_lstm_last_indices]
            # shape: [batch_size, max_num_nodes, char_lstm_dim]
            chars_output = chars_output.view(batch_size, max_num_nodes, -1)

            node_embedding_list.append(chars_output)

        # Properties Embeddings ###############################################
        if self.use_property_embeddings:
            for prop_name, properties in zip(PROPERTY_NAMES,
                                             batch.node_properties_list):
                embedding = self.embeddings[prop_name]
                node_embedding_list.append(embedding(properties))

        # node_embedding
        # shape: [batch_size, max_num_nodes, node_embedding_dim]
        node_embedding = torch.cat(node_embedding_list, dim=2)
        node_embedding = node_embedding * batch.nodes_mask

        if self.compress_node_embedding:
            node_embedding = torch.tanh(self.compress_linear(node_embedding))

        # Dropout layer will automatically disable when self.training = False
        node_embedding = self.dropout(node_embedding)

        if self.use_highway:
            node_embedding = self.multi_highway(node_embedding)

        return node_embedding

    def compute_neighbor_embedding(self, node_embedding,
                                   x_edge_labels, x_node_indices,
                                   x_nodes_mask):
        # shape: [batch_size, max_num_nodes, num_neighbors, edge_dim]
        x_edge_embeddings = self.edge_embedding(x_edge_labels)
        # shape: [batch_size, max_num_nodes, num_neighbors,
        #         node_embedding_dim]
        x_node_embeddings = collect_neighbor_nodes(node_embedding, x_node_indices)
        # shape: [batch_size, max_num_nodes, num_neighbors,
        #         node_embedding_dim + edge_dim]
        x_neigh_embeddings = torch.cat([x_edge_embeddings,
                                        x_node_embeddings], dim=3)
        x_neigh_embeddings = x_neigh_embeddings * x_nodes_mask
        # shape: [batch_size, max_num_nodes, node_embedding_dim + edge_dim]
        x_neigh_embeddings = x_neigh_embeddings.sum(dim=2)

        return torch.tanh(self.neighbor_linear(x_neigh_embeddings))

    def compute_neighbor_hidden(self, hidden_vector,
                                x_node_indices, x_nodes_mask, nodes_mask):
        # shape: [batch_size, max_num_nodes, num_neighbors, hidden_size]
        x_neigh_prev_hidden = collect_neighbor_nodes(hidden_vector, x_node_indices)
        x_neigh_prev_hidden = x_neigh_prev_hidden * x_nodes_mask
        # shape: [batch_size, max_num_nodes, hidden_size]
        x_neigh_prev_hidden = x_neigh_prev_hidden.sum(dim=2)
        x_neigh_prev_hidden = x_neigh_prev_hidden * nodes_mask
        return x_neigh_prev_hidden

    def forward(self, batch):
        # shape: [batch_size, max_num_nodes, node_embedding_dim]
        node_embedding = self.compute_node_embedding(batch)
        # shape: [batch_size, max_num_nodes, hidden_size]
        in_neigh_embeddings = self.compute_neighbor_embedding(node_embedding,
                                                              batch.in_edge_labels,
                                                              batch.in_node_indices,
                                                              batch.in_nodes_mask)
        # shape: [batch_size, max_num_nodes, hidden_size]
        out_neigh_embeddings = self.compute_neighbor_embedding(node_embedding,
                                                               batch.out_edge_labels,
                                                               batch.out_node_indices,
                                                               batch.out_nodes_mask)

        batch_size, max_num_nodes = batch.node_labels.size()
        hidden_shape = [batch_size, max_num_nodes, self.hidden_size]
        hidden_vector = torch.zeros(hidden_shape, device=node_embedding.device)
        cell_vector = torch.zeros(hidden_shape, device=node_embedding.device)

        graph_embeddings = []
        for i in range(self.num_rnn_layers):
            # Incoming hidden states ######################################
            # shape: [batch_size * max_num_nodes, hidden_size]
            in_neigh_prev_hidden = self.compute_neighbor_hidden(hidden_vector,
                                                                batch.in_node_indices,
                                                                batch.in_nodes_mask,
                                                                batch.nodes_mask)
            # Outgoing hidden states ######################################
            # shape: [batch_size * max_num_nodes, hidden_size]
            out_neigh_prev_hidden = self.compute_neighbor_hidden(hidden_vector,
                                                                 batch.out_node_indices,
                                                                 batch.out_nodes_mask,
                                                                 batch.nodes_mask)

            context = (in_neigh_embeddings, out_neigh_embeddings,
                       in_neigh_prev_hidden, out_neigh_prev_hidden)

            input_gate = self.input_gate(*context)
            output_gate = self.output_gate(*context)
            forget_gate = self.forget_gate(*context)
            cell_input = self.cell(*context)

            # [batch_size, max_node_num, hidden_size]
            cell_vector = forget_gate * cell_vector + input_gate * cell_input
            cell_vector = cell_vector * batch.nodes_mask

            # [batch_size, max_node_num, hidden_size]
            hidden_vector = output_gate * torch.tanh(cell_vector)
            # TODO: 对照公式
            graph_embeddings.append(hidden_vector)
        return graph_embeddings, node_embedding, (hidden_vector, cell_vector)
