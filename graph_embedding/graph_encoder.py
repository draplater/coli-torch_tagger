import torch
from allennlp.modules.attention import BilinearAttention
from dataclasses import dataclass
from torch import nn
from torch.nn import Embedding

from graph_embedding.dataset import Batch
from graph_embedding.reader import GraphEmbeddingStatisticsBase


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
    def __init__(self, hidden_size, f, out=False):
        super().__init__()
        self.w_in = nn.Linear(hidden_size, hidden_size, bias=False)
        self.u_in = nn.Linear(hidden_size, hidden_size, bias=True)
        self.f = f
        self.out = out

        if self.out:
            self.w_out = nn.Linear(hidden_size, hidden_size, bias=False)
            self.u_out = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, in_neigh_embeddings, in_neigh_prev_hidden,
                out_neigh_embeddings=None, out_neigh_prev_hidden=None
                ):
        inputs = self.w_in(in_neigh_embeddings) + self.u_in(in_neigh_prev_hidden)
        if self.out:
            inputs += self.w_out(out_neigh_embeddings) + self.u_out(out_neigh_prev_hidden)
        return self.f(inputs)


class GraphRNNEncoder(nn.Module):
    @dataclass
    class Options(object):
        use_property_embeddings: bool = True
        num_rnn_layers: int = 3
        dropout_rate: float = 0.2
        word_vector_trainable: bool = True
        model_hidden_size: int = 512

        compress_node_embedding: bool = False
        compressed_embedding_dim: int = 100

        use_char_embedding: bool = False
        char_lstm_hidden_size: int = 100
        num_char_lstm_layers: int = 1

        use_highway: bool = False
        num_highway_layers: int = 2

        use_attention: bool = False

    def __init__(self, options: Options,
                 statistics: GraphEmbeddingStatisticsBase):
        super().__init__()

        self.use_char_embedding = options.use_char_embedding
        self.use_property_embeddings = options.use_property_embeddings
        self.use_highway = options.use_highway
        self.compress_node_embedding = options.compress_node_embedding
        self.num_rnn_layers = options.num_rnn_layers

        self.dropout = nn.Dropout(p=options.dropout_rate)

        self.word_embedding = Embedding(len(statistics.words),
                                        statistics.get_embedding_dim_of("words"),
                                        padding_idx=0
                                        )
        self.conn_label_embedding = Embedding(len(statistics.conn_labels),
                                              statistics.get_embedding_dim_of("conn_labels"),
                                              padding_idx=0
                                              )
        self.embeddings = {}

        self.properties = statistics.get_properties()
        for vocab_name in self.properties:
            vocab = getattr(statistics, vocab_name)
            embedding = Embedding(len(vocab),
                                  statistics.get_embedding_dim_of(vocab_name),
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
            for prop_name in self.properties:
                node_embedding_dim += self.embeddings[prop_name].embedding_dim

        if self.compress_node_embedding:
            self.compress_linear = nn.Linear(node_embedding_dim,
                                             options.compressed_embedding_dim)
            node_embedding_dim = options.compressed_embedding_dim

        if self.use_highway:
            self.multi_highway = Highway(node_embedding_dim,
                                         options.num_highway_layers,
                                         f=torch.tanh)

        conn_label_dim = self.conn_label_embedding.embedding_dim
        hidden_size = options.model_hidden_size
        self.hidden_size = hidden_size
        self.node_embedding_dim = node_embedding_dim
        self.neighbor_linear = nn.Linear(node_embedding_dim + conn_label_dim,
                                         hidden_size)

        self.use_out = use_out = statistics.use_out
        self.input_gate = GraphRNNGate(hidden_size, torch.sigmoid, use_out)
        self.output_gate = GraphRNNGate(hidden_size, torch.sigmoid, use_out)
        self.forget_gate = GraphRNNGate(hidden_size, torch.sigmoid, use_out)
        self.cell = GraphRNNGate(hidden_size, torch.tanh, use_out)

        if options.use_attention:
            self.embedding_attention = BilinearAttention(
                self.node_embedding_dim,
                self.node_embedding_dim + self.conn_label_embedding.embedding_dim,
                activation=torch.nn.functional.tanh
            )

            self.hidden_attention = BilinearAttention(
                self.hidden_size,
                self.hidden_size,
                activation=torch.nn.functional.tanh
            )
        else:
            # use sum instead of attention
            self.embedding_attention = self.hidden_attention = None

    def compute_entity_embedding(self, batch: Batch):
        batch_size, max_num_nodes = batch.entity_labels.size()

        # shape: [batch_size, max_num_nodes, word_dim]
        # (batch * max_num_nodes) vectors represents nodes
        node_embedding = self.word_embedding(batch.entity_labels)
        node_embedding_list = [node_embedding]

        # CharLSTM ########################################################
        if self.use_char_embedding:
            char_batch_size = batch_size * max_num_nodes
            # shape: [batch_size * max_num_nodes, max_chars_per_word]
            node_chars = batch.entity_chars.view(char_batch_size, -1)
            # shape: [batch_size * max_num_nodes, max_chars_per_word, char_dim]
            node_chars_embedding = self.char_embedding(node_chars)
            # shape: [batch_size * max_num_nodes, max_chars_per_word,
            #         char_lstm_dim]
            chars_output, _ = self.char_lstm(node_chars_embedding)
            # shape: [batch_size * max_num_nodes]
            # The indices of last hidden states of the char lstm outputs
            char_lstm_last_indices = batch.entity_chars_nums.view(-1) - 1
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
            for prop_name, properties in zip(
                    self.properties, batch.entity_properties_list):
                embedding = self.embeddings[prop_name]
                node_embedding_list.append(embedding(properties))

        # node_embedding
        # shape: [batch_size, max_num_nodes, node_embedding_dim]
        node_embedding = torch.cat(node_embedding_list, dim=2)
        node_embedding = node_embedding * batch.entities_mask

        if self.compress_node_embedding:
            node_embedding = torch.tanh(self.compress_linear(node_embedding))

        # Dropout layer will automatically disable when self.training = False
        node_embedding = self.dropout(node_embedding)

        if self.use_highway:
            node_embedding = self.multi_highway(node_embedding)

        return node_embedding

    def compute_neighbor_embedding(self, entity_embedding,
                                   x_conn_labels, x_conn_indices,
                                   x_conn_mask):
        # shape: [batch_size, max_num_nodes, num_neighbors, edge_dim]
        x_conn_embeddings = self.conn_label_embedding(x_conn_labels)
        # shape: [batch_size, max_num_nodes, num_neighbors,
        #         node_embedding_dim]
        x_entity_embeddings = collect_neighbor_nodes(entity_embedding, x_conn_indices)
        # shape: [batch_size, max_num_nodes, num_neighbors,
        #         node_embedding_dim + edge_dim]
        x_neigh_embeddings = torch.cat([x_conn_embeddings,
                                        x_entity_embeddings], dim=3)
        x_neigh_embeddings = x_neigh_embeddings * x_conn_mask

        if self.embedding_attention is not None:
            batch_size, max_num_nodes, num_neighbors = x_conn_labels.shape
            # [batch_size * max_num_nodes, input_dim]
            entity_embedding_2d = entity_embedding.view(batch_size * max_num_nodes, -1)
            # [batch_size * max_num_nodes, num_neighbors, input_dim]
            neigh_embeddings_2d = x_neigh_embeddings.view(batch_size * max_num_nodes, num_neighbors, -1)
            # [batch_size * max_num_nodes, num_neighbors]
            conn_mask_2d = x_conn_mask.view(batch_size * max_num_nodes, num_neighbors)
            attn_weights = self.embedding_attention(entity_embedding_2d, neigh_embeddings_2d, conn_mask_2d)
            x_neigh_embeddings_uni = torch.bmm(attn_weights.unsqueeze(1), neigh_embeddings_2d
                                               ).squeeze(1).view(batch_size, max_num_nodes, -1)
        else:
            x_neigh_embeddings_uni = x_neigh_embeddings.sum(-2)

        return torch.tanh(self.neighbor_linear(x_neigh_embeddings_uni))

    def compute_neighbor_hidden(self, hidden_vector,
                                x_conn_indices, x_conn_mask, entities_mask):
        # shape: [batch_size, max_num_nodes, num_neighbors, hidden_size]
        x_neigh_prev_hidden = collect_neighbor_nodes(hidden_vector, x_conn_indices)
        x_neigh_prev_hidden = x_neigh_prev_hidden * x_conn_mask

        if self.hidden_attention is not None:
            batch_size, max_num_nodes, num_neighbors = x_conn_indices.shape
            # [batch_size * max_num_nodes, hidden_dim]
            hidden_vector_2d = hidden_vector.view(batch_size * max_num_nodes, -1)
            # [batch_size * max_num_nodes, num_neighbors, input_dim]
            neigh_prev_hidden_2d = x_neigh_prev_hidden.view(batch_size * max_num_nodes, num_neighbors, -1)
            # [batch_size * max_num_nodes, num_neighbors]
            conn_mask_2d = x_conn_mask.view(batch_size * max_num_nodes, num_neighbors)
            attn_weights = self.hidden_attention(hidden_vector_2d, neigh_prev_hidden_2d, conn_mask_2d)

            # shape: [batch_size, max_num_nodes, hidden_size]
            x_neigh_prev_hidden_uni = torch.bmm(attn_weights.unsqueeze(1), neigh_prev_hidden_2d
                                                ).squeeze(1).view(batch_size, max_num_nodes, -1)
            x_neigh_prev_hidden_uni = x_neigh_prev_hidden_uni * entities_mask
        else:
            x_neigh_prev_hidden_uni = x_neigh_prev_hidden.sum(-2)
        return x_neigh_prev_hidden_uni

    def forward(self, batch: Batch):
        # shape: [batch_size, max_num_nodes, node_embedding_dim]
        entity_embedding = self.compute_entity_embedding(batch)
        # shape: [batch_size, max_num_nodes, hidden_size]
        neigh_embeddings = self.compute_neighbor_embedding(entity_embedding,
                                                           batch.conn_labels,
                                                           batch.conn_indices,
                                                           batch.conn_mask)

        if self.use_out:
            out_neigh_embeddings = self.compute_neighbor_embedding(entity_embedding,
                                                                   batch.out_conn_labels,
                                                                   batch.out_conn_indices,
                                                                   batch.out_conn_mask)
        else:
            out_neigh_embeddings = None

        batch_size, max_num_nodes = batch.entity_labels.size()
        hidden_shape = [batch_size, max_num_nodes, self.hidden_size]
        hidden_vector = torch.zeros(hidden_shape, device=entity_embedding.device)
        cell_vector = torch.zeros(hidden_shape, device=entity_embedding.device)

        graph_embeddings = []
        for i in range(self.num_rnn_layers):
            # Incoming hidden states ######################################
            # shape: [batch_size * max_num_nodes, hidden_size]
            neigh_prev_hidden = self.compute_neighbor_hidden(hidden_vector,
                                                             batch.conn_indices,
                                                             batch.conn_mask,
                                                             batch.entities_mask)

            if self.use_out:
                out_neigh_prev_hidden = self.compute_neighbor_hidden(hidden_vector,
                                                                 batch.out_conn_indices,
                                                                 batch.out_conn_mask,
                                                                 batch.entities_mask)
            else:
                out_neigh_embeddings = None

            context = (neigh_embeddings, neigh_prev_hidden, out_neigh_embeddings, out_neigh_prev_hidden)

            input_gate = self.input_gate(*context)
            output_gate = self.output_gate(*context)
            forget_gate = self.forget_gate(*context)
            cell_input = self.cell(*context)

            # [batch_size, max_node_num, hidden_size]
            cell_vector = forget_gate * cell_vector + input_gate * cell_input
            cell_vector = cell_vector * batch.entities_mask

            # [batch_size, max_node_num, hidden_size]
            hidden_vector = output_gate * torch.tanh(cell_vector)
            # TODO: 对照公式
            graph_embeddings.append(hidden_vector)
        return graph_embeddings, entity_embedding, (hidden_vector, cell_vector)
