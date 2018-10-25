import json
import pickle

from dataclasses import dataclass, field

from graph_embedding.dataset import PROPERTY_NAMES
from hrgguru.hrg import CFGRule
from hrgguru.hyper_graph import strip_category, HyperEdge
from vocab_utils import Dictionary


@dataclass
class GraphEmbeddingStatistics(object):
    words: Dictionary = field(default_factory=Dictionary)
    edges: Dictionary = field(default_factory=Dictionary)
    postags: Dictionary = field(default_factory=Dictionary)
    senses: Dictionary = field(default_factory=Dictionary)
    external_index: Dictionary = field(default_factory=Dictionary)


def read_hrg_rule(sync_rule: CFGRule, statistics: GraphEmbeddingStatistics,
                  allow_new=True):
    rule = sync_rule.hrg
    assert rule is not None

    left_edge = right_edge = None
    if sync_rule.rhs is not None:
        if isinstance(sync_rule.rhs[0][1], HyperEdge):
            left_edge = sync_rule.rhs[0][1]
        if len(sync_rule.rhs) >= 2 and isinstance(sync_rule.rhs[1][1], HyperEdge):
            right_edge = sync_rule.rhs[1][1]

    node_to_pred_edge = {edge.nodes[0]: edge
                         for edge in rule.rhs.edges
                         if len(edge.nodes) == 1}
    ret = {
        'nodes': [],
        'properties': [[] for _ in range(len(PROPERTY_NAMES))],
        'in_node_indices': [[] for _ in range(len(rule.rhs.nodes))],
        'out_node_indices': [[] for _ in range(len(rule.rhs.nodes))],
        'in_edge_labels': [[] for _ in range(len(rule.rhs.nodes))],
        'out_edge_labels': [[] for _ in range(len(rule.rhs.nodes))],
    }

    for node_id, node in enumerate(sorted(rule.rhs.nodes, key=lambda x: int(x.name))):
        assert int(node.name) == node_id
        try:
            external_index = str(rule.lhs.nodes.index(node))
        except ValueError:
            external_index = "-1"
        pred_edge = node_to_pred_edge.get(node)
        lemma = pos = sense = "None"
        if pred_edge == left_edge:
            lemma = "__LEFT__"
        elif pred_edge == right_edge:
            lemma = "__RIGHT__"
        elif pred_edge is not None:
            assert pred_edge.is_terminal
            lemma, pos, sense = strip_category(pred_edge.label, return_tuple=True)
        node_int = statistics.words.update_and_get_id(lemma, allow_new=allow_new)
        ret["nodes"].append(node_int)
        ret["properties"][0].append(statistics.postags.update_and_get_id(pos, allow_new=allow_new))
        ret["properties"][1].append(statistics.senses.update_and_get_id(sense, allow_new=allow_new))
        ret["properties"][2].append(statistics.external_index.update_and_get_id(external_index, allow_new=allow_new))

        ret["in_node_indices"][node_id].append(node_id)
        ret["out_node_indices"][node_id].append(node_id)
        ret["in_edge_labels"][node_id].append(statistics.edges.update_and_get_id("Self", allow_new=allow_new))
        ret["out_edge_labels"][node_id].append(statistics.edges.update_and_get_id("Self", allow_new=allow_new))

        for edge in rule.rhs.edges:
            if len(edge.nodes) >= 2 and node == edge.nodes[0]:
                label = edge.label
                if edge == left_edge:
                    label = "__LEFT__"
                elif edge == right_edge:
                    label = "__RIGHT__"
                else:
                    assert edge.is_terminal
                for other_node in edge.nodes[1:]:
                    other_node_id = int(other_node.name)
                    ret["out_node_indices"][node_id].append(other_node_id)
                    ret["out_edge_labels"][node_id].append(
                        statistics.edges.update_and_get_id(label, allow_new=allow_new))
                    ret["in_node_indices"][other_node_id].append(node_id)
                    ret["in_edge_labels"][other_node_id].append(
                        statistics.edges.update_and_get_id(label, allow_new=allow_new))
    return ret


if __name__ == '__main__':
    import os

    name = "deepbank1.1-lfrg2-small-qeq-181015"
    home = os.path.expanduser("~")
    grammar_file = home + f"/Development/HRGGuru/deepbank-preprocessed/cfg_hrg_mapping-{name}.pickle"
    simple_graphs = []
    statistics = GraphEmbeddingStatistics()
    with open(grammar_file, "rb") as f:
        grammar = pickle.load(f)
    for rules_and_counts in grammar.values():
        for rule in rules_and_counts.keys():
            if rule.hrg is not None:
                simple_graph = read_hrg_rule(rule, statistics)
                simple_graphs.append((rule, simple_graph))
                print(json.dumps(simple_graph))
                pass
    pass
