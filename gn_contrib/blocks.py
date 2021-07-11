import sonnet as snt
import tensorflow as tf
from graph_nets.blocks import (
    ReceivedEdgesToNodesAggregator,
    SentEdgesToNodesAggregator,
    broadcast_globals_to_edges,
    broadcast_globals_to_nodes,
    broadcast_receiver_nodes_to_edges,
    broadcast_sender_nodes_to_edges,
    _get_static_num_edges,
    _get_static_num_nodes,
    _validate_graph,
    EDGES,
    NODES,
    N_EDGE,
    SENDERS,
    RECEIVERS,
)


__all__ = ["EdgeBlock", "NodeBlock"]


class EdgeBlock(snt.Module):
    def __init__(
        self,
        edge_model_fn,
        use_edges=True,
        use_receiver_nodes=True,
        use_sender_nodes=True,
        use_globals=True,
        name="edge_block",
    ):
        super(EdgeBlock, self).__init__(name=name)
        if not (use_edges or use_sender_nodes or use_receiver_nodes or use_globals):
            raise ValueError(
                "At least one of use_edges, use_sender_nodes, "
                "use_receiver_nodes or use_globals must be True."
            )
        self._use_edges = use_edges
        self._use_receiver_nodes = use_receiver_nodes
        self._use_sender_nodes = use_sender_nodes
        self._use_globals = use_globals
        self._edge_model = edge_model_fn()

    def _collect_features(self, graph):
        edges_to_collect = []

        if self._use_receiver_nodes:
            edges_to_collect.append(broadcast_receiver_nodes_to_edges(graph))

        if self._use_edges:
            _validate_graph(graph, (EDGES,), "when use_edges == True")
            edges_to_collect.append(graph.edges)

        if self._use_sender_nodes:
            edges_to_collect.append(broadcast_sender_nodes_to_edges(graph))

        if self._use_globals:
            num_edges_hint = _get_static_num_edges(graph)
            edges_to_collect.append(
                broadcast_globals_to_edges(graph, num_edges_hint=num_edges_hint)
            )

        collected_edges = tf.concat(edges_to_collect, axis=-1)
        return collected_edges

    def __call__(self, graph, edge_model_kwargs=None):
        if edge_model_kwargs is None:
            edge_model_kwargs = {}

        _validate_graph(graph, (SENDERS, RECEIVERS, N_EDGE), " when using an EdgeBlock")

        collected_edges = self._collect_features(graph)
        updated_edges = self._edge_model(collected_edges, **edge_model_kwargs)
        return graph.replace(edges=updated_edges)


class NodeBlock(snt.Module):
    def __init__(
        self,
        node_model_fn,
        use_received_edges=True,
        use_sent_edges=False,
        use_nodes=True,
        use_globals=True,
        received_edges_reducer=tf.math.unsorted_segment_sum,
        sent_edges_reducer=tf.math.unsorted_segment_sum,
        name="node_block",
    ):
        super(NodeBlock, self).__init__(name=name)
        if not (use_nodes or use_sent_edges or use_received_edges or use_globals):
            raise ValueError(
                "At least one of use_received_edges, use_sent_edges, "
                "use_nodes or use_globals must be True."
            )
        self._use_received_edges = use_received_edges
        self._use_sent_edges = use_sent_edges
        self._use_nodes = use_nodes
        self._use_globals = use_globals

        self._node_model = node_model_fn()
        if self._use_received_edges:
            if received_edges_reducer is None:
                raise ValueError(
                    "If `use_received_edges==True`, `received_edges_reducer` "
                    "should not be None."
                )
            self._received_edges_aggregator = ReceivedEdgesToNodesAggregator(
                received_edges_reducer
            )
        if self._use_sent_edges:
            if sent_edges_reducer is None:
                raise ValueError(
                    "If `use_sent_edges==True`, `sent_edges_reducer` "
                    "should not be None."
                )
            self._sent_edges_aggregator = SentEdgesToNodesAggregator(
                sent_edges_reducer
            )

    def _collect_features(self, graph):
        nodes_to_collect = []

        if self._use_nodes:
            _validate_graph(graph, (NODES,), "when use_nodes == True")
            nodes_to_collect.append(graph.nodes)

        if self._use_received_edges:
            nodes_to_collect.append(self._received_edges_aggregator(graph))

        if self._use_sent_edges:
            nodes_to_collect.append(self._sent_edges_aggregator(graph))

        if self._use_globals:
            # The hint will be an integer if the graph has node features and the total
            # number of nodes is known at tensorflow graph definition time, or None
            # otherwise.
            num_nodes_hint = _get_static_num_nodes(graph)
            nodes_to_collect.append(
                broadcast_globals_to_nodes(graph, num_nodes_hint=num_nodes_hint)
            )

        collected_nodes = tf.concat(nodes_to_collect, axis=-1)
        return collected_nodes

    def __call__(self, graph, node_model_kwargs=None):
        if node_model_kwargs is None:
            node_model_kwargs = {}

        collected_nodes = self._collect_features(graph)
        updated_nodes = self._node_model(collected_nodes, **node_model_kwargs)
        return graph.replace(nodes=updated_nodes)
