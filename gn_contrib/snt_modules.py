from functools import partial

import sonnet as snt
import tensorflow as tf


__all__ = [
    "LeakyReluMLP",
    "EdgeTau",
    "NodeTau",
]


class LeakyReluMLP(snt.Module):
    def __init__(self, hidden_sizes, dropout_rate, alpha, name="LeakyReluMLP"):
        super(LeakyReluMLP, self).__init__(name=name)
        self._linear_layers = []
        self._alpha = alpha
        self._dropout_rate = dropout_rate
        for hs in range(hidden_sizes):
            self._linear_layers.append(snt.Linear(hs))

    def __call__(self, inputs, is_training):
        outputs_op = inputs
        for linear in self._linear_layers:
            if is_training:
                outputs_op = tf.nn.dropout(outputs_op, rate=self._dropout_rate)
            outputs_op = linear(outputs_op)
            outputs_op = tf.nn.leaky_relu(outputs_op, alpha=self._alpha)
        return outputs_op


class EdgeTau(snt.Module):
    def __init__(
        self,
        key_model_fn,
        query_model_fn,
        value_model_fn,
        node_feature_dim,
        key_feature_dim,
        name="EdgeTau",
    ):
        super(EdgeTau, self).__init__(name=name)
        self._key_model = key_model_fn()
        self._query_model = query_model_fn()
        self._value_model = value_model_fn()
        self._predecessor_dim = node_feature_dim
        self._ratio = tf.cast(key_feature_dim, tf.float32)

    def __call__(self, inputs, is_training):
        predecessor_features = inputs[:, 0 : self._predecessor_dim]
        query = self._query_model(predecessor_features, is_training)
        key = self._key_model(inputs, is_training)
        value = self._value_model(inputs, is_training)
        alpha = tf.math.exp(
            tf.math.sigmoid(
                tf.math.reduce_sum(query * key, keepdims=True, axis=-1) / self._ratio
            )
        )
        return tf.concat([alpha, value], axis=-1)


class NodeTau(snt.Module):
    def __init__(self, value_model_fn, name="NodeTau"):
        super(NodeTau, self).__init__(name=name)
        self._value_model = value_model_fn()

    def __call__(self, inputs, is_training):
        return self._value_model(inputs, is_training)


def make_leaky_relu_mlp(hidden_sizes, dropout_rate=0.32, alpha=0.2):
    return LeakyReluMLP(hidden_sizes, dropout_rate, alpha)


def make_edge_tau(
    key_hidden_sizes,
    query_hidden_sizes,
    value_hidden_sizes,
    key_dropout_rate=0.32,
    key_alpha=0.2,
    query_dropout_rate=0.32,
    query_alpha=0.2,
    value_dropout_rate=0.32,
    value_alpha=0.2,
):
    key_model_fn = partial(
        make_leaky_relu_mlp,
        key_hidden_sizes,
        key_dropout_rate,
        key_alpha,
    )
    query_model_fn = partial(
        make_leaky_relu_mlp,
        query_hidden_sizes,
        query_dropout_rate,
        query_alpha,
    )
    value_model_fn = partial(
        make_leaky_relu_mlp,
        value_hidden_sizes,
        value_dropout_rate,
        value_alpha,
    )
    return EdgeTau(key_model_fn, query_model_fn, value_model_fn)


def make_node_tau(value_hidden_sizes, value_dropout_rate=0.32, value_alpha=0.2):
    value_model_fn = partial(
        make_leaky_relu_mlp,
        value_hidden_sizes,
        value_dropout_rate,
        value_alpha,
    )
    return NodeTau(value_model_fn)


def make_layer_norm(axis, scale=True, offset=True):
    return snt.LayerNorm(axis, scale, offset)
