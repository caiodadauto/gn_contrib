import sonnet as snt
import tensorflow as tf


__all__ = [
    "LeakyReluMLP",
    "EdgeTau",
    "NodeTau",
]


class LeakyReluMLP(snt.Module):
    def __init__(
        self, hidden_size, num_of_layers, dropout_rate, alpha, name="LeakyReluMLP"
    ):
        super(LeakyReluMLP, self).__init__(name=name)
        self._linear_layers = []
        self._alpha = alpha
        self._hidden_size = hidden_size
        self._dropout_rate = dropout_rate
        self._num_of_layers = num_of_layers
        for _ in range(self._num_of_layers):
            self._linear_layers.append(snt.Linear(self._hidden_size))

    def __call__(self, inputs, is_training):
        outputs_op = inputs
        for linear in self._linear_layers:
            if is_training:
                outputs_op = tf.nn.dropout(outputs_op, rate=self._dropout_rate)
            outputs_op = linear(outputs_op)
            outputs_op = tf.nn.leaky_relu(outputs_op, alpha=self._alpha)
        return outputs_op


class EdgeTau(snt.Module):
    # TODO: Include the parameters in the initialization to add possible different dimensions for values, queries and keys
    def __init__(self, key_model_fn, query_model_fn, value_model_fn, name="EdgeTau"):
        super(EdgeTau, self).__init__(name=name)
        self._key_model = key_model_fn()
        self._query_model = query_model_fn()
        self._value_model = value_model_fn()

    @snt.once
    def _initialize_feature_dimension(self, inputs):
        dim_concat = inputs.shape[-1]
        if dim_concat % 3 != 0:
            raise ValueError(
                "It is expected the concatenation of three"
                " entity features for edge feature"
            )
        self._dim_feature = dim_concat // 3

    def __call__(self, inputs, is_training):
        self._initialize_feature_dimension(inputs)
        predecessor_features = inputs[:, 0 : self._dim_feature]
        query = self._query_model(predecessor_features, is_training)
        key = self._key_model(inputs, is_training)
        value = self._value_model(inputs, is_training)
        d = tf.math.sqrt(tf.cast(key.shape[-1], tf.float32))
        alpha = tf.math.exp(tf.math.reduce_sum(query * key, keepdims=True, axis=-1) / d)
        return tf.concat([alpha, value], axis=-1)


class NodeTau(snt.Module):
    def __init__(self, value_model_fn, name="NodeTau"):
        super(NodeTau, self).__init__(name=name)
        self._value_model = value_model_fn()

    def __call__(self, inputs, is_training):
        return self._value_model(inputs, is_training)
