import numpy as np
import tensorflow as tf


__all__ = ["binary_crossentropy"]


def binary_crossentropy(
    output_graphs, expected, entity, class_weight=[1.0, 1.0], msg_ratio=1.0
):
    loss_for_all_msg = []
    start_idx = int(np.ceil(len(output_graphs) * msg_ratio))
    for predicted_graphs in output_graphs[start_idx:]:
        predicted = predicted_graphs.__getattribute__(entity)
        msg_losses = tf.keras.losses.binary_crossentropy(expected, predicted)
        msg_losses = tf.gather(class_weight, tf.cast(expected, tf.int32)) * msg_losses
        msg_loss = tf.math.reduce_mean(msg_losses)
        loss_for_all_msg.append(msg_loss)
    loss = tf.math.reduce_sum(tf.stack(loss_for_all_msg))
    loss = loss / len(output_graphs)
    return loss
