import tensorflow as tf


__all__ = ["binary_crossentropy"]


def binary_crossentropy(
    expected, output_graphs, entity, min_num_msg, class_weights
):
    loss_for_all_msg = []
    idx_expected = tf.cast(expected, tf.int32)
    sample_weights = tf.squeeze(tf.gather(class_weights, idx_expected))
    for predicted_graphs in output_graphs[min_num_msg:]:
        predicted = predicted_graphs.__getattribute__(entity)
        msg_losses = tf.keras.losses.binary_crossentropy(expected, predicted)
        msg_losses = sample_weights * msg_losses
        msg_loss = tf.math.reduce_mean(msg_losses)
        loss_for_all_msg.append(msg_loss)
    loss = tf.math.reduce_sum(tf.stack(loss_for_all_msg))
    loss = loss / len(output_graphs)
    return loss
