#!/usr/bin/env python3
"""summary
"""
import tensorflow.compat.v1 as tf

def evaluate(X, Y, save_path):
    """Function that evaluates the output of a neural network

    Args:
        X (numpy.ndarray): contains the input data to evaluate
        Y (numpy.ndarray): contains the one-hot labels for X
        save_path (string): location to load the model from
    """
    with tf.Session() as ses:
        saver = tf.train.import_meta_graph(save_path + '.meta')
        saver.restore(ses, save_path)

        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        y_pred = tf.get_collection("y_pred")[0]
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]

        y_eval = ses.run(y_pred, feed_dict={x: X, y: Y})
        accuracy_eval = ses.run(accuracy, feed_dict={x: X, y: Y})
        loss_eval = ses.run(loss, feed_dict={x: X, y: Y})

        return y_eval,accuracy_eval, loss_eval
