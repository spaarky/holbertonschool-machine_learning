#!/usr/bin/env python3
"""Summary
"""
import tensorflow.compat.v1 as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """Function that trains a loaded neural network model using mini-batch
            gradient descent

    Args:
        X_train (numpy.ndarray): contains training data
        Y_train (numpy.ndarray): contains training labels
        X_valid (numpy.ndarray): contains validation data
        Y_valid (numpy.ndarray): contains validation labels
        batch_size (int, optional): number of datapoints in a batch.
            Defaults to 32.
        epochs (int, optional): number of time the training should pass
            throught the whole data set. Defaults to 5.
        load_path (str, optional): the path from which to load the model.
            Defaults to "/tmp/model.ckpt".
        save_path (str, optional): the path to where the model should be saved
            after training. Defaults to "/tmp/model.ckpt".

    Returns:
        str: the path where the model was saved
    """

    with tf.Session() as ses:
        saver = tf.train.import_meta_graph(load_path + ".meta")
        saver.restore(ses, load_path)

        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection("loss")[0]
        train_op = tf.get_collection("train_op")[0]

        m = X_train.shape[0]
        # Mini batch definition
        if m % batch_size == 0:
            n_batches = m // batch_size
        else:
            n_batches = m // batch_size + 1

        # Training loop
        for i in range(epochs + 1):
            cost_train = ses.run(loss, feed_dict={x: X_train, y: Y_train})
            accuracy_train = ses.run(accuracy,
                                     feed_dict={x: X_train, y: Y_train})
            cost_val = ses.run(loss, feed_dict={x: X_valid, y: Y_valid})
            accuracy_val = ses.run(accuracy,
                                   feed_dict={x: X_valid, y: Y_valid})
            print("After {} epochs:".format(i))
            print("\tTraining Cost: {}".format(cost_train))
            print("\tTraining Accuracy: {}".format(accuracy_train))
            print("\tValidation Cost: {}".format(cost_val))
            print("\tValidation Accuracy: {}".format(accuracy_val))

            if i < epochs:
                shuffled_X, shuffled_Y = shuffle_data(X_train, Y_train)

                # mini batches
                for b in range(n_batches):
                    start = b * batch_size
                    end = (b + 1) * batch_size
                    if end > m:
                        end = m
                    X_mini_batch = shuffled_X[start:end]
                    Y_mini_batch = shuffled_Y[start:end]

                    next_train = {x: X_mini_batch, y: Y_mini_batch}
                    ses.run(train_op, feed_dict=next_train)

                    if (b + 1) % 100 == 0 and b != 0:
                        loss_mini_batch = ses.run(loss, feed_dict=next_train)
                        acc_mini_batch = ses.run(accuracy,
                                                 feed_dict=next_train)
                        print("\tStep {}:".format(b + 1))
                        print("\t\tCost: {}".format(loss_mini_batch))
                        print("\t\tAccuracy: {}".format(acc_mini_batch))

        return saver.save(ses, save_path)
