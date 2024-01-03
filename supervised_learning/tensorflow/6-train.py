#!/usr/bin/env python3
"""summary
"""
import tensorflow.compat.v1 as tf
create_placeholders = __import__('0-create_placeholders').create_placeholders
forward_prop = __import__('2-forward_prop').forward_prop
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_train_op = __import__('5-create_train_op').create_train_op


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations,
          alpha, iterations, save_path="/tmp/model.ckpt"):
    """Function that builds, trains, and saves a neural network classifier

    Args:
        X_train (numpy.ndarray): contains the training input data
        Y_train (numpy.ndarray): contains the training labels
        X_valid (numpy.ndarray): contains the validation input data
        Y_valid (numpy.ndarray): contains the validationa labels
        layer_sizes (list): containsthe number of nodes in each
            layer of the network
        activations (list): contains the activation functions in
            each layer of the network
        alpha (float): learning rate
        iterations (int): number of iterations to train over
        save_path (str, optional): designates where to save the model.
            Defaults to "/tmp/model.ckpt".
    """
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)

    y_pred = forward_prop(x, layer_sizes, activations)
    tf.add_to_collection('y_pred', y_pred)

    loss = calculate_loss(y, y_pred)
    tf.add_to_collection('loss', loss)

    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuracy', accuracy)

    train = create_train_op(loss, alpha)
    tf.add_to_collection('train', train)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as ses:
        ses.run(init)
        for i in range(iterations + 1):
            # Cost and accuracy for training sets
            cost_train = ses.run(loss, feed_dict={x: X_train, y: Y_train})
            accuracy_train = ses.run(accuracy,
                                     feed_dict={x: X_train, y: Y_train})
            # Cost and accuracy for validation sets
            cost_val = ses.run(loss, feed_dict={x: X_valid, y: Y_valid})
            accuracy_val = ses.run(accuracy,
                                   feed_dict={x: X_valid, y: Y_valid})
            if i % 100 == 0 or i == iterations:
                print('After {} iterations:'.format(i))
                print('\tTraining Cost: {}'.format(cost_train))
                print('\tTraining Accuracy: {}'.format(accuracy_train))
                print('\tValidation Cost: {}'.format(cost_val))
                print('\tValidation Accuracy: {}'.format(accuracy_val))
            if i < iterations:
                ses.run(train, feed_dict={x: X_train, y: Y_train})
        return saver.save(ses, save_path)
