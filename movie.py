"""
Simple script to make traffic signs prediction movie
"""
import pickle
import random

import tensorflow as tf
import numpy as np
import cv2
import pandas
import sklearn.utils


def get_test_data(path):

    with open(path, mode='rb') as f:
        test = pickle.load(f)

    X_test, y_test = test['features'], test['labels']
    return X_test, y_test


def get_model(images_placeholder, keep_probability_placeholder, n_classes):
    """
    A simple fully convolutional network
    """

    W = tf.Variable(tf.truncated_normal(mean=0, stddev=0.1, shape=[3, 3, 3, 32]))
    b = tf.Variable(tf.zeros(32))
    x = tf.nn.elu(tf.nn.conv2d(images_placeholder, W, strides=[1, 1, 1, 1], padding='VALID') + b)

    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    x = tf.nn.dropout(x, keep_probability_placeholder)

    W = tf.Variable(tf.truncated_normal(mean=0, stddev=0.1, shape=[3, 3, 32, 32]))
    b = tf.Variable(tf.zeros(32))
    x = tf.nn.elu(tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID') + b)

    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    x = tf.nn.dropout(x, keep_probability_placeholder)

    W = tf.Variable(tf.truncated_normal(mean=0, stddev=0.1, shape=[6, 6, 32, n_classes]))
    b = tf.Variable(tf.zeros(n_classes))
    x = tf.nn.elu(tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID') + b)

    return tf.squeeze(x)


def get_statistics(
    session, loss_op, accuracy_op, images_placeholder,
    labels_placeholder, keep_probability_placeholder, x, y, batch_size):
    """
    Simple helper to get loss and accuracy model attains on a given dataset
    """

    batches_count = len(y) // batch_size

    losses = []
    accuracies = []

    for batch_index in range(batches_count):
        batch_start = batch_size * batch_index
        batch_end = batch_start + batch_size

        x_batch = x[batch_start:batch_end]
        y_batch = y[batch_start:batch_end]

        feed_dictionary = {
            images_placeholder: x_batch,
            labels_placeholder: y_batch,
            keep_probability_placeholder: 1.0}

        batch_loss, batch_accuracy = session.run([loss_op, accuracy_op], feed_dict=feed_dictionary)

        losses.append(batch_loss)
        accuracies.append(batch_accuracy)

    return np.mean(losses), np.mean(accuracies)


def equalize_channel(channel):

    min_value = np.min(channel)
    max_value = np.max(channel)

    equalized_channel = (255 * (channel - min_value) / (max_value - min_value))
    return equalized_channel


def equalize_image(image):

    equalized_image = np.dstack([equalize_channel(image[:, :, channel].astype(np.float32)) for channel in range(3)])
    return equalized_image.astype(np.uint8)


def get_preprocessed_samples(samples):

    return np.array([equalize_image(image) for image in samples]).astype(np.float32) / 255



def get_samples_dictionary(images, labels, n_classes, samples_per_class):

    samples_dictionary = {}

    for label_index in range(n_classes):

        indices = np.where(labels == label_index)[0]
        random.shuffle(indices)

        samples_dictionary[label_index] = images[indices[:samples_per_class]]

    return samples_dictionary


def main():

    x_test, y_test = get_test_data("../../data/traffic-signs-data/test.p")
    n_classes = len(set(y_test))

    data_frame = pandas.read_csv("./signnames.csv")
    classes_dictionary = {int(id): class_name for id, class_name in zip(data_frame['ClassId'], data_frame['SignName'])}

    images_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3])
    keep_probability_placeholder = tf.placeholder(dtype=tf.float32)

    logits_op = get_model(images_placeholder, keep_probability_placeholder, n_classes)
    softmax_op = tf.nn.softmax(logits_op)
    top_5_prediction = tf.nn.top_k(softmax_op, k=5)

    samples_dictionary = get_samples_dictionary(x_test, y_test, n_classes, 10)

    images_predictions_tuples = []

    with tf.Session() as session:

        saver = tf.train.Saver()
        saver.restore(session, "../../data/traffic-signs-data/model/model.ckpt")

        for label, samples in samples_dictionary.items():

            feed_dictionary = {
                images_placeholder: get_preprocessed_samples(samples),
                keep_probability_placeholder: 1}

            predictions = session.run(top_5_prediction, feed_dict=feed_dictionary)

            images_predictions_tuples.append((samples, predictions))

    for images, predictions in sklearn.utils.shuffle(images_predictions_tuples):

        print()

        image = cv2.resize(cv2.cvtColor(images[0], cv2.COLOR_RGB2BGR), (128, 128))
        cv2.imshow("image", image)

        for prediction, prediction_index in zip(predictions.values[0], predictions.indices[0]):
            print("{}: {}%".format(classes_dictionary[prediction_index], 100 * prediction))

        cv2.waitKey(0)





if __name__ == "__main__":

    main()