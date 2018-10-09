#! /usr/bin/env python3

""" Handwritten digits recoginization with tensorflow.

Model: Feed forward neural network
using minst data set.

"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


data = input_data.read_data_sets("/tmp/data", one_hot=True)


network_classes = 10

batch_features_size = 100

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')


def neural_model(data_in):
    hidden_layer1 = {'weights':tf.Variable(tf.random_normal([784, 500])),
                    'biases': tf.Variable(tf.random_normal([500]))}
    print("The", hidden_layer1)

    hidden_layer2 = {'weights':tf.Variable(tf.random_normal([500, 500])),
                    'biases': tf.Variable(tf.random_normal([500]))}

    hidden_layer3 = {'weights':tf.Variable(tf.random_normal([500, 500])),
                     'biases': tf.Variable(tf.random_normal([500]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([500, 10])),
                    'biases': tf.Variable(tf.random_normal([10]))}

    layer1 = tf.add(tf.matmul(data_in, hidden_layer1['weights']) , hidden_layer1['biases'])
    print("The layer", layer1)
    layer1 = tf.nn.relu(layer1)

    layer2 = tf.add(tf.matmul(layer1, hidden_layer2['weights']), hidden_layer2['biases'])
    layer2 = tf.nn.relu(layer2)
    layer3 = tf.add(tf.matmul(layer2, hidden_layer3['weights']),  hidden_layer3['biases'])
    layer3 = tf.nn.relu(layer3)

    final_layer = tf.matmul(layer3, output_layer['weights'] + output_layer['biases'])

    return final_layer


def train_model(data_in):
    prediction = neural_model(data_in)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # no of cycles
    no_epochs = 10

    with tf.Session() as s:
        s.run(tf.global_variables_initializer())

        for epoch in range(no_epochs):
            epoch_loss = 0

            for i in range(int(data.train.num_examples/batch_features_size)):
                ex, ey = data.train.next_batch(batch_features_size)
                n, c = s.run([optimizer, cost], feed_dict={x:ex, y:ey})
                epoch_loss += c
            print("Epoch: ", epoch, "completed: ", no_epochs, "Loss: ", epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print("Accuracy: ", accuracy.eval({x:data.test.images, y:data.test.labels}))

train_model(x)
