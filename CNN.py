import tensorflow as tf
import numpy as np
import jaffe_parser
import matplotlib.pyplot as plt
import fer_parser
import pickle


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden):

    l1a = tf.nn.relu(tf.nn.conv2d(X, w, [1, 1, 1, 1], 'SAME'))
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    l1 = tf.nn.dropout(l1, p_keep_conv)

    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2, [1, 1, 1, 1], 'SAME'))
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_conv)

    l3a = tf.nn.relu(tf.nn.conv2d(l2, w3, [1, 1, 1, 1], 'SAME'))
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])
    l3 = tf.nn.dropout(l3, p_keep_conv)

    l4 = tf.nn.relu(tf.matmul(l3, w4))
    l4 = tf.nn.dropout(l4, p_keep_hidden)

    pyx = tf.matmul(l4, w_o)
    return pyx
'''
parser = jaffe_parser.Jaffee_Parser()

X_data = parser.images_to_tensor()
Y_data = parser.text_to_one_hot()



split_index = int(len(X_data)*.8)

X_tr = X_data[:split_index]
X_te = X_data[split_index:]

Y_tr = Y_data[:split_index]
Y_te = Y_data[split_index:]
'''
parser = fer_parser.Fer_Parser()
X_tr, Y_tr, X_te, Y_te = parser.parse_all()
X_tr = X_tr
Y_tr = Y_tr

image_dim = 48

X = tf.placeholder("float", [None, image_dim, image_dim, 1])
Y = tf.placeholder("float", [None, 7])

w = init_weights([6, 6, 1, image_dim])
w2 = init_weights([6, 6, image_dim,2*image_dim])
w3 = init_weights([6, 6, 2*image_dim, 4*image_dim])
w4 = init_weights([image_dim*144, 1250])
w_o = init_weights([1250, 7])

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
py_x = model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x,1)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

num_iterations = 10

train_correctness = []
test_correctness = []
fig = plt.figure()
ax = fig.add_subplot(111)
Ln, = ax.plot(train_correctness)
Ln2, = ax.plot(test_correctness)
ax.autoscale(enable = 'True', axis = 'both', tight = None)

plt.ion()
plt.show()


print 'Training model...'
print ''
for i in range(num_iterations):
    minibatch_size = 128
    test_batch_size = 256
    subbatch_count = 1
    for start, end in zip(range(0, len(X_tr), minibatch_size), range(128, len(X_tr),minibatch_size)):
    
        if subbatch_count % 100 == 0:
            print subbatch_count
        sess.run(train_op, feed_dict={X:X_tr[start:end], Y:Y_tr[start:end],
                                      p_keep_conv: 0.8, p_keep_hidden: 0.5})

    test_indices = np.arange(len(X_te)) # Get A Test Batch
    np.random.shuffle(test_indices)
    test_indices = test_indices[0:test_batch_size]


    train_eval_indices = np.arange(len(X_tr))
    np.random.shuffle(train_eval_indices)
    train_eval_indices = train_eval_indices[0:test_batch_size]

    print 'Iteration: ' + str(i)
    train_correctness_iter= np.mean(np.argmax(Y_tr[train_eval_indices], axis=1) ==
                     sess.run(predict_op, feed_dict={X: X_tr[train_eval_indices],
                                                     Y: Y_tr[train_eval_indices],
                                                     p_keep_conv: 1.0,
                                                     p_keep_hidden: 1.0}))
    print 'Train correctness:'
    print train_correctness_iter

    test_correctness_iter = np.mean(np.argmax(Y_te[test_indices], axis=1) ==
                     sess.run(predict_op, feed_dict={X: X_te[test_indices],
                                                     Y: Y_te[test_indices],
                                                     p_keep_conv: 1.0,
                                                     p_keep_hidden: 1.0}))
    print 'Test correctness:'
    print test_correctness_iter

    train_correctness.append(train_correctness_iter)
    test_correctness.append(test_correctness_iter)
    Ln.set_ydata(train_correctness)
    Ln.set_xdata(range(len(train_correctness)))
    Ln2.set_ydata(test_correctness)
    Ln2.set_xdata(range(len(train_correctness)))
    ax.relim()
    ax.autoscale_view()


    plt.draw()

    print ''
