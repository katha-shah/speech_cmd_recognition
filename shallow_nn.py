from config import *
from get_data import AudioData

train = True

ckpt_dir = "ckpt_dir"
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

X = tf.placeholder(dtype=tf.float32, shape=[None, NUM_MFCC_SAMPLES])
y = tf.placeholder(dtype=tf.float32, shape=[None, num_classes])


def nn_model(X, weights, biases):
    X = tf.reshape(X, shape=[-1, NUM_MFCC_SAMPLES])

    z1 = tf.matmul(X, weights['w1']) + biases['b1']
    z1 = tf.nn.relu(z1)

    z2 = tf.matmul(z1, weights['w2']) + biases['b2']
    z2 = tf.nn.relu(z2)

    out = tf.matmul(z2, weights['out']) + biases['out']

    return out


def get_parameters():
    weights = {'w1': tf.Variable(tf.random_normal([NUM_MFCC_SAMPLES, 64])),
               'w2': tf.Variable(tf.random_normal([64, 128])),
               'out': tf.Variable(tf.random_normal([128, num_classes])),
               }

    biases = {'b1': tf.Variable(tf.random_normal([64])),
              'b2': tf.Variable(tf.random_normal([128])),
              'out': tf.Variable(tf.random_normal([num_classes])),
              }

    return weights, biases


wave_data = AudioData()

wave_data.process_train_test_data(parent_path=train_data_path, audio_classes=audio_classes)
weights, biases = get_parameters()
logits = nn_model(X, weights, biases)
ypred = tf.nn.softmax(logits)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(ypred, 1), tf.argmax(y, 1)), tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()
global_epoch = tf.Variable(0, dtype=tf.int32, name="global_epoch", trainable=False)

with tf.Session() as sess:
    sess.run(init)
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print(ckpt.model_checkpoint_path)
        # saver.restore(sess, ckpt.model_checkpoint_path)
    start_epoch = 0  # global_epoch.eval()
    if train:
        train_batch_gen = wave_data.train_batch_gen()
        performance = []
        for epoch in range(start_epoch, start_epoch + epochs):
            batch_x, batch_y = next(train_batch_gen)
            sess.run(optimizer, feed_dict={X: batch_x, y: batch_y})

            if epoch % 10 == 0:
                l, a = sess.run([loss, accuracy], feed_dict={X: batch_x, y: batch_y})
                performance.append((l,a))
                print("At epoch {}, minibatch loss = {:.4f} | accuracy = {:.3f}".format(epoch, l, a))
        print("Training done, saving parameters!")
        epoch_plot = list(range(0,len(performance)*10, 10))
        #plt.plot(performance, epoch_plot)
        #plt.show()
        # global_epoch.assign(epoch).eval
        #saver.save(sess, ckpt_dir + "/nn.ckpt")

    X_test, y_test = wave_data.get_test_data()
    test_accuracy = sess.run(accuracy,
                             feed_dict={X: X_test, y: y_test})

    print("Test_accuracy = {:.3f}".format(test_accuracy))
