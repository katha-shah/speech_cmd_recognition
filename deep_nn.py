from config import *
from get_data import AudioData

train = True

ckpt_dir = "ckpt_dir"
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)


X = tf.placeholder(dtype=tf.float32, shape=[None, NUM_MFCC_SAMPLES])
y = tf.placeholder(dtype=tf.float32, shape=[None, num_classes])

