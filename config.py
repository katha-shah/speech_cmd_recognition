import os
import librosa
from fnmatch import fnmatch
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from pprint import pprint

train_data_path = 'data/train/audio'
audio_classes = ['yes', 'no', 'go']
num_classes = len(audio_classes)

NUM_MFCC_SAMPLES = 40
learning_rate = 0.01
epochs = 100