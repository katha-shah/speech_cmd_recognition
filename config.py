import os
import librosa
from fnmatch import fnmatch
import numpy as np
from sklearn.model_selection import train_test_split

train_data_path = 'data/train/audio'
audio_classes = ['yes', 'no', 'go']

NUM_MFCC_SAMPLES = 40