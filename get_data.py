from config import *


class AudioData(object):

    def __init__(self):
        self.X_train, self.X_test, self.y_train, self.y_test = [], [], [], []

    def mfcc(self, wav_file, label):
        """
        Converts wav file to mfcc and appends it with the label
        :param wav_file: path to the wav file
        :param label: np array
        :return: mfcc_data: np array with mfcc data appended with label column
        """
        label = np.array(label)
        X, sample_rate = librosa.load(wav_file, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=NUM_MFCC_SAMPLES)
        mfccs = np.mean(mfccs.T, axis=0)
        mfcc_data = np.concatenate((mfccs, label))
        return mfcc_data

    def read_aud(self, audio_type_path):
        """
        :param audio_type_path: path to specific audio type
        :return: wav_files : list of paths to wav_files of audio_type
        """
        wav_files = []
        for dirpath, dirname, files in os.walk(audio_type_path):
            wav_files = [os.path.join(dirpath, f) for f in files if fnmatch(f, "*.wav")]
        return wav_files

    def process_train_test_data(self, parent_path, audio_classes=[], test_size=0.33):
        """
        Generates train test data
        :param parent_path:
        :param audio_classes:
        :return: None
        """
        num_classes = len(audio_classes)
        if num_classes < 2:
            print("Error: Min 2 classes required.")

        data = np.empty(shape=[0, NUM_MFCC_SAMPLES + num_classes])
        for label, audio_class in enumerate(audio_classes):
            try:
                audio_path = os.path.join(parent_path, audio_class)
            except IOError:
                print("Error: Invalid path - invalid class: {}".format(audio_class))
                exit(1)

            wav_files = self.read_aud(audio_path)

            # converting the label to one hot encoding
            # label = 1 [1] [1 0 0]
            # label = 2 [2] [0 1 0]
            # label = 3 [3] [0 0 1]
            wav_label = np.zeros(num_classes)
            wav_label[label] = 1
            # print(wav_label)
            wav_data = np.array([self.mfcc(wav_file=wav_f, label=wav_label) for wav_f in wav_files])
            # print(wav_data.shape)
            data = np.concatenate((data, wav_data))

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data[:, :NUM_MFCC_SAMPLES],
                                                                                data[:, NUM_MFCC_SAMPLES:],
                                                                                test_size=test_size)

    def get_next_train_batch(self):
        """
        Batches on training data in circular fashion
        :return: (X_batch, y_batch) feature batch and label batch
        """
        batch_gen = self.get_next_train_batch()
        X_batch, y_batch = next(batch_gen)
        return X_batch, y_batch

    def train_batch_gen(self):
        """
        Batches on training data in circular fashion
        :return: (X_batch, y_batch) feature batch and label batch
        """
        #if self.X_train:
            #raise ValueError("Missing training data!")
        #if not self.y_train:
            #raise ValueError("Missing label data!")
        start_indx = 0
        num_train_entries = len(self.X_train)

        while True:
            if start_indx > num_train_entries - BATCH_SIZE:
                X_batch = np.concatenate(
                    (self.X_train[start_indx:], self.X_train[:BATCH_SIZE - (num_train_entries - start_indx)]))
                y_batch = np.concatenate(
                    (self.y_train[start_indx:], self.y_train[:BATCH_SIZE - (num_train_entries - start_indx)]))

            else:
                X_batch = self.X_train[start_indx: start_indx + BATCH_SIZE]
                y_batch = self.y_train[start_indx: start_indx + BATCH_SIZE]
            yield (X_batch, y_batch)
            start_indx = (start_indx + BATCH_SIZE) % num_train_entries

    def get_test_data(self):
        """
        :return:
        """
        return (self.X_test, self.y_test)

def main():
    wave_data = AudioData()
    wave_data.process_train_test_data(train_data_path, audio_classes)

    g = wave_data.train_batch_gen()
    for _ in range(10):
        x,y = next(g)
        print(x,y)
        print("--"*10)



if __name__ == "__main__":
    main()
