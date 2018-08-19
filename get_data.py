from config import *

class AudioData(object):

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

    def get_train_test_data(self, parent_path, audio_classes=[]):

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
            #print(wav_label)
            wav_data = np.array([self.mfcc(wav_file=wav_f, label=wav_label) for wav_f in wav_files[:3]])
            #print(wav_data.shape)
            data = np.concatenate((data, wav_data))

        #print(data.shape)
        return data


def main():

    num_classes = len(audio_classes)

    wave_data = AudioData()
    data = wave_data.get_train_test_data(train_data_path, audio_classes)

    X_train, X_test, y_train, y_test = train_test_split(data[:, :NUM_MFCC_SAMPLES], data[:, NUM_MFCC_SAMPLES: ], test_size=0.33)
    X_train = X_train.reshape(-1, NUM_MFCC_SAMPLES)
    X_test = X_test.reshape(-1, NUM_MFCC_SAMPLES)
    y_train = y_train.reshape(-1, num_classes)
    y_test = y_test.reshape(-1, num_classes)

    print(y_train,y_test)


if __name__ == "__main__":
    main()
