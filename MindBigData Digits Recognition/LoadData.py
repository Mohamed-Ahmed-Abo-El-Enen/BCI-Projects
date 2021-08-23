import zipfile
import numpy as np
from keras.preprocessing.sequence import pad_sequences


class MindBigData:
    def __init__(self):
        self.dataset_file_text_name = 'EP1.01.txt'
        self.dataset_file_zip_name = 'MindBigData-EP-v1.0.zip'
        self.dataset_path = 'MindBigData/'+self.dataset_file_zip_name
        self.device_name = "EP"
        self.device_channels = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
        self.dataset_size = 1000

    def get_dataset_file(self):
        return open(self.dataset_path, 'rb')

    def get_dataset(self):
        file = self.get_dataset_file()
        zip_file = zipfile.ZipFile(file)
        data_file = zip_file.open(self.dataset_file_text_name, 'r')
        entire_dataset = {}
        print("Reading Data File")
        for line in data_file:
            if len(entire_dataset) > self.dataset_size:
                break
            id, event, device, channel, code, size, data = line.split(b'\t')
            if self.device_name.encode('utf-8') != device:
                continue
            signal = np.array([float(val) for val in data.split(b',')])

            if event in entire_dataset:
                _, channels_signal = entire_dataset[event]
                channels_signal = np.append(channels_signal, [np.array(signal)], axis=0)
                entire_dataset[event] = _, channels_signal
            else:
                entire_dataset[event] = int(code), [np.array(signal)]
        return entire_dataset

    def get_specific_label(self, entire_dataset, label):
        result = {}
        for key in entire_dataset:
            if entire_dataset[key][0] == label:
                result[key] = entire_dataset[key]
        return result

    def get_max_feature_vector_length(self, entire_dataset):
        max_shape = -1
        for key in entire_dataset:
            shape = entire_dataset[key][1].shape[1]
            if shape > max_shape:
                max_shape = shape
        return max_shape

    def save_labe_data_file(self, data, label, pre_max_padding):
        num_samples = 0
        with open("features_dataset_for_"+str(label)+".txt", 'w') as outfile:
            outfile.write('@'+str(len(self.device_channels))+'/'+str(pre_max_padding)+'\n')
            for sample in data:
                outfile.write('#'+str(sample.decode('utf-8'))+'\n')
                sequences = pad_sequences(data[sample][1], maxlen=pre_max_padding)
                np.savetxt(outfile, sequences, fmt='%s')
                outfile.write('\n')
                num_samples += 1




