import numpy as np
import gzip
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

# create dictionary of target classes
label_dict = {
        0: 'A',
        1: 'B',
        2: 'C',
        3: 'D',
        4: 'E',
        5: 'F',
        6: 'G',
        7: 'H',
        8: 'I',
        9: 'J',
    }


def extract_data(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(28 * 28 * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, 28, 28)
        return data


def extract_labels(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        return labels


def get_dataset():
    train_data = extract_data('Dataset/train-images-idx3-ubyte.gz', 60000)
    test_data = extract_data('Dataset/t10k-images-idx3-ubyte.gz', 10000)
    train_labels = extract_labels('Dataset/train-labels-idx1-ubyte.gz', 60000)
    test_labels = extract_labels('Dataset/t10k-labels-idx1-ubyte.gz', 10000)
    # shape of training set
    print("Training set (images) shape: {shape}".format(shape=train_data.shape))
    # shape of testing set
    print("Test set (images) shape: {shape}".format(shape=test_data.shape))
    return train_data, train_labels, test_data, test_labels


def plot_train_dataset(train_data, train_labels):
    plt.figure()
    # Display the first image in training data
    curr_img = np.reshape(train_data[0], (28, 28))
    curr_lbl = train_labels[0]
    plt.imshow(curr_img, cmap='gray')
    plt.title("(Label: " + str(label_dict[curr_lbl]) + ")")
    plt.show()


def plot_test_dataset(test_data, test_labels):
    # Display the first image in testing data
    plt.figure()
    curr_img = np.reshape(test_data[0], (28, 28))
    curr_lbl = test_labels[0]
    plt.imshow(curr_img, cmap='gray')
    plt.title("(Label: " + str(label_dict[curr_lbl]) + ")")
    plt.show()


def reshape_train_test_dataset(train_data, test_data):
    train_data = train_data.reshape(-1, 28, 28, 1)
    test_data = test_data.reshape(-1, 28, 28, 1)
    print(train_data.shape, test_data.shape)
    print(train_data.dtype, test_data.dtype)
    print(np.max(train_data), np.max(test_data))
    return train_data, test_data


def rescale_train_test_dataset(train_data, test_data):
    train_data = train_data / np.max(train_data)
    test_data = test_data / np.max(test_data)
    print(np.max(train_data), np.max(test_data))
    return train_data, test_data


def split_data_set(train_data):
    train_x, valid_x, train_ground, valid_ground = train_test_split(train_data, train_data, test_size=0.2,
                                                                    random_state=13)
    print(train_x.shape, valid_x.shape, train_ground.shape, valid_ground.shape)
    return train_x, train_ground, valid_x, valid_ground


def get_train_test_dataset():
    train_data, train_labels, test_data, test_labels = get_dataset()
    plot_train_dataset(train_data, train_labels)
    plot_test_dataset(test_data, test_labels)
    train_data, test_data = reshape_train_test_dataset(train_data, test_data)
    train_data, test_data = rescale_train_test_dataset(train_data, test_data)
    train_x, train_ground, valid_x, valid_ground = split_data_set(train_data)
    return train_x, train_ground, valid_x, valid_ground, label_dict, test_data, test_labels

