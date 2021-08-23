from LoadData import get_train_test_dataset
from keras.layers import Input
from AutoencodeModel import run_experiment


if __name__ == "__main__":
    train_x, train_ground, valid_x, valid_ground, label_dict, test_data, test_labels = get_train_test_dataset()
    batch_size = 128
    epochs = 50
    inChannel = 1
    x, y, = 28, 28
    input_img = Input(shape=(x, y, inChannel))
    run_experiment(input_img, batch_size, epochs, train_x, train_ground, valid_x, valid_ground,
                   label_dict, test_data, test_labels)



