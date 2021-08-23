from LoadData import get_train_test_dataset
from keras.layers import Input
from DenoisingAutoencoderModel import run_experiment
from NoiseAddation import add_noise

if __name__ == "__main__":
    train_x, train_ground, valid_x, valid_ground, label_dict, test_data, test_labels = get_train_test_dataset()
    x_train_noisy, x_valid_noisy, x_test_noisy = add_noise(train_x, valid_x, test_data)
    batch_size = 1000
    epochs = 1
    inChannel = 1
    x, y, = 28, 28
    input_img = Input(shape=(x, y, inChannel))
    run_experiment(input_img, label_dict, batch_size, epochs, x_train_noisy, train_x, valid_x, valid_ground,
                   x_test_noisy, test_data, test_labels)
