from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.optimizers import RMSprop
from matplotlib import pyplot as plt


def denoisnig_autoencoder(input_img):
    # encoder
    # 32-3 x 3
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_img)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # 64-3 x 3
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # 128-3 x 3
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(pool2)

    # dcoder
    # 128-3 x 3
    conv4 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(conv3)
    up1 = UpSampling2D(size=(2, 2))(conv4)
    # 64-3 x 3
    conv5 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu',  padding='same')(up1)
    up2 = UpSampling2D(size=(2, 2))(conv5)
    # 1-3 x 3
    decoded = Conv2D(filters=1, kernel_size=(3, 3), activation='sigmoid', padding='same')(up2)
    return decoded


def fit_model(input_img, batch_size, epochs, x_train_noisy, train_x, valid_x, valid_ground):
    denoisnig_autoencoder_model = Model(input_img, denoisnig_autoencoder(input_img))
    denoisnig_autoencoder_model.compile(loss='mean_squared_error', optimizer=RMSprop())
    denoisnig_autoencoder_model.summary()

    autoencoder_train = denoisnig_autoencoder_model.fit(x_train_noisy, train_x, batch_size=batch_size, epochs=epochs,
                                                        verbose=1, validation_data=(valid_x, valid_ground))
    return autoencoder_train, denoisnig_autoencoder_model


def plot_loss_validation(denoisnig_autoencoder_train, epochs):
    loss = denoisnig_autoencoder_train.history['loss']
    val_loss = denoisnig_autoencoder_train.history['val_loss']
    epochs = range(epochs)
    plt.figure()
    plt.plot(epochs, loss, 'bo', label="Training loss")
    plt.plot(epochs, val_loss, 'r+', label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.show()


def plot_test_reconstruction_images(label_dict, predict_result, test_data, x_test_noisy, test_labels):
    plt.figure(figsize=(20, 4))
    print("Test Images")
    for i in range(10, 20, 1):
        plt.subplot(2, 10, i+1)
        plt.imshow(test_data[i, ..., 0], cmap='gray')
        curr_lbl = test_labels[i]
        plt.title("(Label: "+str(label_dict[curr_lbl])+")")
    plt.show()
    plt.figure(figsize=(20, 4))
    print("Test Images with Noise")
    for i in range(10, 20, 1):
        plt.subplot(2, 10, i+1)
        plt.imshow(x_test_noisy[i, ..., 0], cmap='gray')
    plt.show()
    plt.figure(figsize=(20, 4))
    print("Reconstruction of Noisy Test Images")
    for i in range(10, 20, 1):
        plt.subplot(2, 10, i+1)
        plt.imshow(predict_result[i, ..., 0], cmap='gray')
    plt.show()


def predict(label_dict, denoisnig_autoencoder_model, test_data, x_test_noisy, test_labels):
    predict_result = denoisnig_autoencoder_model.predict(x_test_noisy)
    print(predict_result.shape)
    plot_test_reconstruction_images(label_dict, predict_result, test_data, x_test_noisy, test_labels)
    return predict_result


def run_experiment(input_img, label_dict, batch_size, epochs, x_train_noisy, train_x, valid_x, valid_ground,
                   x_test_noisy, test_data, test_labels):
    denoisnig_autoencoder_train, denoisnig_autoencoder_model = fit_model(input_img, batch_size, epochs, x_train_noisy,
                                                                         train_x, valid_x, valid_ground)
    plot_loss_validation(denoisnig_autoencoder_train, epochs)
    predict(label_dict, denoisnig_autoencoder_model, test_data, x_test_noisy, test_labels)
