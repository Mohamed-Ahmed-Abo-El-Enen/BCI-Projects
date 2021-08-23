from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.optimizers import RMSprop
from matplotlib import pyplot as plt


def autoencoder(input_img):
    # encoder
    # input = 28 x 28 x 1 (wide and thin)
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_img)
    # 14 x 14 x 32
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # #14 x 14 x 64
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(pool1)
    # 7 x 7 x 64
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # 7 x 7 x 128 (small and thick)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(pool2)

    # decoder
    # 7 x 7 x 128
    conv4 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(conv3)
    # 14 x 14 x 128
    up1 = UpSampling2D(size=(2, 2))(conv4)
    # 14 x 14 x 64
    conv5 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(up1)
    # 28 x 28 x 64
    up2 = UpSampling2D((2, 2))(conv5)
    # 28 x 28 x 1
    decoded = Conv2D(filters=1, kernel_size=(3, 3), activation='sigmoid', padding='same')(up2)
    return decoded


def fit_model(input_img, batch_size, epochs, train_x, train_ground, valid_x, valid_ground):
    autoencoder_model = Model(input_img, autoencoder(input_img))
    autoencoder_model.compile(loss='mean_squared_error', optimizer=RMSprop())
    autoencoder_model.summary()

    autoencoder_train = autoencoder_model.fit(train_x, train_ground, batch_size=batch_size, epochs=epochs, verbose=1,
                                              validation_data=(valid_x, valid_ground))
    return autoencoder_train, autoencoder_model


def plot_loss_validation(autoencoder_train, epochs):
    loss = autoencoder_train.history['loss']
    val_loss = autoencoder_train.history['val_loss']
    epochs = range(epochs)
    plt.figure()
    plt.plot(epochs, loss, 'bo', label="Training loss")
    plt.plot(epochs, val_loss, 'b', label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.show()


def plot_test_reconstruction_images(predict_result, label_dict, test_data, test_labels):
    plt.figure(figsize=(20, 4))
    print("Test Images")
    for i in range(10):
        plt.subplot(2, 10, i + 1)
        plt.imshow(test_data[i, ..., 0], cmap='gray')
        curr_lbl = test_labels[i]
        plt.title("(Label: " + str(label_dict[curr_lbl]) + ")")
    plt.show()

    plt.figure(figsize=(20, 4))
    print("Reconstruction of Test images")
    for i in range(10):
        plt.subplot(2, 10, i + 1)
        plt.imshow(predict_result[i, ..., 0], cmap='gray')
    plt.show()


def predict(autoencoder_model, label_dict, test_data, test_labels):
    predict_result = autoencoder_model.predict(test_data)
    print(predict_result.shape)
    plot_test_reconstruction_images(predict_result, label_dict, test_data, test_labels)
    return predict_result


def run_experiment(input_img, batch_size, epochs, train_x, train_ground, valid_x, valid_ground,
                   label_dict, test_data, test_labels):
    autoencoder_train, autoencoder_model = fit_model(input_img, batch_size, epochs, train_x, train_ground, valid_x,
                                                     valid_ground)
    plot_loss_validation(autoencoder_train, epochs)
    predict(autoencoder_model, label_dict, test_data, test_labels)