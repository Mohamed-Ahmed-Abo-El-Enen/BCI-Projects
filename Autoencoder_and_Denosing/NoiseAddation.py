import numpy as np
from matplotlib import pyplot as plt


def add_noise(train_x, valid_x, test_x):
    noise_factor = 0.5
    x_train_noisy = train_x + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=train_x.shape)
    x_valid_noisy = valid_x + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=valid_x.shape)
    x_test_noisy = test_x + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=test_x.shape)
    x_train_noisy = np.clip(x_train_noisy, 0.0, 1.0)
    x_valid_noisy = np.clip(x_valid_noisy, 0.0, 1.0)
    x_test_noisy = np.clip(x_test_noisy, 0.0, 1.0)
    visualize_noisy_data(x_train_noisy, x_test_noisy)
    return x_train_noisy, x_valid_noisy, x_test_noisy


def visualize_noisy_data(x_train_noisy, x_test_noisy):
    plt.figure(figsize=[5, 5])
    # display the first image in training data
    plt.subplot(121)
    curr_img = np.reshape(x_train_noisy[1], (28, 28))
    plt.imshow(curr_img, cmap='gray')

    # display the first image in testing data
    plt.subplot(122)
    curr_img = np.reshape(x_test_noisy[1], (28, 28))
    plt.imshow(curr_img, cmap='gray')
    plt.show()
