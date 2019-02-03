from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers import BatchNormalization, Convolution2D , MaxPooling2D
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam

import numpy as np
import pandas as pd


def normalize(X):
    """
    Return normalized version of X with a mean of 0 and std of 1
    :param X: input data features of arbitrary shape
    :return: normalized X
    """
    mean_x = np.mean(X, axis=0)
    std_x = np.var(X, axis=0) ** 0.5
    return (X - mean_x) / (1e-6 + std_x), mean_x, std_x


def create_model(params):
    """
    Create model and return it.
    :return:
    """
    # create model
    model = Sequential()

    # adding conv layers with batch norm and maxpooling with size of 2 (keras default)
    model.add(Convolution2D(params["conv_width"][0], params["kernels"][0], input_shape=(28, 28, 1),
                    kernel_initializer='normal', activation='relu'))
    model.add(BatchNormalization(axis=1))
    model.add(MaxPooling2D())
    for l in range(1, len(params["conv_width"])):
        model.add(Dropout(params["rate"]))
        model.add(Convolution2D(params["conv_width"][l], params["kernels"][l],
                                kernel_initializer='normal', activation='relu'))
        model.add(BatchNormalization(axis=1))
        model.add(MaxPooling2D())


    # flattenning output of last conv layer and adding fully connected layers
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(10, activation='softmax'))

    return model


def get_hyperparams():

    # specifying parameters
    params = dict()
    params["conv_width"] = [30, 30]  # number of filters in each conv layer
    params["kernels"] = [(3, 3), (5, 5)]  # size of kernels used in each conv layer, len shd be same as conv_width
    params["alpha"] = 0.001   # learning rate
    params["lambda"] = 0.001  # weight penalty coefficient
    params["mbs"] = 64   # mini-batch size
    params["num_epochs"] = 10  # number of training epochss
    params["rate"] = 0.1 # dropout rate
    return params


if __name__ == "__main__":

    params = get_hyperparams()

    train_images = pd.read_csv("train.csv")
    test_images = pd.read_csv("test.csv")

    # not much image processing is needed, just normalize the images and pass on to a classifier
    train_labels = train_images["label"]
    train_images.drop(["label"], axis=1, inplace=True)

    # converting images from grayscale to binary
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # reshaping images to 28 x 28 x 1 matrices, last dimension is number of channels
    X_train = train_images.values.astype('float32').reshape(train_images.shape[0], 28, 28, 1)
    X_test = test_images.values.astype('float32').reshape(test_images.shape[0], 28, 28, 1)
    y_train = train_labels.values.astype('int32')

    # lets look at a few images to make sure everything is ok
    img_idx = np.random.choice(range(X_train.shape[0]), 3)
    for i in range(len(img_idx)):
        plt.subplot(1, 3, i + 1)
        plt.imshow(X_train[img_idx[i]][:, :, 0], cmap=plt.get_cmap('gray'))
        plt.title(y_train[img_idx[i]])
    plt.show()

    # minor data processing, normalizing features and one-hot encoding of labels
    X_train_normal, mean_X, std_X = normalize(X_train)
    X_test = (X_test - mean_X) / std_X
    y_train = to_categorical(y_train)

    # creating and compiling model
    model = create_model(params)
    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    # training!
    gen = ImageDataGenerator()
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=0.80, random_state=0)
    train_batches = gen.flow(X_train, y_train, batch_size=params["mbs"])
    val_batches = gen.flow(X_val, y_val, batch_size=params["mbs"])
    train_record = model.fit_generator(generator=train_batches, steps_per_epoch=train_batches.n,
                                       epochs=params["num_epochs"], validation_data=val_batches,
                                       validation_steps=val_batches.n)

    # submission
    predictions = model.predict_classes(X_test, verbose=0)

    submissions = pd.DataFrame({"ImageId": list(range(1, len(predictions) + 1)),
                                "Label": predictions})
    submissions.to_csv("DR.csv", index=False, header=True)




