import pandas as pd
import numpy as np
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support


def get_kfold_result(features, labels, c, kernel, folds):
    """
    Return an estimate of classification error, or accuracy, precision, recall or any other measure using k-fold
    cross-validation.

    This function is used in a grid search wrapper for tuning the hyperparameters

    :param features:  Dataframe containing training featurs
    :param labels: Dataframe containing training labels
    :param c: c in SVM
    :param kernel: kernel used in SVM
    :param folds: number of splits of data to use for cross validation
    :return: performance measure, accuracy is repoorted here
    """

    kf = KFold(n_splits=folds, shuffle=True)
    results = []
    for train_index, valid_index in kf.split(features):
        train_feats, train_labels = features.loc[train_index], labels.loc[train_index]
        valid_feats, valid_labels = features.loc[valid_index], labels.loc[valid_index]

        model = svm.SVC(C=c, kernel=kernel)
        model.fit(train_feats, train_labels)
        # predictions = model.predict(valid_feats)
        # precision, recall, fscore, _ = precision_recall_fscore_support(valid_labels, predictions)
        accuracy = model.score(valid_feats, valid_labels)
        results.append(accuracy)

    return np.mean(results, axis=1)


if __name__ == "__main__":

    n_folds = 5  # k in k-fold cross validation

    train_images = pd.read_csv("train.csv")

    # not much image processing is needed, just normalize the images and pass on to a classifier
    labels = train_images["label"]
    images = train_images.drop(["label"], axis=1)

    # converting images from grayscale to binary
    images = images / 255.0

    # generating train/validation split, since we have 5000 data points we may not need kfold cross validation
    train_images, test_images, train_labels, test_labels = \
        train_test_split(images, labels, train_size=0.8, random_state=0)

    max_accuracy = 0
    kernels = ["rbf", "linear"]
    for kernel in kernels:
        for c in np.arange(1, 11):
            res = get_kfold_result(images, labels, c, kernel, n_folds)
            print("With kernel = {}, c = {}\nAccuracy = {}".format(kernel, c, res))
            print("*" * 80 + "\n")

            if res > max_accuracy:
                max_accuracy = res
                best_c = c
                best_kernel = kernel

    print("Max accuracy = {} with c = {} and kernel = {}".format(max_accuracy, best_c, best_kernel))