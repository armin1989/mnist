import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support


def get_kfold_result(features, labels, nn, folds):
    """
    Return an estimate of classification error, or accuracy, precision, recall or any other measure using k-fold
    cross-validation.

    This function is used in a grid search wrapper for tuning the hyperparameters

    :param features:  Dataframe containing training featurs
    :param labels: Dataframe containing training labels
    :param nn: number of nearest neighbours to use
    :param folds: number of splits of data to use for cross validation
    :return: performance measure, accuracy is used here
    """

    kf = KFold(n_splits=folds, shuffle=True)
    results = []
    for train_index, valid_index in kf.split(features):
        train_feats, train_labels = features.loc[train_index], labels.loc[train_index]
        valid_feats, valid_labels = features.loc[valid_index], labels.loc[valid_index]

        model = KNeighborsClassifier(n_neighbors=nn)
        model.fit(train_feats, train_labels)
        # predictions = model.predict(valid_feats)
        # precision, recall, fscore, _ = precision_recall_fscore_support(valid_labels, predictions)
        accuracy = model.score(valid_feats, valid_labels)
        results.append(accuracy)

    return np.mean(results)


if __name__ == "__main__":
    n_folds = 5   # k in k-fold cross validation

    train_images = pd.read_csv("train.csv")

    # not much image processing is needed, just normalize the images and pass on to a classifier
    labels = train_images["label"]
    images = train_images.drop(["label"], axis=1)

    # converting images from grayscale to binary
    images = images / 255.0
    max_accuracy = 0
    for neighs in range(4, 10):
        res = get_kfold_result(images, labels, neighs, n_folds)
        print("With nn = {} Accuracy = {}".format(neighs, res))
        print("*" * 80 + "\n")

        if res[0] > max_accuracy:
            max_accuracy = res
            best_nn = neighs

    print("Max accuracy = {} with nn = {}".format(max_accuracy, best_nn))