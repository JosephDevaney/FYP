from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import numpy as np
import scipy.fftpack as fft
from VideoFeatures import VideoFeatures
import pickle as pkl


def get_classifier_from_cmd():
    print("Please select the number that corresponds to the desired Classifier:")
    print("1. Decision Tree")
    print("2. Naive Bayes")
    print("3. SVM")
    print("4. kNN")

    cls = int(input())

    if cls == 1:
        classifier = tree.DecisionTreeClassifier(criterion="entropy")
    elif cls == 2:
        # NB
        classifier = GaussianNB()
    elif cls == 3:
        # SVM
        classifier = svm.SVC()
    elif cls == 4:
        classifier = NearestNeighbors()

    return classifier


def get_feature_choice_cmd():
    print("Please select the feature:")
    print("1. Beat Variance Ratio")
    print("2. Silence Ratio")
    print("3. MFCC")
    print("4. MFCC Delta")
    print("5. Chromagram")
    print("6. Spectroid")
    print("7. FFT")

    start = True
    ftr = int(input())

    path = input("Enter path to features: \n")

    features = np.empty(shape=(0, 0))
    f_max_len = 0
    classes = []

    with open(path + "features.ftr", "rb") as inp:
        unpickle = pkl.Unpickler(inp)
        while True:
            try:
                vid = unpickle.load()
                if ftr == 1:
                    cur_feature = vid.bvratio
                elif ftr == 2:
                    cur_feature = vid.silence_ratio
                elif ftr == 3:
                    cur_feature = vid.mfcc
                elif ftr == 4:
                    cur_feature = vid.mfcc_delta
                elif ftr == 5:
                    cur_feature = vid.chromagram
                elif ftr == 6:
                    cur_feature = vid.spectroid
                elif ftr == 7:
                    cur_feature = fft.rfft(vid.data)

                if start:
                    features = np.array([cur_feature])
                    # classes = np.array(vid.get_category_from_name())
                    classes = [vid.get_category_from_name()]
                    start = False

                    if hasattr(cur_feature, "__len__"):
                        if len(features.shape) > 1:
                            f_max_len = features.shape[1]
                        else:
                            f_max_len = len(features.shape)

                else:
                    if hasattr(cur_feature, "__len__"):
                        if len(cur_feature.shape) > 1:
                            if cur_feature.shape[1] > f_max_len:
                                if len(features.shape) > 1:
                                    features = np.pad(features, ((0, 0), (0, cur_feature.shape[1] - f_max_len)),
                                                      mode="constant")
                                    f_max_len = cur_feature.shape[1]
                                else:
                                    features = np.pad(features, (0, cur_feature.shape[1] - f_max_len),
                                                      mode="constant")
                                    f_max_len = cur_feature.shape[1]

                            elif cur_feature.shape[1] < f_max_len:
                                cur_feature = np.pad(cur_feature, ((0, 0), (0, f_max_len - cur_feature.shape[1])),
                                                     mode="constant")

                        elif len(cur_feature.shape) == 1:
                            if cur_feature.shape[0] > f_max_len:
                                if len(features.shape) > 1:
                                    features = np.pad(features, ((0, 0), (0, cur_feature.shape[0] - f_max_len)),
                                                      mode="constant")
                                    f_max_len = cur_feature.shape[0]
                                else:
                                    features = np.pad(features, (0, cur_feature.shape[0] - f_max_len),
                                                      mode="constant")
                                    f_max_len = cur_feature.shape[0]

                            elif cur_feature.shape[0] < f_max_len:
                                cur_feature = np.pad(cur_feature, (0, f_max_len - cur_feature.shape[0]),
                                                     mode="constant")

                    features = np.vstack((features, [cur_feature]))
                    # classes = np.append(classes, [vid.get_category_from_name()])
                    classes.append(vid.get_category_from_name())

            except EOFError:
                print("EOF")
                break
            except TypeError:
                print("Unable to load object")
            except pkl.UnpicklingError:
                print("Unable to load object2")

    # targets = np.array(classes)
    targets = classes

    return features, targets


def main():
    clf = get_classifier_from_cmd()

    features, targets = get_feature_choice_cmd()

    train_f = features[:-10]
    train_t = targets[:-10]
    test_f = features[-10:]
    test_t = targets[-10:]

    # train_f = features
    # train_t = targets
    # test_f = features
    # test_t = targets

    # clf = clf.fit(features[: int(len(features)/10)], targets[: int(len(features)/10)])
    # predictions = clf.predict(features[- int(len(features)/10):], targets[- int(len(features)/10):])

    clf = clf.fit(train_f, train_t)
    predictions = clf.predict(test_f)

    print(test_t, predictions)

    print("Accuracy is : " + str(accuracy_score(test_t, predictions)))
    print("----------------------------")
    print("Confusion Matrix: ")
    print(confusion_matrix(test_t, predictions))
    print("\n\n")


if __name__ == "__main__":
    main()

# D:\Documents\DT228_4\FYP\Datasets\Test\
# D:\Documents\DT228_4\FYP\Datasets\080327\0_Audio
