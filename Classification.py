from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import cross_val_predict

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
    print("7. FFT average over 1s window")

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
                    cur_feature = vid.get_windowed_fft(vid.rate)

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

    use_cv = int(input("Enter 1 to use Stratified Kfold CV"))

    features, targets = get_feature_choice_cmd()

    numtest = {}
    train_t = []
    test_t = []
    test_ind = []
    train_ind = []

    print("*****Starting to Test*****")
    if use_cv != 1:
        for i in range(0, len(targets)):
            t = targets[i]
            if t not in numtest:
                numtest[t] = 0
            if numtest[t] < 2:
                test_ind.append(i)
                test_t.append(targets[i])
                numtest[t] += 1
            else:
                train_ind.append(i)
                train_t.append(targets[i])

        train_f = features[train_ind]
        # train_t = targets[train_ind]
        test_f = features[test_ind]
        # test_t = targets[test_ind]

        # train_f = features
        # train_t = targets
        # test_f = features
        # test_t = targets

        # clf = clf.fit(features[: int(len(features)/10)], targets[: int(len(features)/10)])
        # predictions = clf.predict(features[- int(len(features)/10):], targets[- int(len(features)/10):])

        clf = clf.fit(train_f, train_t)
        predictions = clf.predict(test_f)

        print("Accuracy is : " + str(accuracy_score(test_t, predictions)))
        print("----------------------------")
        print("Confusion Matrix: ")
        print(confusion_matrix(test_t, predictions))
        print("\n\n")
    else:
        skf = StratifiedKFold(targets, n_folds=5)

        result = cross_val_score(clf, features, targets, cv=skf)

        print("Accuracy: %0.2f (+/- %0.2f)" % (result.mean(), result.std() * 2))

        preds = cross_val_predict(clf, features, targets, cv=skf)
        # cor_preds = targets[skf]

        for train_i, test_i in skf:
            # print("Predicted: " + preds[i] + "\t|\tCorrect Class: " + cor_preds[i])
            cv_target = [targets[x] for x in test_i]
            print("Accuracy is : " + str(accuracy_score(cv_target, preds[test_i])))
            print("----------------------------")
            print("Confusion Matrix: ")
            print(confusion_matrix(cv_target, preds[test_i]))
            print("\n\n")

if __name__ == "__main__":
    main()

# D:\Documents\DT228_4\FYP\Datasets\Test\
# D:\Documents\DT228_4\FYP\Datasets\080327\0_Audio\
