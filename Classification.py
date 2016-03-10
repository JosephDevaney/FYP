from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import cross_val_predict

import numpy as np
import scipy.fftpack as fft
from VideoFeatures import VideoFeatures
import pickle as pkl
import time

CONF_FILE = "config.txt"
REG_FEATS = "features.ftr"
SHORT_FEATS = "features30sec.ftr"


def get_classifier_from_cmd(cls=None):
    if cls is None:
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


def get_feature_choice_cmd(ftr=None, path=None, cls=None, win_len=None):
    if ftr is None:
        print("Please select the feature (Separated by | for multiple):")
        print("1. Beat Variance Ratio")
        print("2. Silence Ratio")
        print("3. MFCC")
        print("4. MFCC Delta")
        print("5. Chromagram")
        print("6. Spectroid")
        print("7. FFT average over 1s window")
        print("8. ZCR over window")

        ftr = input()
        ftr = [int(x) for x in ftr.split('|')]

    if path is None:
        path = input("Enter path to features: \n")

    if cls is None:
        cls = ["Entertainment", "Music", "Comedy", "Film & Animation", "News & Politics", "Sports", "People & Blogs",
               "Howto & Style", "Pets & Animals"]
    if win_len is None:
        win_len = 0.04
    start = True

    # features = np.empty(shape=(0, 0))
    features = {}
    f_max_len = 0
    classes = []

    with open(path + SHORT_FEATS, "rb") as inp:
        unpickle = pkl.Unpickler(inp)
        while True:
            try:
                cur_feature = {}
                vid = unpickle.load()
                if vid.get_category_from_name() in cls:
                    if 1 in ftr:
                        cur_feature[1] = vid.bvratio
                    if 2 in ftr:
                        cur_feature[2] = vid.silence_ratio
                    if 3 in ftr:
                        cur_feature[3] = np.array(vid.mfcc).reshape((1, -1))[0]
                    if 4 in ftr:
                        cur_feature[4] = vid.mfcc_delta
                    if 5 in ftr:
                        cur_feature[5] = vid.chromagram
                    if 6 in ftr:
                        cur_feature[6] = vid.spectroid
                    if 7 in ftr:
                        cur_feature[7] = vid.get_windowed_fft(int(np.ceil(vid.rate * win_len)))
                    if 8 in ftr:
                        cur_feature[8] = vid.get_windowed_zcr(int(np.ceil(vid.rate * win_len)))

                    if start:
                        for i in ftr:
                            features[i] = np.array([cur_feature[i]])

                            f_shape = features[i].shape
                            if hasattr(cur_feature[i], "__len__"):
                                if len(f_shape) > 1:
                                    f_max_len = f_shape[1]
                                else:
                                    f_max_len = len(f_shape)

                        start = False
                        # classes = np.array(vid.get_category_from_name())
                        classes = [vid.get_category_from_name()]

                    else:
                        for i in ftr:
                            if hasattr(cur_feature[i], "__len__"):
                                if len(cur_feature[i].shape) > 1:
                                    if cur_feature[i].shape[1] > f_max_len:
                                        if len(features[i].shape) > 1:
                                            features[i] = np.pad(features[i],
                                                                 ((0, 0), (0, cur_feature[i].shape[1] - f_max_len)),
                                                                 mode="constant")
                                            f_max_len = cur_feature[i].shape[1]
                                        else:
                                            features[i] = np.pad(features[i], (0, cur_feature[i].shape[1] - f_max_len),
                                                                 mode="constant")
                                            f_max_len = cur_feature[i].shape[1]

                                    elif cur_feature[i].shape[1] < f_max_len:
                                        cur_feature[i] = np.pad(cur_feature[i],
                                                                ((0, 0), (0, f_max_len - cur_feature[i].shape[1])),
                                                                mode="constant")

                                elif len(cur_feature[i].shape) == 1:
                                    if cur_feature[i].shape[0] > f_max_len:
                                        if len(features[i].shape) > 1:
                                            features[i] = np.pad(features[i],
                                                                 ((0, 0), (0, cur_feature[i].shape[0] - f_max_len)),
                                                                 mode="constant")
                                            f_max_len = cur_feature[i].shape[0]
                                        else:
                                            features[i] = np.pad(features[i], (0, cur_feature[i].shape[0] - f_max_len),
                                                                 mode="constant")
                                            f_max_len = cur_feature[i].shape[0]

                                    elif cur_feature[i].shape[0] < f_max_len:
                                        cur_feature[i] = np.pad(cur_feature[i], (0, f_max_len - cur_feature[i].shape[0]),
                                                                mode="constant")

                            features[i] = np.vstack((features[i], [cur_feature[i]]))
                        # classes = np.append(classes, [vid.get_category_from_name()])
                        classes.append(vid.get_category_from_name())

            except EOFError:
                print("EOF")
                break
            except TypeError:
                print("Unable to load object")
            except pkl.UnpicklingError:
                print("Unable to load object2")

    total_feature = features[ftr[0]]
    for i in range(1, len(ftr)):
        total_feature = np.hstack((total_feature, features[ftr[i]]))
    # targets = np.array(classes)
    targets = classes

    return total_feature, targets


def main():
    clf_choice = None
    ftr_choice = None
    win_len = None
    path = None
    opts = []
    cls_choice = []
    try:
        opts = [line.strip('\n') for line in open(CONF_FILE)]
        clf_choice = int(opts[0])
        use_cv = int(opts[1])
        ftr_choice = [int(x) for x in opts[2].split('|')]
        win_len = float(opts[3])
        path = opts[4]
        cls_choice = [x for x in opts[5].split('|')]
    except FileNotFoundError:
        use_cv = int(input("Enter 1 to use Stratified Kfold CV: \n"))

    clf = get_classifier_from_cmd(clf_choice)

    features, targets = get_feature_choice_cmd(ftr=ftr_choice, path=path, cls=cls_choice, win_len=win_len)

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
        skf = StratifiedKFold(targets, n_folds=3)

        result = cross_val_score(clf, features, targets, cv=skf)

        filename = time.strftime("Tests\\%Y_%m_%d__%H_%M_%S", time.gmtime()) + '.txt'
        savefile = open(filename, 'w+')

        if opts is not []:
            savefile.write("Options for this experiment are as follows: \n")
            savefile.write(str(opts) + '\n\n')

        savefile.write("Total Accuracy: %0.2f (+/- %0.2f)\n\n" % (result.mean(), result.std() * 2))

        preds = cross_val_predict(clf, features, targets, cv=skf)
        # cor_preds = targets[skf]

        start_cls = True

        total_targets = []
        total_preds = []
        for train_i, test_i in skf:
            # print("Predicted: " + preds[i] + "\t|\tCorrect Class: " + cor_preds[i])
            cv_target = [targets[x] for x in test_i]
            cm = confusion_matrix(cv_target, preds[test_i])
            if start_cls:
                total_confusion = cm
                start_cls = False
            else:
                total_confusion += cm
            total_targets.extend(cv_target)
            total_preds.extend(preds[test_i])
            acc = str(accuracy_score(cv_target, preds[test_i]))

            # savefile.write("Accuracy is : " + acc)
            # savefile.write("\n----------------------------\n")
            # savefile.write("Confusion Matrix: \n")
            # savefile.write(str(cm))
            # savefile.write('\n\n')
            #
            # savefile.write("%-25s %s\n" % ("Target", "Prediction"))
            # savefile.write("----------------------------------\n")
            # [savefile.write("%-25s %s\n" % (c1, c2)) for c1, c2 in zip(cv_target, preds[test_i])]
            # savefile.write('\n\n')

        savefile.write("Summed CMs\n--------------------\n")
        savefile.write(str(total_confusion))
        savefile.write("\n\nFinal CM from aggregate targets\n-----------------------------")
        savefile.write(str(confusion_matrix(total_targets, total_preds)))

        unique_cls = []
        [unique_cls.append(x) for x in total_targets if x not in unique_cls]

        savefile.write("\n\nREPORT!!!\n-----------------------------------\n")
        savefile.write(classification_report(total_targets, total_preds))

        savefile.write("%-25s %s\n" % ("Target", "Prediction"))
        savefile.write("----------------------------------\n")
        [savefile.write("%-25s %s\n" % (c1, c2)) for c1, c2 in zip(total_targets, total_preds)]
        savefile.write('\n\n')

        savefile.close()


if __name__ == "__main__":
    main()

# D:\Documents\DT228_4\FYP\Datasets\Test\
# D:\Documents\DT228_4\FYP\Datasets\080327\0_Audio\
