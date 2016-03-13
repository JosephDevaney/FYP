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
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest

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
    f_max_len = {}
    classes = {}

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
                                    f_max_len[i] = f_shape[1]
                                else:
                                    f_max_len[i] = len(f_shape)

                        start = False
                        # classes = np.array(vid.get_category_from_name())
                        classes[i] = [vid.get_category_from_name()]

                    else:
                        for i in ftr:
                            if hasattr(cur_feature[i], "__len__"):
                                if len(cur_feature[i].shape) > 1:
                                    if cur_feature[i].shape[1] > f_max_len[i]:
                                        if len(features[i].shape) > 1:
                                            features[i] = np.pad(features[i],
                                                                 ((0, 0), (0, cur_feature[i].shape[1] - f_max_len[i])),
                                                                 mode="constant")
                                            f_max_len[i] = cur_feature[i].shape[1]
                                        else:
                                            features[i] = np.pad(features[i], (0, cur_feature[i].shape[1] - f_max_len[i]),
                                                                 mode="constant")
                                            f_max_len[i] = cur_feature[i].shape[1]

                                    elif cur_feature[i].shape[1] < f_max_len[i]:
                                        cur_feature[i] = np.pad(cur_feature[i],
                                                                ((0, 0), (0, f_max_len[i] - cur_feature[i].shape[1])),
                                                                mode="constant")

                                elif len(cur_feature[i].shape) == 1:
                                    if cur_feature[i].shape[0] > f_max_len[i]:
                                        if len(features[i].shape) > 1:
                                            features[i] = np.pad(features[i],
                                                                 ((0, 0), (0, cur_feature[i].shape[0] - f_max_len[i])),
                                                                 mode="constant")
                                            f_max_len[i] = cur_feature[i].shape[0]
                                        else:
                                            features[i] = np.pad(features[i], (0, cur_feature[i].shape[0] - f_max_len[i]),
                                                                 mode="constant")
                                            f_max_len[i] = cur_feature[i].shape[0]

                                    elif cur_feature[i].shape[0] < f_max_len[i]:
                                        cur_feature[i] = np.pad(cur_feature[i], (0, f_max_len[i] - cur_feature[i].shape[0]),
                                                                mode="constant")

                            features[i] = np.vstack((features[i], [cur_feature[i]]))
                        # classes = np.append(classes, [vid.get_category_from_name()])
                        classes[i].append(vid.get_category_from_name())

            except EOFError:
                print("EOF")
                break
            except TypeError:
                print("Unable to load object")
            except pkl.UnpicklingError:
                print("Unable to load object2")

    # feat2 = {}
    # for i in range(0, len(ftr)):
    #     if True:
    #         size = int(features[ftr[i]].shape[1] / 10)
    #         feat2[ftr[i]] = SelectKBest(score_func=f_classif, k=size).fit_transform(features[ftr[i]], classes[ftr[i]])

    select_ind = []

    total_feature = features[ftr[0]]
    if True:
        select_ind = [(0,len(total_feature[0]))]
    for i in range(1, len(ftr)):
        if True:
            select_ind.append((len(total_feature[0]), len(total_feature[0]) + len(features[ftr[i]][0])))
        total_feature = np.hstack((total_feature, features[ftr[i]]))
    # targets = np.array(classes)
    targets = [value for key, value in classes.items()]

    return total_feature, targets[0], select_ind


def feature_selection(features, select_ind, targets):
    start_sel = True
    last_ind = 0
    for inds in select_ind:
        if start_sel:
            if inds[0] > 0:
                feat2 = features[:, 0:inds[0]]
                total_supp = np.arange(0, inds[0])
                start_sel = False
        elif inds[0] > last_ind:
            feat2 = np.hstack((feat2, features[:, last_ind:inds[0]]))
            total_supp = np.hstack((total_supp, np.arange(last_ind, inds[0])))

        size = (inds[1] - inds[0]) / 10
        skb = SelectKBest(score_func=f_classif, k=size)
        f_select = skb.fit_transform(features[:, inds[0]:inds[1]], targets)
        f_supp = skb.get_support(indices=True)
        f_supp += last_ind
        if start_sel:
            feat2 = f_select
            total_supp = f_supp
            start_sel = False
        else:
            feat2 = np.hstack((feat2, f_select))
            total_supp = np.hstack((total_supp, f_supp))

        last_ind = inds[1]

    return feat2, total_supp


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

    features, targets, select_ind = get_feature_choice_cmd(ftr=ftr_choice, path=path, cls=cls_choice, win_len=win_len)

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

        # result = cross_val_score(clf, features, targets, cv=skf)

        filename = time.strftime("Tests\\%Y_%m_%d__%H_%M_%S", time.gmtime()) + '.txt'
        savefile = open(filename, 'w+')

        if opts is not []:
            savefile.write("Options for this experiment are as follows: \n")
            savefile.write(str(opts) + '\n\n')

        # savefile.write("Total Accuracy: %0.2f (+/- %0.2f)\n\n" % (result.mean(), result.std() * 2))

        # preds = cross_val_predict(clf, features, targets, cv=skf)
        # cor_preds = targets[skf]

        start_cls = True

        total_targets = []
        total_preds = []
        for train_i, test_i in skf:
            # print("Predicted: " + preds[i] + "\t|\tCorrect Class: " + cor_preds[i])
            train_target = [targets[x] for x in train_i]
            train_feats, train_supp = feature_selection(features[train_i], select_ind, train_target)

            clf = clf.fit(train_feats, train_target)

            test_target = [targets[x] for x in test_i]
            # test_feats, test_supp = feature_selection(features[test_i], select_ind, test_target)
            test_feats = features[test_i, :]
            test_feats = test_feats[:, train_supp]
            preds = clf.predict(test_feats)
            sc = accuracy_score(test_target, preds)
            cm = confusion_matrix(test_target, preds)
            if start_cls:
                total_confusion = cm
                total_score = sc
                start_cls = False
            else:
                total_confusion += cm
                total_score += sc

            total_targets.extend(test_target)
            total_preds.extend(preds)
            acc = str(accuracy_score(test_target, preds))

            # savefile.write("Accuracy is : " + acc)
            # savefile.write("\n----------------------------\n")
            # savefile.write("Confusion Matrix: \n")
            # savefile.write(str(cm))
            # savefile.write('\n\n')
            #
            # savefile.write("%-25s %s\n" % ("Target", "Prediction"))
            # savefile.write("----------------------------------\n")
            # [savefile.write("%-25s %s\n" % (c1, c2)) for c1, c2 in zip(test_target, preds[test_i])]
            # savefile.write('\n\n')

        avg_score = total_score / 3
        savefile.write("Total Accuracy: %0.2f \n\n" % avg_score)

        savefile.write("Summed CMs\n--------------------\n")
        savefile.write(str(total_confusion))
        # savefile.write("\n\nFinal CM from aggregate targets\n-----------------------------\n")
        # savefile.write(str(confusion_matrix(total_targets, total_preds)))

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
