from sklearn import tree
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import cross_val_predict
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import RandomizedPCA

from hmmlearn import hmm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import numpy as np
import scipy.fftpack as fft
from VideoFeatures import VideoFeatures
import WnLFeatures as wnl
import pickle as pkl
import time
import gc


# These should be loaded from the config file
CONF_FILE = "config.txt"
REG_FEATS = "features.ftr"
SHORT_FEATS = "features30sec.ftr"


# This function takes instantiates the desired Classification object and returns it
# If the options are not loaded the user is prompted for the decision
def get_classifier_from_cmd(cls=None):
    if cls is None:
        print("Please select the number that corresponds to the desired Classifier:")
        print("1. Decision Tree")
        print("2. Naive Bayes")
        print("3. SVM")
        print("4. kNN")
        print("5. Random Forest")

        cls = int(input())

    cls_opts = [x for x in cls.split('|')]
    cls = int(cls_opts[0])
    if cls == 1:
        classifier = tree.DecisionTreeClassifier(criterion="entropy")
    elif cls == 2:
        # NB
        # classifier = GaussianNB()
        # classifier = MultinomialNB()
        classifier = BernoulliNB()
    elif cls == 3:
        # SVM
        opts = [x for x in cls_opts[1].split('$')]
        gm = 'auto'
        if opts[0] in ['rbf', 'poly', 'sigmoid']:
            if opts[2] != 'auto':
                gm = float(opts[2])
        classifier = svm.SVC(kernel=opts[0], decision_function_shape=opts[1], gamma=gm)
    elif cls == 4:
        opts = [x for x in cls_opts[1].split('$')]
        classifier = KNeighborsClassifier(n_neighbors=int(opts[0]), algorithm='ball_tree', weights=opts[1], leaf_size=int(opts[2]))
    elif cls == 5:
        classifier = RandomForestClassifier(criterion="entropy")

    return classifier


# This function builds the matrix of features per instance and returns it.
# Pads individual features with 0's to ensure each instance has the same number of columns/features
# Tracks the columns that will have a feature reduction applied and the amount this will be.
def get_feature_choice_cmd(ftr=None, ftr_sel=None, path=None, cls=None, win_len=None):
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
        print("9. MFCC over window")

        ftr = input()
        # ftr = [int(x) for x in ftr.split('|')]
        ftr = [opts.split('$')[0] for opts in ftr.split('|')]
        ftr_sel = {opts.split('$')[0]: opts.split('$')[1] for opts in ftr.split('|')}

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
    # Path is an array containing potentially multiple features files that can be used to load Video objects from disk.
    for p in path:
        with open(p + SHORT_FEATS, "rb") as inp:
            unpickle = pkl.Unpickler(inp)
            count = 0
            # Create the UnPickler object and loop until there are no objects left in the file. Break from loop then.
            while True:
                try:
                    cur_feature = {}
                    vid = unpickle.load()
                    # If video is in the approved class list add all selected features to a dictionary cur_feature
                    if vid.get_category_from_name() in cls:
                        count += 1
                        if 1 in ftr:
                            cur_feature[1] = vid.bvratio
                        if 2 in ftr:
                            cur_feature[2] = vid.silence_ratio
                        if 3 in ftr:
                            cur_feature[3] = np.array(vid.mfcc).reshape((1, -1))[0]
                        if 4 in ftr:
                            cur_feature[4] = np.array(vid.mfcc_delta).reshape((1, -1))[0]
                        if 5 in ftr:
                            cur_feature[5] = np.array(vid.chromagram).reshape((1, -1))[0]
                        if 6 in ftr:
                            cur_feature[6] = vid.spectroid[0]
                        if 7 in ftr:
                            cur_feature[7] = vid.get_windowed_fft(int(np.ceil(vid.rate * win_len)))
                        if 8 in ftr:
                            cur_feature[8] = vid.get_windowed_zcr(int(np.ceil(vid.rate * win_len)))
                        if 9 in ftr:
                            cur_feature[9] = np.array(wnl.get_window_mfcc(vid.mfcc, int(np.ceil(vid.rate * win_len)))) \
                                .reshape((1, -1))[0]

                        # This section was designed under the assumption that the features could be returned in various
                        # 2d layouts. It essentially checks the size of the current feature against the largest
                        # number of columns so far. It then pads the smaller one with 0's
                        # This can definitely be refactored into simpler, more readable code.
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
                                                features[i] = np.pad(features[i],
                                                                     (0, cur_feature[i].shape[1] - f_max_len[i]),
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
                                                features[i] = np.pad(features[i],
                                                                     (0, cur_feature[i].shape[0] - f_max_len[i]),
                                                                     mode="constant")
                                                f_max_len[i] = cur_feature[i].shape[0]

                                        elif cur_feature[i].shape[0] < f_max_len[i]:
                                            cur_feature[i] = np.pad(cur_feature[i],
                                                                    (0, f_max_len[i] - cur_feature[i].shape[0]),
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

                gc.collect()

    # Join each feature into one large array.
    # Keep track of the indices for each feature that needs reduction applied later
    select_ind = []
    print("Count = ", count)
    total_feature = features[ftr[0]]
    if ftr_sel[ftr[0]] > 0:
        select_ind = [(0, len(total_feature[0]), ftr_sel[ftr[0]])]
    for i in range(1, len(ftr)):
        if ftr_sel[ftr[i]] > 0:
            start = len(total_feature[0])
            select_ind.append((start, start + len(features[ftr[i]][0]), ftr_sel[ftr[i]]))
        total_feature = np.hstack((total_feature, features[ftr[i]]))
    # targets = np.array(classes)
    targets = [value for key, value in classes.items()]

    return total_feature, targets[0], select_ind


# Apply the feature selection to the features matrix as necessary.
def feature_selection(features, select_ind, targets):
    start_sel = True
    last_ind = 0
    feat2 = features
    for inds in select_ind:
        # If this is the first loop check if the first feature to be reduced is not the first index.
        # If so add all features up to that point to an array and track the original indices also
        # Do the same if it is not the first loop, but the next reduce index is > the last index
        if start_sel:
            if inds[0] > 0:
                feat2 = features[:, 0:inds[0]]
                total_supp = np.arange(0, inds[0])
                # total_supp = np.ones(inds[0], dtype=bool)
                start_sel = False
                last_ind = inds[0]
        elif inds[0] > last_ind:
            feat2 = np.hstack((feat2, features[:, last_ind:inds[0]]))
            total_supp = np.hstack((total_supp, np.arange(last_ind, inds[0] + 1)))
            # total_supp = np.hstack((total_supp, np.ones((inds[0]-last_ind), dtype=bool)))

        # Get the number of columns to retain and create object
        size = (inds[1] - inds[0]) / inds[2]
        skb = SelectKBest(score_func=f_classif, k=size)
        # skb = SelectPercentile(score_func=f_classif, percentile=inds[2])

        # slice out the columns relating to the current feature
        f = features[:, inds[0]:inds[1]]

        # Return an array of the selected features. Get the indices and add them to an array
        f_select = skb.fit_transform(f, targets)
        # skb.fit(f, targets)
        # f_select = skb.transform(f)
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


# Perform PCA on the desired features
def feature_reduction_fit(features, select_ind, red_pca, fit=False):
    start_sel = True
    last_ind = 0
    feat2 = features

    for inds in select_ind:
        if start_sel:
            if inds[0] > 0:
                feat2 = features[:, 0:inds[0]]
                start_sel = False
        elif inds[0] > last_ind:
            feat2 = np.hstack((feat2, features[:, last_ind:inds[0]]))

        # If this is the training set, fit the data to the object before transforming.
        # If its not then just transform
        # Create the new array and return it and the PCA objects
        if fit:
            red_pca[inds[0]].fit(features[:, inds[0]:inds[1]])
        f_reduct = red_pca[inds[0]].transform(features[:, inds[0]:inds[1]])

        if start_sel:
            feat2 = f_reduct
            start_sel = False
        else:
            feat2 = np.hstack((feat2, f_reduct))

        last_ind = inds[1]

    return feat2, red_pca


def main():
    clf_choice = None
    ftr_choice = None
    ftr_sel = None
    win_len = None
    path = []
    opts = []
    cls_choice = []

    # If the config.txt file exists, extract the options from it.
    try:
        opts = [line.strip('\n') for line in open(CONF_FILE)]
        clf_choice = opts[0]
        use_cv = int(opts[1].split('$')[0])
        num_folds = int(opts[1].split('$')[1])

        ftr_choice = [int(opts.split('$')[0]) for opts in opts[2].split('|')]
        ftr_sel = {int(opts.split('$')[0]): int(opts.split('$')[1]) for opts in opts[2].split('|')}

        # ftr_choice, ftr_sel = [int(x), int(y) for x, y in setting.split('$') for settings in ftr_settings]
        win_len = float(opts[3])
        reduction_choice = int(opts[4])
        path = [f for f in opts[5].split('$')]
        cls_choice = [x for x in opts[6].split('|')]
        Num_tests = int(opts[7])
    except FileNotFoundError:
        use_cv = int(input("Enter 1 to use Stratified Kfold CV: \n"))

    clf = get_classifier_from_cmd(clf_choice)

    features, targets, select_ind = get_feature_choice_cmd(ftr=ftr_choice, ftr_sel=ftr_sel, path=path, cls=cls_choice,
                                                           win_len=win_len)

    numtest = {}
    train_t = []
    test_t = []
    test_ind = []
    train_ind = []

    print("*****Starting to Test*****")
    # In practice this option is never taken
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
        # perform total classification for the specified number of iterations
        for _ in range(Num_tests):
            skf = StratifiedKFold(targets, n_folds=num_folds)

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
                train_feats = features[train_i]

                # Choose Selection or PCA reduction
                if reduction_choice == 1:
                    train_feats, train_supp = feature_selection(features[train_i], select_ind, train_target)
                elif reduction_choice == 2:
                    reduct_pca = {}
                    for inds in select_ind:
                        size = int((inds[1] - inds[0]) / inds[2])
                        reduct_pca[inds[0]] = PCA(size)
                        # reduct_pca[inds[0]] = TruncatedSVD(n_components=size)
                        # reduct_pca[inds[0]] = KernelPCA(n_components=size, kernel='linear')
                        # reduct_pca[inds[0]] = RandomizedPCA(n_components=size)

                    train_feats, reduct_pca = feature_reduction_fit(features[train_i], select_ind, reduct_pca, fit=True)

                # Fit the model to the training data
                clf = clf.fit(train_feats, train_target)

                # Prepare the test data in the same format as the training data
                test_target = [targets[x] for x in test_i]
                test_feats = features[test_i, :]
                if reduction_choice == 1:
                    test_feats = test_feats[:, train_supp]
                elif reduction_choice == 2:
                    test_feats, reduct_pca = feature_reduction_fit(features[test_i], select_ind, reduct_pca)

                # Test the model of test set. Calculate the Accuracy Score and Confusion Matrix
                # Accuracy score will have the mean of all results calculated. CM will be summed.
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

                # gc.collect()
                # print("Took out the trash!!")

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

            avg_score = total_score / num_folds
            savefile.write("Total Accuracy: %0.10f \n\n" % avg_score)

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
# D:\Documents\DT228_4\FYP\Datasets\080327\0_Audio
