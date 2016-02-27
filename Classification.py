from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neighbors import NearestNeighbors
import numpy as np
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
        classifier = tree.DecisionTreeClassifier()
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

    ftr = int(input())

    path = input("Enter path to features")

    features = np.array()
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

                np.vstack((features, cur_feature))
                classes.append(vid.get_category_from_name())

            except EOFError:
                print("EOF")
                break
            except TypeError:
                print("Unable to load object")
            except pkl.UnpicklingError:
                print("Unable to load object2")

    targets = np.array(classes)

    return features, targets


def main():
    clf = get_classifier_from_cmd()

    features, targets = get_feature_choice_cmd()

    clf = clf.fit(features[: int(len(features)/10)], targets[: int(len(features)/10)])
    clf.predict(features[- int(len(features)/10):], targets[- int(len(features)/10):])


if __name__ == "__main__":
    main()