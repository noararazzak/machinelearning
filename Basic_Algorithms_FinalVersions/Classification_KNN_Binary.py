import pandas as pd
import numpy as np
import math
import operator
from sklearn.neighbors import KNeighborsClassifier


def create_split_dataset():
    # Loading the dataset and adding a header row
    names = ['id', 'clump_thickness', 'cell_size', 'cell_shape', 'marginal_adhesion', 'epithelial_cell_size',
             'bare_nuclei', 'bland_chromatin', 'nucleoli', 'mitoses', 'label']
    data_frame = pd.read_csv(r'D:\MachineLearning\Types of Analysis\Classification_KNN\breast-cancer-wisconsin.data.txt',
                             header=None, names=names)

    data_frame.replace('?', -99999, inplace=True)
    data_frame.drop(['id'], 1, inplace=True)
    print(data_frame.head(10))

    # Randomizing the data_frame so that train and test sets are truly random
    train = data_frame.sample(frac=0.75, random_state=99)
    test = data_frame.loc[~data_frame.index.isin(train.index), :]

    # Creating train and test tests separately
    x_train = np.array(train.drop(['label'], 1), dtype=np.int64)
    y_train = np.array(train['label'], dtype=np.int64)
    x_test = np.array(test.drop(['label'], 1), dtype=np.int64)
    y_test = np.array(test['label'], dtype=np.int64)

    return x_train, y_train, x_test, y_test


def euclidean_distance(instance1, instance2):
    distance = 0
    for x in range(len(instance2)):
        distance = distance + pow((instance1[x]-instance2[x]), 2)

    return math.sqrt(distance)


def get_nearest_neighbors(x_train, y_train, x_test_instance, k):
    distances = []
    neighbors = []
    for x in range(len(x_train)):
        euclidean_distance_x = euclidean_distance(x_train[x], x_test_instance)
        distances.append((y_train[x], euclidean_distance_x))
    distances.sort(key=operator.itemgetter(1))
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def get_votes_on_neighbors(neighbors):
    label_votes = []
    for x in set(neighbors):
        label_votes.append((x, neighbors.count(x)))
    label_votes.sort(key=operator.itemgetter(1), reverse=True)
    return label_votes[0][0]


def get_votes_on_neighbors_v2(neighbors):
    case = {}
    for x in set(neighbors):
        case.update({x: neighbors.count(x)})
    sorted_votes = sorted(case.items(), key=operator.itemgetter(1), reverse=True)
    # sorted_votes_2 = sorted(case.items(), key=lambda kv: kv[1], reverse=True)
    return sorted_votes[0][0]


def get_accuracy(y_test, results):
    correct = 0
    for x in range(len(y_test)):
        if y_test[x] == results[x]:
            correct = correct + 1
    accuracy = (correct/float(len(y_test))) * 100
    return accuracy


def use_sklearn_knn_classifier(x_train, y_train, x_test, y_test):
    # Instantiating the model
    knn = KNeighborsClassifier()

    # Fitting the model
    knn.fit(x_train, y_train)

    # Predicting the response
    prediction_sklearn = knn.predict(x_test)

    # Getting the accuracy
    accuracy_sklearn = (knn.score(x_test, y_test))*100
    return prediction_sklearn, accuracy_sklearn


def main():
    x_train, y_train, x_test, y_test = create_split_dataset()
    prediction_sklearn, accuracy_sklearn = use_sklearn_knn_classifier(x_train, y_train, x_test, y_test)
    results = []
    for x in range(len(x_test)):
        neighbors = get_nearest_neighbors(x_train, y_train, x_test[x], k=5)
        result = get_votes_on_neighbors_v2(neighbors)
        results.append(result)
        print('Predicted_scratch: ' + repr(result) + ', ' + 'Predicted_sklearn: ' + repr(prediction_sklearn[x]) + ', '
              + 'Actual: ' + repr(y_test[x]))
    accuracy = get_accuracy(y_test, results)
    print('Accuracy_scratch: ' + repr(accuracy) + '%')
    print('Accuracy_sklearn: ' + repr(accuracy_sklearn) + '%')


def using_dictionary():
    our_list = ["a", "b", "b", "c", "b"]
    case = {}
    for x in set(our_list):
        case.update({x: our_list.count(x)})
    print(case)
    # sorted_votes = sorted(case.items(), key=operator.itemgetter(1), reverse=True)
    sorted_votes_2 = sorted(case.items(), key=lambda kv: kv[1], reverse=True)
    print(sorted_votes_2[0][0])
    return sorted_votes_2[0][0]


