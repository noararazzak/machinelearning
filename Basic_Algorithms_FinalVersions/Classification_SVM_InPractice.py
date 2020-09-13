from Basic_Algorithms_FinalVersions.Linear_Classifiers import LinearSVMClassifier
import numpy as np
import pandas as pd
import logging
import os


def create_dataset():

    x, y = use_iris_data()
    create_log("logs", "SVM_Logs.log")

    # for i in x:
    #     logging.info(i)
    # for i in y:
    #     logging.info(i)

    logging.info(x.shape)
    logging.info(y.shape)
    return x, y


def create_log(folder, name):
    log_address = os.path.join(os.getcwd(), folder)
    logfile_name = os.path.join(log_address, name)
    print(logfile_name)
    logging.basicConfig(filename=logfile_name, format='%(message)s', level=logging.DEBUG)


def use_linear_svm_to_train():
    x, y = create_dataset()

    svm = LinearSVMClassifier(x, y)

    loss_information, random_ids = svm.train_data(x, y, 100, 100, False)
    min_loss, W = svm.minimum_loss(loss_information)
    x_test, y_test = svm.find_test_data(x, y, random_ids)
    y_predict = svm.predict(x_test, W)
    accuracy = svm.accuracy(y_test, y_predict)

    print (min_loss, W, W.shape, accuracy)

    return min_loss, W, accuracy


def use_iris_data():
    names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'label']
    data_frame = pd.read_csv(r'D:\MachineLearning\Types of Analysis\Classification_KNN\iris.data.txt', header=None,
                             names=names)

    # print(data_frame)
    data_frame_x = data_frame.drop(['label'], axis=1)
    data_frame_y = data_frame['label']

    x = np.array(data_frame_x, dtype=np.int64)
    y_org = np.array(data_frame_y)

    y_list = []
    for i in y_org:
        if i == 'Iris-setosa':
            y_list.append(0)
        elif i == 'Iris-versicolor':
            y_list.append(1)
        else:
            y_list.append(2)

    y = np.array(y_list)

    return x, y


def use_breast_cancer_data():
    names = ['id', 'clump_thickness', 'cell_size', 'cell_shape', 'marginal_adhesion', 'epithelial_cell_size',
             'bare_nuclei', 'bland_chromatin', 'nucleoli', 'mitoses', 'label']
    data_frame = pd.read_csv( r'D:\MachineLearning\Types of Analysis\Classification_KNN\breast-cancer-wisconsin.data.txt',
        header=None, names=names)

    data_frame.replace('?', -99999, inplace=True)
    data_frame.drop(['id'], 1, inplace=True)
    # print(data_frame.head(10))
    # Creating x and y sets
    x = np.array(data_frame.drop(['label'], 1), dtype=np.int64)
    y_org = np.array(data_frame['label'], dtype=np.int64)

    y_list = []
    for i in y_org:
        if i == 2:
            y_list.append(0)
        else:
            y_list.append(1)

    y = np.array(y_list)

    return x, y
