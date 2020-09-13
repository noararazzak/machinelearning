import numpy as np
import operator
import random

class LinearSVMClassifier:

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.W = None

        # self.loss_information, self.random_ids_list = self.train_data(self.x, self.y, num_iterations, batch_size, verbose=False)
        # self.min_loss, self.W = self.minimum_loss(self.loss_information)
        # self.x_test, self.y_test = self.find_test_data(self.x, self.y, self.random_ids_list)
        # self.y_predict = self.predict(self.x_test, self.W)
        # self.accuracy_number = self.accuracy(self.y_test, self.y_predict)

    def linear_svm_vectorized(self, W, x, y, reg):
        num_train = x.shape[0]
        scores = x.dot(W)
        # print('score first: ' + repr(scores))
        correct_class_scores = scores[np.arange(num_train), y]
        # print('score second: ' + repr(correct_class_scores))
        correct_class_scores = correct_class_scores.reshape(num_train, -1)
        margins = np.maximum(0, scores - correct_class_scores + 1)
        # print('margins first:' + repr(margins))
        margins[np.arange(num_train), y] = 0
        # print('margins second:' + repr(margins))

        # Calculating loss
        loss = np.sum(margins)
        loss = loss + (0.5 * reg * np.sum(W * W))

        # Calculating dW
        margins[margins > 0] = 1
        print('margins third:' + repr(margins))
        margins[np.arange(num_train), y] = -np.sum(margins, axis=1)
        print('margins fourth:' + repr(margins))
        dw = x.T.dot(margins)
        dw = dw / num_train
        dw = dw + 2 * reg * W

        return loss, dw


    def train_data(self, x, y, num_iterations, batch_size=100, verbose=False):
        num_train, dim = x.shape
        # num_classes = np.max(y) + 1
        # We calculate the number of classes by calculating unique values in y.
        y_unique = np.unique(y)
        num_classes = y_unique.shape[0]
        learning_rate = 0.001
        reg = 0.01
        loss_information = []
        if self.W is None:
            self.W = 0.001 *np.random.randn (dim, num_classes)
        random_ids = np.random.choice(np.arange(num_train), size=batch_size, replace=False)
        random_ids_list = random_ids.tolist()
        for i in range(num_iterations):
            x_batch = None
            y_batch = None

            #  Randomize the training data into x_batch and y_batch based on batch_size
            #  random_ids = np.random.choice(np.arange(num_train), size = batch_size)

            x_batch = x[random_ids]
            y_batch = y[random_ids]

            # Calculate loss and gradient using linear_svm_vectorized
            loss, grad = self.linear_svm_vectorized(self.W, x_batch, y_batch, reg)
            # print('Loss: ' + repr(loss))
            # Calculate new W based on gradient
            self.W = self.W - learning_rate*grad

            loss_information.append((loss, self.W))

            if verbose and i % 100 == 0:
                print('iteration %d / %d: loss %f' % (i, num_iterations, loss))

        return loss_information, random_ids_list

    def minimum_loss(self, loss_information):
        loss_information.sort(key=operator.itemgetter(0))
        min_loss = loss_information[0][0]
        W = loss_information[0][1]
        return min_loss, W

    def find_test_data(self, x, y, random_ids_list):
        num_train = x.shape[0]
        entire_list = list(range(num_train))
        test_list = []
        print(len(entire_list))
        print(len(random_ids_list))
        for n in entire_list:
            if n in random_ids_list:
                continue
            else:
                test_list.append(n)

        # test_list = list(set(entire_list) - set(random_ids_list))
        print(len(test_list))
        test_list_x = []
        for i in test_list:
            test_list_x.append(x[i])
        x_test = np.array(test_list_x)

        test_list_y = []
        for j in test_list:
            test_list_y.append(y[j])
        y_test = np.array(test_list_y)

        return x_test, y_test


    def predict(self, x_test, W):
        y_predict = np.zeros(x_test.shape[0])

        y_predict = x_test.dot(W).argmax(axis=1)

        return y_predict

    def accuracy(self, y_test, y_predict):
        accuracy_score = 0.0
        num_test = y_test.shape[0]
        for i in y_predict:
            if y_predict[i] == y_test[i]:
                accuracy_score +=1
        print(accuracy_score)
        final_score = accuracy_score/num_test
        return final_score