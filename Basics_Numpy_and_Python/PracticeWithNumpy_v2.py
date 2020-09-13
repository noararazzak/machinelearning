import numpy as np
from random import shuffle

W = np.random.randint(1,3,12)
W = W.reshape(4, 3)


x = np.random.randint(1, 8, 20)
x= x.reshape(5,4)

y = np.random.randint(0,3,5)
y = y.reshape(5,)

reg = 0.01


def linear_svm_with_loops(W, x, y, reg):
    dw = np.zeros(W.shape)
    loss = 0.0
    num_classes = W.shape[1]
    num_train = x.shape[0]

    for i in range(num_train):
        scores = x[i].dot(W)
        # print(scores)
        correct_class_score = scores[y[i]]
        num_classes_greater_margin = 0

        for j in range(num_classes):
            print('j: ' + repr(j))
            if j == y[i]:
                continue
            margins = scores[j] - correct_class_score +1
            if margins > 0:
                num_classes_greater_margin += 1
                dw[:, j] = dw[:, j] + x[i, :].T
                print ('dw[:, j]: ' + repr(dw[:, j]) + 'x[i, :]: ' + repr(x[i, :]))
                loss = loss + margins

        dw[:, y[i]] = dw[:, y[i]] - x[i, :].T * num_classes_greater_margin

    loss = loss/num_train

    loss = loss + reg*np.sum(W*W)

    dw = dw/num_train + 2*reg*W

    return loss, dw


def main():
    loss, dw = linear_svm_with_loops(W, x, y, reg)
    y_unique = np.unique(y)
    print('loss: ' + repr(loss) + ', ' + 'dw: ' + repr(dw) + ', ' + 'y_unique: ' + repr(y_unique))




