import numpy as np
from random import shuffle

W = np.arange(1,4,0.25)
W = W.reshape(4, 3)


x = np.arange(1, 7, 0.25)
x= x.reshape(6,4)

y = np.array([1,2,0,2,1,0])
y = y.reshape(6,)

reg = 0.05


def linear_svm_with_loops(W, x, y, reg):
    dw = np.zeros(W.shape)
    loss = 0.0
    num_classes = W.shape[1]
    num_train = x.shape[0]

    for i in range(num_train):
        scores = x[i].dot(W)
        correct_class_score = scores[y[i]]
        num_classes_greater_margin = 0

        for j in range(num_classes):
            print('j: ' + repr(j))
            if j ==y[i]:
                continue
            margins = scores[j] - correct_class_score +1
            if margins > 0:
                num_classes_greater_margin += 1
                dw[:, j] = dw[:, j] + x[i, :].T
                print('dw[:, j]: ' + repr(dw[:, j]) + 'x[i, :]: ' + repr(x[i, :]))
                loss = loss + margins

        dw[:, y[i]] = dw[:, y[i]] - x[i, :].T *num_classes_greater_margin

    loss = loss/num_train

    loss = loss + 0.5*reg*np.sum(W*W)

    dw = dw/num_train + 2*reg*W

    return loss, dw


def linear_svm_vectorized(W, x, y, reg):
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
    loss = loss + (0.5*reg*np.sum(W*W))

    # Calculating dW
    margins[margins>0] =1
    print('margins third:' + repr(margins))
    margins[np.arange(num_train), y] = -np.sum(margins, axis =1)
    print('margins fourth:' + repr(margins))
    dw = x.T.dot(margins)
    dw = dw/num_train
    dw = dw + 2*reg*W

    return loss, dw


def main():
    loss, dw = linear_svm_vectorized(W, x, y, reg)
    y_unique = np.unique(y)
    num_classes = y_unique.shape[0]
    print(num_classes)
    print('loss: ' + repr(loss) + ', ' + 'dw: ' + repr(dw) + ', ' + 'y: ' + repr(y))





