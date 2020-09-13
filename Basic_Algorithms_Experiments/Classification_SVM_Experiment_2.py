from typing import Dict
import numpy as np
import matplotlib.pyplot as plot
import matplotlib.style as style
from numpy.core.multiarray import ndarray

style.use('ggplot')

data_dictionary: Dict[int, ndarray] = {-1: np.array([[1, 18], [2, 12], [3, 15], [4, 19]]),
                                       1: np.array([[5, 1], [6, -1], [7, 3], [8, 1]])}

data_dictionary_2: Dict[int, ndarray] = {-1: np.array([[1, 7], [2, 7], [3, 7]]),
                                         1: np.array([[5, 1], [6, -1], [7, 3]])}

data_dictionary_3: Dict[int, ndarray] = {-1: np.array([[1, 13], [2, 14], [3, 15], [4, 19], [2, 10]]),
                                         1: np.array([[10, 1], [12, -1], [12, 2], [13, 1], [13, 2]])}


def plot_essentials():
    colors_dictionary = {-1: 'r', 1: 'b'}
    figure = plot.figure()
    axes = figure.add_subplot(1, 1, 1)
    return colors_dictionary, axes


def matrix_optimization(data):
    opt_dictionary = {}
    transforms = [[1, 1], [-1, 1], [-1, -1], [1, -1]]

    all_features = []

    for yi in data:
        for x in data[yi]:
            for feature in x:
                all_features.append(feature)

    max_feature_value = max(all_features)
    min_feature_value = min(all_features)

    step_sizes = [max_feature_value*0.1, max_feature_value*0.01, max_feature_value*0.001]

    b_range = 5
    b_step = 5
    latest_optimum = max_feature_value*10
    w = np.array([latest_optimum, latest_optimum])
    print(w)

    for step in step_sizes:
        w = np.array([latest_optimum, latest_optimum])
        optimized = False
        while not optimized:
            for b in np.arange(-1*(max_feature_value*b_range), max_feature_value*b_range, b_step):
                for transform in transforms:
                    w_t = w * transform
                    option_found = True
                    for yi in data:
                        for xi in data[yi]:
                            if not yi * (np.dot(w_t, xi) + b) >= 1:
                                option_found = False

                    if option_found:
                        opt_dictionary[np.linalg.norm(w_t)] = [w_t, b]
            if w[1] < 0:
                optimized = True
                print('Optimized a step')
            else:
                w = w - step

        norms = sorted([n for n in opt_dictionary])
        opt_choice = opt_dictionary[norms[0]]
        w = opt_choice[0]
        b = opt_choice[1]
        latest_optimum = opt_choice[0][0] + step*2

    print(w, b, "Min ||w||: " + repr(np.linalg.norm(w)))
    return w, b, max_feature_value, min_feature_value


def hyperplane(x, w, b, v):
    return -(w[0]*x - b + v)/w[1]


def visualize():
    colors_dictionary, axes = plot_essentials()
    [[axes.scatter(x[0], x[1], s=100, color=colors_dictionary[i]) for x in data_dictionary_2[i]] for i in
     data_dictionary]
    w, b, max_value, min_value = matrix_optimization(data_dictionary_2)
    data_range = [min_value*0.9, max_value*1.1]
    min_x = data_range[0]
    max_x = data_range[1]

    positive_sv_1 = hyperplane(min_x, w, b, 1)
    positive_sv_2 = hyperplane(max_x, w, b, 1)
    axes.plot([min_x, max_x], [positive_sv_1, positive_sv_2], 'k')

    negative_sv_1 = hyperplane(min_x, w, b, -1)
    negative_sv_2 = hyperplane(max_x, w, b, -1)
    axes.plot([min_x, max_x], [negative_sv_1, negative_sv_2], 'k')

    decision_sv_1 = hyperplane(min_x, w, b, 0)
    decision_sv_2 = hyperplane(max_x, w, b, 0)
    axes.plot([min_x, max_x], [decision_sv_1, decision_sv_2], 'y--')

    plot.show()


def main():
    visualize()
