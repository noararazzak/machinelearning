import numpy as np
import matplotlib.pyplot as plot
import matplotlib.style as style
from numpy.core.multiarray import ndarray
from typing import Dict

style.use('ggplot')


data_dictionary: Dict[int, ndarray] = {-1: np.array([[1, 18], [2, 12], [3, 15], [4, 19]]),
                                       1: np.array([[5, 1], [6, -1], [7, 3], [8, 1]])}

data_dictionary_2: Dict[int, ndarray] = {-1: np.array([[1, 7], [2, 8], [3, 8]]),
                                         1: np.array([[5, 1], [6, -1], [7, 3]])}

data_dictionary_3: Dict[int, ndarray] = {-1: np.array([[1, 13], [2, 14], [3, 15], [4, 19], [2, 10]]),
                                         1: np.array([[10, 1], [12, -1], [12, 2], [13, 1], [13, 2]])}


def matrix_optimization(data):
    all_features = []
    for yi in data:
        for xi in data[yi]:
            for feature in xi:
                all_features.append(feature)

    max_value = max(all_features)
    min_value = min(all_features)
    b_range = .001
    b_step = 0.005
    step_sizes = [max_value*0.01, max_value * 0.001, max_value * 0.0001]
    min_w = 1000
    current_w = [max_value, max_value]
    opt_dictionary = {}
    for step in step_sizes:
        optimized = False
        while not optimized:
            for bias in np.arange(-1*(b_range*max_value), b_range*max_value, b_step):
                for a in np.arange(-current_w[0], current_w[0], step):
                    for b in np.arange(-current_w[1], current_w[1], step):
                        w = [a, b]
                        option_found = True
                        for yi in data:
                            for xi in data[yi]:
                                if not yi*(np.dot(w, xi) + bias) >= 1:
                                    option_found = False
                                    break

                        if option_found:
                            print('Option found')
                            print('value of a: ' + repr(a) + ', ' + 'value of b: ' + repr(b))
                            current_w = w
                            modulo_current_w = np.linalg.norm(current_w)
                            print('Current Modulo: ' + repr(modulo_current_w))
                            opt_dictionary[np.linalg.norm(current_w)] = [current_w, bias]
                            if modulo_current_w < min_w:
                                min_w = modulo_current_w
                                # min_w_vector = previous_w
                                # current_w = min_w_vector
                                optimized = True
                                print('Optimized a step.')
                            # else:
                                # previous_w = current_w

        norms = sorted([key for key in opt_dictionary])
        opt_choice = opt_dictionary[norms[0]]
        w = opt_choice[0]
        bias = opt_choice[1]
        print(w, bias)
        return w, bias, max_value, min_value


def plot_essentials():
    colors_dictionary = {-1: 'r', 1: 'b'}
    figure = plot.figure()
    axes = figure.add_subplot(1, 1, 1)
    return colors_dictionary, axes


def hyperplane(x, w, b, v):
    return -(w[0]*x - b + v)/w[1]


def visualize(data):
    colors_dictionary, axes = plot_essentials()
    [[axes.scatter(x[0], x[1], s=100, color=colors_dictionary[i]) for x in data[i]] for i in
     data_dictionary]
    w, bias, max_value, min_value = matrix_optimization(data)
    data_range = [min_value*0.9, max_value*1.1]
    min_x = data_range[0]
    max_x = data_range[1]

    positive_sv_1 = hyperplane(min_x, w, bias, 1)
    positive_sv_2 = hyperplane(max_x, w, bias, 1)
    axes.plot([min_x, max_x], [positive_sv_1, positive_sv_2], 'k')

    negative_sv_1 = hyperplane(min_x, w, bias, -1)
    negative_sv_2 = hyperplane(max_x, w, bias, -1)
    axes.plot([min_x, max_x], [negative_sv_1, negative_sv_2], 'k')

    decision_sv_1 = hyperplane(min_x, w, bias, 0)
    decision_sv_2 = hyperplane(max_x, w, bias, 0)
    axes.plot([min_x, max_x], [decision_sv_1, decision_sv_2], 'y--')

    plot.show()


def understanding_loops():
    value = 10
    step = 2
    w = []
    previous_w = [12, 12]
    for a in np.arange(-value, value+1, step):
        for b in np.arange(-value, value+1, step):
            w = [a, b]
            print('value of a: ' + repr(a) + ', ' + 'value of b: ' + repr(b))
            current_w = [a, b]
            modulo_current_w = np.linalg.norm(current_w)
            modulo_previous_w = np.linalg.norm(previous_w)
            if modulo_previous_w < modulo_current_w:
                print(current_w)
                return current_w
            else:
                previous_w = current_w


def main():
    visualize(data_dictionary)