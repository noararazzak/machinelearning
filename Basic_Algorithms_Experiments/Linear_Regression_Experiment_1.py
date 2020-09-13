from statistics import mean, stdev
import numpy as np
import random
import matplotlib.pyplot as plot
import matplotlib.style as style
style.use('dark_background')


def create_dataset(number, variance, step, correlation=False, correlation_type=None):
    val = 1
    ys = []
    for i in range(number):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation_type == 'pos':
            val += step
        elif correlation and correlation_type == 'neg':
            val -= step

    xs = [i for i in range(number)]
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def best_fit_slope (xs, ys):
    lower = mean (xs*xs)-(mean(xs)*mean (xs))
    upper = mean (xs*ys) - (mean (xs)* mean(ys))
    beta_hat = (upper/lower)
    intercept = mean(ys) - beta_hat*mean(xs)
    return beta_hat, intercept


def goodness_of_fit(y_original, y_other):
    sum_of_squares = sum ((y_original - y_other)*(y_original - y_other))
    return sum_of_squares


def r_squared_calculation(y_original, y_estimated):
    y_mean = [mean(y_original) for y in y_original]
    sum_of_residuals = goodness_of_fit(y_original, y_estimated)
    sum_of_differences = goodness_of_fit(y_original, y_mean)
    r_squared = 1 - (sum_of_residuals/sum_of_differences)
    return r_squared


def main():
    xs, ys = create_dataset(100, 25, 1, correlation=True, correlation_type='pos')
    beta_hat, intercept = best_fit_slope(xs, ys)
    y_estimated = [((beta_hat*x) + intercept) for x in xs ]
    beta_hat_std_dev = stdev(y_estimated)
    r_squared = r_squared_calculation(ys, y_estimated)
    plot.scatter(xs, ys, color='b')
    plot.plot(xs, y_estimated)
    plot.show()
    print(beta_hat, intercept, beta_hat_std_dev,r_squared)
    return beta_hat, intercept, beta_hat_std_dev, r_squared





