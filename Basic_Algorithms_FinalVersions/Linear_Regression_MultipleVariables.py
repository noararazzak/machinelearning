import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from pylab import rcParams


def create_split_dataset():
    # Loading the data set into data_frame and dropping unnecessary variables.
    # my_pc='D:\MachineLearning\Types of Analysis\Regression\AirQualityUCI\AirQualityUCINew.csv'
    # alvi_pc='G:\Projects\Noara\Machine Learning\Types of Analysis\Regression\AirQualityUCI\AirQualityUCINew.csv'
    data_frame = pd.read_csv(r'D:\MachineLearning\Types of Analysis\Regression\AirQualityUCI\AirQualityUCINew.csv')
    data_frame = data_frame.drop(['Date', 'Time'], axis=1)

    data = np.array(data_frame)
    data_scaled = preprocessing.scale(data)
    data_frame = pd.DataFrame(data_scaled, columns=['CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)',
                                                    'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)',
                                                    'T', 'RH', 'AH'])

    correlation = data_frame.corr()
    corr_values = correlation['CO(GT)'].abs().sort_values(ascending=False)
    print(corr_values)

    # Splitting the data_frame into training and testing tests
    train = data_frame.sample(frac=0.8, random_state=99)
    test = data_frame.loc[~data_frame.index.isin(train.index), :]

    x = np.array(train.drop(['CO(GT)'], 1))
    y_train = np.array(train['CO(GT)'])
    x_train = np.c_[np.ones(x.shape[0]), train['NOx(GT)'], train['NO2(GT)'], train['NMHC(GT)']]
    x_train_sklearn = np.c_[train['NOx(GT)'], train['NO2(GT)'], train['NMHC(GT)']]

    x_t = np.array(test.drop(['CO(GT)'], 1))
    y_test = np.array(test['CO(GT)'])
    x_test = np.c_[np.ones(x_t.shape[0]), test['NOx(GT)'], test['NO2(GT)'], test['NMHC(GT)']]
    x_test_sklearn = np.c_[test['NOx(GT)'], test['NO2(GT)'], test['NMHC(GT)']]

    print(x_train.shape)
    print(x_test.shape)

    return x_train, x_train_sklearn, y_train, x_test, x_test_sklearn,  y_test


def gradient_descent(x, y, alpha):
    n = y.size
    theta = np.random.rand(x.shape[1])
    cost_list = []
    theta_list = []
    prediction_list = []
    run = True
    cost_list.append(1000000000)
    i = 0
    while run:
        prediction = np.dot(x, theta)
        prediction_list.append(prediction)
        error = prediction - y
        cost = 1/(2*n) * np.dot(error.T, error)
        cost_list.append(cost)
        theta = theta - (alpha * (1/n) * np.dot(x.T, error))
        theta_list.append(theta)
        if cost_list[i] - cost_list[i+1] < 0.00000001:
            run = False

        i = i+1
    cost_list.pop(0)
    theta = theta_list[-1]
    return prediction_list, cost_list, theta


def goodness_of_fit(y_original, y_other):
    return sum((y_original - y_other)**2)


def r_squared_calculation(y_original, y_predicted):
    upper = goodness_of_fit(y_original, y_predicted)
    lower = goodness_of_fit(y_original, y_original.mean())
    r_squared = 1 - (upper/lower)
    return r_squared


def use_sklearn_linear_regression(x_train, y_train):
    linear_model = LinearRegression()
    linear_model = linear_model.fit(x_train, y_train)
    theta_sklearn_list = [linear_model.intercept_, linear_model.coef_[0], linear_model.coef_[1],
                          linear_model.coef_[2]]
    r_squared_sklearn = linear_model.score(x_train, y_train)
    return theta_sklearn_list, r_squared_sklearn


def plotting(x_train, y_train):
    rcParams['figure.figsize'] = 10, 6
    plot.scatter(y_train, x_train[:, 1], s=5, label='NOx(GT)')
    plot.scatter(y_train, x_train[:, 2], s=5, label='NO2(GT)')
    plot.scatter(y_train, x_train[:, 3], s=5, label='NMHC(GT)')
    plot.xlabel('Hourly Averaged NOx, NO2 and NMHC Concentration', size=10)
    plot.ylabel('Hourly Averaged CO Concentration', size=10)
    plot.legend()
    plot.show()


def main():
    x_train, x_train_sklearn, y_train, x_test, x_test_sklearn, y_test = create_split_dataset()
    prediction_list, cost_list, theta = gradient_descent(x_train, y_train, 0.001)
    theta_sklearn, r_squared_sklearn = use_sklearn_linear_regression(x_train_sklearn, y_train)
    r_squared = r_squared_calculation(y_train, prediction_list[-1])
    print('Gradient Descent R Calculation: ' + str(r_squared) + ',' + 'Gradient Descent Theta: ' + str(theta))
    print('Scikit Learn R Calculation: ' + str(r_squared_sklearn) + ',' + 'Scikit Learn Theta: ' + str(theta_sklearn))
    plotting(x_train, y_train)
    plot.title('Cost Function J', size=15)
    plot.xlabel('No. of Iterations', size=10)
    plot.ylabel('Cost', size=10)
    plot.plot(cost_list)
    plot.show()



