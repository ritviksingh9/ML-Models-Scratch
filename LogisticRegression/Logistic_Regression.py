import numpy as np
import matplotlib.pyplot as plt
import math
import random

def sigmoid(z):
    return 1 / (1 + np.exp(-1*z))

def cost_reg(theta, X, y, reg_param, m):
    h = sigmoid(np.matmul(np.transpose(X), theta))
    #J = -1/m * sum(y * np.log(sigmoid(np.matmul(np.transpose(X), theta))) + (np.ones(m) - y) * np.log(np.ones(m) - sigmoid(np.matmul(np.transpose(X), theta)))) + reg_param / m * np.matmul(np.transpose(theta[1:]), theta[1:]) 
    J = -1 / m * (np.matmul(np.transpose(y), np.log(h)) + np.matmul(np.transpose(np.ones(m) - y), np.ones(m) - h))
    J = (-1 / m) * np.sum(y*np.log(sigmoid(np.transpose(X) @ theta)) + (np.ones(m) - y)*np.log(np.ones(m) - sigmoid(np.transpose(X) @ theta))) + (reg_param / (2 * m))*np.sum(theta[1:]*theta[1:])
    grad = 1/m * np.matmul(X, (sigmoid(np.transpose(X) @ theta) - y))
    grad[1:] += reg_param / m * theta[1:]
    return (J, grad)

def predict(theta, X):
    return sigmoid(np.matmul(np.transpose(X), theta))

if __name__ == '__main__':
    m = 80
    m_test = 20
    X = np.zeros((3, m))
    x_test = np.zeros((3, m_test))
    learning_rate = 0.01
    iterations = 100000
    reg_param = 0
    theta = np.zeros(3)
    theta[0] = 1
    y = np.zeros(m)

    for i in range(0, 3):
        theta[i] = random.randint(-1, 1)

    points = np.genfromtxt("log_reg_data.txt", delimiter=",")
    X[0][:] = np.ones(m)
    X[1][:] = [points[i][0] for i in range(0, m)]
    X[2][:] = [points[i][1] for i in range(0, m)]
    y = np.array([points[i][2] for i in range(0, m)])

    x_test[0][:] = np.ones(m_test)
    x_test[1][:] = [points[i][0] for i in range(m, m+m_test)]
    x_test[2][:] = [points[i][1] for i in range(m, m+m_test)]
    y_test = np.array([points[i][2] for i in range(m, m+m_test)])

    for i in range(0, iterations):
        error, grad = cost_reg(theta, X, y, reg_param, m)
        theta -= learning_rate / m * grad
        if(i % 20 == 0):
            print("Iterations: {0}  Error: {1}".format(i+1, error))

    for i in range(0, m):
        y_pred = predict(theta, (X[0][i], X[1][i], X[2][i]))
        #print(y_pred)
        #print(y_test[i])
        print("Prediction: {0}  Actual: {1}".format(y_pred, y[i]))

