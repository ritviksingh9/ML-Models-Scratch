from numpy import *
import random
import matplotlib.pyplot as plt

def error(m, b, points):
    error = 0
    for i in range(0, len(points)):
        error += 0.5 * (((m * points[i, 0] + b) - points[i, 1]) ** 2) / len(points)
    return error

def gradient_descent(m, b, learning_rate, points):
    delta_m = 0
    delta_b = 0

    for i in range(0, len(points)):
        delta_m += (m * points[i, 0] + b - points[i, 1]) * points[i, 0] / len(points)
        delta_b += (m * points[i, 0] + b - points[i, 1]) / len(points)

    return m - learning_rate * delta_m, b - learning_rate * delta_b


def run():
    points = genfromtxt("data.csv", delimiter=",")
    
    x = [points[i][0] for i in range(0, len(points))]
    y = [points[i][1] for i in range(0, len(points))]
    
    plt.scatter(x, y)

    learning_rate = 0.0001
    m = random.randint(0, 1)
    b = random.randint(0, 1)
    iterations = 2000

    plt.plot(x,m*array(x)+b, 'r') #initial guess

    print("Starting gradient descent at m = {0}, b = {1}, error = {2}".format(m, b, error(m, b, points)))
    print("Running...")
    for i in range(0, iterations):
        m, b = gradient_descent(m, b, learning_rate, points)
    print("After {0} iterations m = {1}, b = {2}, error = {3}".format(iterations, m, b, error(m, b, points)))

    plt.plot(x,m*array(x)+b, 'g') #final
    plt.show()


if __name__ == '__main__':
    run()
