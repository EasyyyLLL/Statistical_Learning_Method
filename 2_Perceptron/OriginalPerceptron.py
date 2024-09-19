import numpy as np


def Perceptron(X, Y, eta):
    dim = X.ndim
    wi = np.array([0 for i in range(dim)])
    bi = 0
    while True:
        for x, y in zip(X,Y):
            points = np.dot(x, wi)
            if y * (points + bi) > 0:
                continue
            else:
                wi = wi + eta * y * x
                bi += eta * y
                break
        else:
            wi = tuple(wi)
            return wi,bi


if __name__ == '__main__':
    X = np.array([[3, 3], [4, 3], [1, 1]])
    Y = [1, 1, -1]
    eta = 1
    (wi, bi) = Perceptron(X, Y, eta)
    fmt = "感知机模型f = sign({}·x + {:.2f})"
    print(fmt.format(wi, bi))
