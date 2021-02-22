import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import cauchy, laplace, poisson, uniform
from scipy.special import factorial
from enum import Enum


class Distribution(Enum):
    NORMAL = 1
    CAUCHY = 2
    LAPLACE = 3
    POISSON = 4
    UNIFORM = 5

    @staticmethod
    def in_str(distribution):
        if distribution == Distribution.NORMAL:
            return 'Normal'
        elif distribution == Distribution.CAUCHY:
            return 'Cauchy'
        elif distribution == Distribution.LAPLACE:
            return 'Laplace'
        elif distribution == Distribution.POISSON:
            return 'Poisson'
        elif distribution == Distribution.UNIFORM:
            return 'Uniform'
        else:
            return ''


def density(x, a, b, distribution):
    if distribution == Distribution.NORMAL:
        mu = a
        sigma = np.sqrt(b)
        f = lambda x, sigma, mu: 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))
        return f(x, sigma, mu)
    elif distribution == Distribution.CAUCHY:
        return cauchy.pdf(x, a, b)
    elif distribution == Distribution.LAPLACE:
        return laplace.pdf(x, a, b)
    elif distribution == Distribution.POISSON:
        f = lambda x, a: np.exp(-a) * np.power(a, x) / factorial(x)
        return f(x, a)
    elif distribution == Distribution.UNIFORM:
        return uniform.pdf(x, a, b)
    else:
        return None


def selection(mu, sigma, size, distribution):
    if distribution == Distribution.NORMAL:
        return np.random.normal(mu, sigma, size)
    elif distribution == Distribution.CAUCHY:
        return cauchy.rvs(mu, sigma, size)
    elif distribution == Distribution.LAPLACE:
        return laplace.rvs(mu, sigma, size)
    elif distribution == Distribution.POISSON:
        return poisson.rvs(mu, size=size)
    elif distribution == Distribution.UNIFORM:
        return uniform.rvs(mu, sigma, size)
    else:
        return None


def plot_building(size, number_bins, a, b, distribution, color):
    x = selection(a, b, size, distribution)
    count, bins, ignored = plt.hist(x, number_bins, density=True, fill=None)
    plt.plot(bins, density(bins, a, b, distribution), linewidth=1, color=color)

    str_distribution = Distribution.in_str(distribution)
    plt.title(str_distribution + ' numbers n=' + str(size))
    plt.xlabel(str_distribution + ' numbers')
    plt.ylabel('Density')
    plt.show()
    plt.savefig(str_distribution + str(size))


size = [10, 50, 1000]
numbers_bins = [16, 26, 56]
a_parameters = [0, 0, 0, 10, -np.sqrt(3)]
b_parameters = [1, 1, np.sqrt(2), 0, np.sqrt(3)]
distributions = [Distribution.NORMAL, Distribution.CAUCHY, Distribution.LAPLACE, Distribution.POISSON, Distribution.UNIFORM]
colors = ['blue', 'red', 'green', 'pink', 'gray']

for a, b, distribution, color in zip(a_parameters, b_parameters, distributions, colors):
    for n, number_bins in zip(size, numbers_bins):
        plot_building(n, number_bins, a, b, distribution, color)