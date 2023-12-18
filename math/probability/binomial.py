#!/usr/bin/env python3
"""
Representing Binomial distribution
"""


class Binomial:
    """Class Binomial Distribution"""
    def __init__(self, data=None, n=1, p=0.5):
        """Class constructor"""
        if data is None:
            if n < 1:
                raise ValueError('n must be a positive value')
            if p < 0 or p > 1:
                raise ValueError('p must be greater that 0 and less than 1')
            self.p = float(p)
            self.n = int(n)
        else:
            if type(data) is not list:
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain mulitple values')
            mean = sum(data) / len(data)
            var = sum(map(lambda i: (i - mean) ** 2, data)) / len(data)
            self.p = 1 - (var / mean)
            self.n = int(round(mean / self.p))
            self.p = mean / self.n

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of 'successes'

        Args:
            k (int/float): number of successes

        Returns:
            float: value of the PMF for k
        """
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        n_factorial = 1
        for i in range(1, self.n + 1):
            n_factorial *= i
        x_factorial = 1
        for i in range(1, k + 1):
            x_factorial *= i
        op_factorial = 1
        for i in range(1, (self.n - k) + 1):
            op_factorial *= i
        comb = n_factorial / (x_factorial * op_factorial)
        pmf = comb * (self.p ** k) * ((1 - self.p) ** (self.n - k))
        return pmf

    def cdf(self, k):
        """Calculates the value of the CDF for a given number of 'successes'

        Args:
            k (int/float): number of successes

        Returns:
            float: value of the CDF for k
        """
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        cdf = 0
        for i in range(k + 1):
            cdf += Binomial.pmf(self, i)
        return cdf
