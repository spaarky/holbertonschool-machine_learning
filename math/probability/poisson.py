#!/usr/bin/env python3
"""
Representing poisson distribution
"""


class Poisson:
    """Class Poisson Distribution"""
    e = 2.7182818285
    def __init__(self, data=None, lambtha=1.):
        """Class contructor"""
        if data is None:
            if lambtha <= 0:
                raise ValueError('lambtha must be a positive value')
            else:
                self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError('data must be a list')
            elif len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.lambtha = sum(data) / len(data)

    def pmf(self, k):
        """PMF"""
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        e_mean = Poisson.e ** - self.lambtha
        mean_k = self.lambtha ** k
        k_factorial = 1
        for f in range(1, k + 1):
            k_factorial *= f
        pmf = e_mean * mean_k / k_factorial
        return pmf

    def cdf(self, k):
        """CDF"""
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        cdf = 0
        for i in range(k + 1):
            cdf += self.pmf(i)
        return cdf
