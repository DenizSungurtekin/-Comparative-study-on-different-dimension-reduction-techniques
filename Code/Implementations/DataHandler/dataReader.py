from numpy import genfromtxt

def csv(filename):
    return genfromtxt(filename, delimiter=',',dtype=None,encoding=None)