import numpy as np

from skfuzzy import gaussmf, gbellmf, sigmf

def gaussian_mf(x, mean, sigma):
    return gaussmf(x, mean, sigma)

def gbellmf(x, params):
    return gbellmf(x, params)

def sigmf(x, params):
    return sigmf(x,params)