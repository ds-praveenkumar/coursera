# -*- coding: utf-8 -*-
"""
Created on Wed May 27 22:00:41 2020

@author: PRAVEEN KUMAR -1
"""
import numpy as np


def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def tanh(Z):
    return (np.exp(Z) - np.exp(-Z))/(np.exp(Z) + np.exp(-Z))

def relu(Z):
    return np.max(0,Z)

def leaky_relu(Z):
    return np.max(0.01*Z,Z)

