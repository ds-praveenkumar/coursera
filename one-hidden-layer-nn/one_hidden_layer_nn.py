# -*- coding: utf-8 -*-
"""
Created on Wed May 27 23:20:43 2020

@author: PRAVEEN KUMAR -1
"""

import numpy as np

class OneHiddenLayerNN:
    
    def __init__(self,n_x,n_h,n_y):
        """
            Initilizes size of input, hidden layer unit, and output layer
            ARGS:
                - n_x: input size
                - n_h: hidden layer units
                - n_y: output size
        """
        self.n_x = n_x
        self.n_h = n_h
        self.n_y = n_y
        print("The size of the input layer is: n_x = " + str(n_x))
        print("The size of the hidden layer is: n_h = " + str(n_h))
        print("The size of the output layer is: n_y = " + str(n_y))
        print("*"*50)
    
    def initilize_parameters(self):
        """
            Initilizes weights and Bias
        """
        W1 = np.random.randn(self.n_h,self.n_x) * 0.01
        b1 = np.zeros((self.n_h,1))
        W2 = np.random.randn(self.n_y,self.n_h) * 0.01
        b2 = np.zeros((self.n_y,1))
        parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
        print("The size of the Weights 1st layer: W1 = " + str(W1.shape))
        print("The size of the bais 1st layer: b1 = " + str(b1.shape))
        print("The size of the weights 2nd layer: W2 = " + str(W2.shape))
        print("The size of the bais 2nd layer: b2 = " + str(b2.shape))
        print("*"*50)
        return parameters

if __name__=='__main__':
    nn = OneHiddenLayerNN(5,4,4)
    nn.initilize_parameters()