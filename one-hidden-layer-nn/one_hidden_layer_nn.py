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
        """
        self.n_x = n_x
        self.n_h = n_h
        self.n_y = n_y
        print("The size of the input layer is: n_x = " + str(n_x))
        print("The size of the hidden layer is: n_h = " + str(n_h))
        print("The size of the output layer is: n_y = " + str(n_y))
        
        


if __name__=='__main__':
    nn = OneHiddenLayerNN(5,4,4)