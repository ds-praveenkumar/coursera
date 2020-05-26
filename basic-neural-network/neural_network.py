# -*- coding: utf-8 -*-
"""
Created on Tue May 26 13:13:15 2020

@author: PRAVEEN KUMAR -1
"""

import numpy as np
from utility.sigmoid import Sigmoid

class neural_network(object):
    """
        Base class for neural network 
    """
    W = np.zeros((2,2))
    b = 0 
    
    def __init__(self,dim):
        """
            Initilizes Weight matrics and Bias to 0
            ARGS:
                - dims: dimentions of he matrics
        """
        self.W =  np.zeros((dim,1))
        self.b = 0  
    
    def forward_propagation(self,W,b,X,Y):
        """
            Computes activation function and Cost function
            A = sigma(W.T*X+b)
            J = sigmoid
            ARGS:
                W = weight matrix
                b = Bias vector
                X = Input Matrix
                Y = Output Vector
        """
        m = X.shape[0]
        sigmoid = Sigmoid(0)
        activation =  sigmoid.sigmoid(Z=(np.matmul(W.T,X)+b))
        cost_function = -(1/m) * np.sum( np.matmul(Y,np.log(activation)) + 
                          np.matmul(1-Y,np.log(1 - activation))) 
        return activation,cost_function
    
    def backward_propagation(self,X,activation,Y):
        """
            Calculates Jw and Jb for backwaord propagation
            ARGS:
                - activation: activation function
                - Y: Output label
        """
        m = X.shape[0]
        dw = (1/m) * np.matmul(X,(activation-Y).T)
        db = (1/m) * np.sum(activation - Y)
        return dw,db
    
if __name__=='__main__':
    X = np.random.randn((4)) * 0.01
    Y = np.array([1,0,1,0]).reshape(4,-1)
    X = X.reshape((4,1))
    print('X:',X,end='\n')
    print('Y:',Y,end='\n')
    nn = neural_network(dim=4)
    activation,cost_function = nn.forward_propagation(nn.W,nn.b,X,Y)
    dw,db = nn.backward_propagation(X,activation=activation,Y=Y)
    print('activation: ',activation,'\ncost_function: ', cost_function)
    print('dw: ',dw,'\ndb: ',db)