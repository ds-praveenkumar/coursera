# -*- coding: utf-8 -*-
"""
Created on Wed May 27 00:51:19 2020

@author: PRAVEEN KUMAR -1
"""
import numpy as np
from utility import activation_function

class BasicNeuralNetwork:
    """
        This class implements a basic neural Network
    """
    def __init__(self,dimensions):
        self.dimensions = dimensions
        
        
    def initilize_parameters(self):
        """
            Initilizes Weight matrics and Bias to 0
        """
        self.w = np.zeros((self.dimensions,1))
        self.b = 0
        print('Neural Network Initilised: Weight_dimensions: {0}, b: {1}'.format(str(self.w.shape),str(self.b)))
        
    def propagation(self,X,Y,w,b):
        """
        
        """
        m = X.shape[1]
        activation = activation_function.sigmoid(np.matmul(w.T,X)+b)
        cost = (1/m) * np.sum(Y * np.log(activation) + (1-Y) * np.log(1-activation))
        #print('cost: ',cost)
        #print('activation: ',activation)
        
        dw = (1/m) * np.matmul(X,(activation-Y).T)
        db = (1/m) * np.sum(np.subtract(activation , Y))
        
        gradients = {
                     'dw': dw,
                     'db': db
                     }
        #print('dw: ', gradients['dw'])
        #print('db: ',gradients['db'])
        
        return gradients,cost
    def train(self,X,Y,w,b,learning_rate,number_iteration,print_iteration=False):
        """
        
        """
        
        costs = []
        
        for i in range(number_iteration):
            grads, cost = self.propagation(X=X,Y=Y,w=w,b=b)
            
            dw = grads['dw']
            db = grads['db']
            
            w = w - np.multiply(learning_rate ,dw)
            b = b - np.multiply(learning_rate,db)
            
            if i % 100 == 0:
                costs.append(cost)
            
            if print_iteration and i % 100 == 0:
                print("cost after iteration %i: %f" %(i,cost))
        params = {'w': w,
                  'b' : b
                  }
        
        grads = {'dw': dw,
                 'db': db                
                }
        return params,grads
    
    def predict():
        """
        
        """
        pass
    
    
if __name__=='__main__':
    nn = BasicNeuralNetwork(2)
    nn.initilize_parameters()
    X = np.array([1.,3.3]) * 0.01
    X = X.reshape(X.shape[0],-1)
    Y = np.array([1,0])
    Y = Y.reshape(Y.shape[0],-1)
    w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([[1,0,1]])
    nn.propagation(X=X,Y=Y,w=nn.w,b=nn.b)
    nn.train(X=X,Y=Y,w=w,b=b,learning_rate=0.03,number_iteration=2900,print_iteration=True)
        
    