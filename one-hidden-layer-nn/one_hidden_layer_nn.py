# -*- coding: utf-8 -*-
"""
Created on Wed May 27 23:20:43 2020

@author: PRAVEEN KUMAR -1
"""

import numpy as np
from utility import activation_function

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
    
    def layer_size(self,X):
        """
            Returns shape of the layer
            ARGS:
                - X: Array/Matrix
            RETURNS:
                - size: tuple of the Shape of matrix
        """
        size = X.shape
        return size
    
    def initilize_parameters(self):
        """
            Initilizes weights and Bias
        """
        np.random.seed(123)
        W1 = np.random.randn(self.n_h,self.n_x) * 0.01
        b1 = np.zeros((self.n_h,1))
        W2 = np.random.randn(self.n_y,self.n_h) * 0.01
        b2 = np.zeros((self.n_y,1))
        parameters = {
                  "W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
        print("The size of the Weights 1st layer: W1 = " + str(W1.shape))
        print("The size of the bais 1st layer: b1 = " + str(b1.shape))
        print("The size of the weights 2nd layer: W2 = " + str(W2.shape))
        print("The size of the bais 2nd layer: b2 = " + str(b2.shape))
        print("*"*50)
        return parameters

    def forward_propagation(self,X):
        """
        
        """
        parameters = self.initilize_parameters()
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']
        activation1 = (activation_function.tanh(np.add(np.matmul(W1,X),b1))) 
        activation2 = (activation_function.sigmoid(np.matmul(W2,activation1)+b2)) 
        
        cache = {
                'A1' : activation1,
                'A2' : activation2
                }
        print('size of X: ', X.shape)
        print('size of activation1: ',activation1.shape)
        print('size of actiavtion2: ',activation2.shape)
        print("*"*50)
        return cache
    
    def compute_cost(self,Y,A2,parameters):
        """
        
        """
        m = Y.shape[1]
        print('m:',m)
        logprobs = logprobs = 1/m*(np.sum(np.multiply(np.log(A2),Y)+np.multiply(np.log(1-A2),(1-Y))))
        cost = - np.sum(logprobs)
        
        cost = float(np.squeeze(cost))
        print('cost: ',cost)
        print("*"*50)
        return cost
    
    def backward_propagation(self,X,Y,cache,parameters):
        """
        
        """
        
        W1 = parameters['W1']
        W2 = parameters['W2']
        m = X.shape[1]
        A2 = cache['A2']
        A1 = cache['A1']
        dZ2 =  A2 - Y
        dW2 = (1/m)* np.matmul(dZ2,A2.T)
        db2 = (1/m) * np.sum(dZ2,axis=1, keepdims=True)
        dZ1 = np.matmul(W2.T,dZ2) * (1 - np.power(A1, 2))
        dW1 = np.matmul(dZ1,X.T)
        db1 = (1/m) * np.sum(dZ1,axis=1,keepdims=True)
        grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
        
        #print('grads: ', grads)
        return grads
    
    def update_parameters(self,parameters,grads,learning_rate=1.2):
        """
        
        """
        W1 = parameters['W1']
        W2 = parameters['W2']
        b1 = parameters['b1']
        b2 = parameters['b2']
        
        dW1 = grads['dW1']
        dW2 = grads['dW2']
        db1 = grads['db1']
        db2 = grads['db2']
        
        W1 = W1 - (learning_rate * dW1)
        W2 = W2 - (learning_rate * dW2)
        b1 = b1 - (learning_rate * db1)
        b2 = b2 - (learning_rate * db2)
        
        parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
        #print('updated parameters: ',parameters)
        return parameters
    
    def nn_model(self,X, Y, n_h, num_iterations = 2400, print_cost=True):
        """
        
        """
        n_x = self.layer_size(X)[0]
        n_y = self.layer_size(Y)[0]
        print("The size of the input layer is: n_x = " + str(n_x))
        print("The size of the input layer is: n_y = " + str(n_y))
        
        # initilize parameters
        parameters = self.initilize_parameters()
        
        for i in range(num_iterations):
            #forwaard propagation
            cache = nn.forward_propagation(X)
            A2 = cache['A2']
           
            #compute cost
            cost = nn.compute_cost(A2,Y,parameters)
            
            #Backword propagation
            grads = nn.backward_propagation(X,Y,cache,parameters)
            
            #optimize weights
            parameters = self.update_parameters(parameters,grads, 0.15)
            
            if print_cost and i % 1000 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))
        return parameters
        
        
        
if __name__=='__main__':
    nn = OneHiddenLayerNN(n_x=5,n_h=4,n_y=4)
    X = np.random.randn(5) * 0.01
    X = X.reshape((X.shape[0],1))
    Y = np.array([1,0,0,1])
    Y = Y.reshape((Y.shape[0],1))
    #cache = nn.forward_propagation(X)
    #parameters = nn.initilize_parameters() 
    #print('parameters: ',parameters)
    #cost = nn.compute_cost(Y,cache['A2'],parameters)
    #grads = nn.backward_propagation(X,Y,cache,parameters)
    #updated_params = nn.update_parameters(parameters,grads,0.15)
    nn.nn_model(X,Y,5)
    
    
    