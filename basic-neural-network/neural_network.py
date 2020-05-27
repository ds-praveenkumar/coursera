# -*- coding: utf-8 -*-
"""
Created on Tue May 26 13:13:15 2020

@author: PRAVEEN KUMAR -1
"""

import numpy as np
from utility import sigmoid


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
        m = X.shape[1]
        activation =  sigmoid.sigmoid(Z=(np.matmul(W.T,X)+b))
        cost = -(1/m) * np.sum( Y*np.log(activation) + 
                          (1-Y)*np.log(1 - activation)) 
        
        cost = np.squeeze(np.asarray(cost))
        
        return activation,cost
    
    def backward_propagation(self,X,activation,Y):
        """
            Calculates Jw and Jb for backwaord propagation
            ARGS:
                - activation: activation function
                - Y: Output label
        """
        m = X.shape[1]
        dw = (1/m) * np.matmul(X,(activation-Y).T)
        db = (1/m) * np.sum(activation - Y)
        
        gradients = {
                "dw": dw,
                "db": db
                }
        return gradients
    
    def optimize(self,W,b,dw,db,learning_rate):
        """
            updates weight matrix
            ARGS:
                - dw: Weight Matrix
                - db: Bias vector
                - learning_rate: step_size for dradient descent
        """
        W = W - np.dot(learning_rate ,dw,out=None)
        b = b - np.dot(learning_rate , db,out=None)
        return W,b
    
    def train(self,X,W,b,number_iterations,Y,learning_rate,print_cost=False):
        """
            train the neural Network
        """
        costs = []
        
        for i in range(number_iterations):
            activation, cost = self.forward_propagation(W,b,X,Y)
            grads = self.backward_propagation(X=X,activation=activation,Y=Y)
            
            
            dw = grads['dw']
            db = grads['db']
            
            W = W - np.dot(learning_rate , dw,out=None)
            b = b - np.dot(learning_rate , db,out=None)
            
            if i % 100 == 0:
                costs.append(cost)
            
            if print_cost and  i % 100 == 0:
                print("Cost after iteration %i: %f" %(i, cost))
                
        params = {"w": W,
              "b": b}
    
        grads = {"dw": dw,
             "db": db}
            
        return params,grads,costs
            
    def predict(W,b,test):
        """
            predicts data based on weights
        """
        m = test.shape[1]
        Y_predictions = np.zeros((1,m))
        W = W.reshape((test.shape[0],1))
        
        activation = sigmoid.sigmoid(Z=np.matmul(W.T,test)+b)
        
        for i in range(activation.shape[1]):
            if activation[0,i] <  0.5:
                Y_predictions[0,i] = 0
            else:
                Y_predictions[0,i] = 1
        return Y_predictions
        
        
if __name__=='__main__':
    nn = neural_network(dim=2)
    X = np.random.randn((2)) * 0.01
    Y = np.array([1,0]).reshape(2,1)
    X = X.reshape((2,1))
    print('X:',X,'\nX shape: ',X.shape,end='\n')
    print('Y:',Y,'\nY Shape: ',Y.shape,end='\n')
    print('Weight:',nn.W, '\nW Shape: ',nn.W.T.shape,'\nBias: ',nn.b,end='\n')
    activation,cost_function = nn.forward_propagation(nn.W,nn.b,X,Y)
    grads = nn.backward_propagation(X,activation=activation,Y=Y)
    print('activation: ',activation,'\ncost_function: ', cost_function)
    print('dw: ',grads['dw'],'\ndb: ',grads['db'])
    W,b = nn.optimize(nn.W,nn.b,grads['dw'],grads['db'],learning_rate=0.1)
    print('updated weight: ', W,'\nupdated bias: ',b,end='\n')
    params,grads,costs = nn.train(X=X,W=nn.W,b=nn.b,number_iterations=2400,Y=Y,learning_rate=0.01,print_cost=True)
    w = np.array([[0.1124579],[0.23106775]])
    b = -0.3
    test = np.array([[1.,-1.1,-3.2],[1.2,2.,0.1]])
    predicts = nn.predict(test=test,b=b,W=w)
    #print('predictions: ', predicts)
   