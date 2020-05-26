# -*- coding: utf-8 -*-
import numpy as np

class Sigmoid:
    """
        This class implements sigmoid function 
    """
    def __init__(self,Z):
        self.Z = Z
        
    def sigmoid(self,Z):
        """
            This function calculates the value for the sigmoid function for a given Z
            ARGS:
                - Z: Real Number
            RETURNS:
                - sigmoid_function: value calculated by sigmoid function
        """
        sigmoid_function = 1/(1+np.exp(-Z))
        return sigmoid_function
    
if __name__=="__main__":
    sig = Sigmoid(0)
    print(sig.sigmoid(Z=0))