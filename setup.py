# -*- coding: utf-8 -*-
"""
Created on Tue May 26 13:14:35 2020

@author: PRAVEEN KUMAR -1
"""

from setuptools import setup,find_packages

setup(
      name="basic-nn",
      version="0.1.0",
      author="Praveen Kumar",
      author_email="kr.praveen399@gmail.com",
      description="Implementation of neural network from scratch",
      url="https://github.com/ds-praveenkumar/coursera",
      packages=find_packages("basic-neural-network,basic-neural-network.utility"),
      install_requires=["pandas",
                        "numpy"
                        ],
      python_requires='>=3.6'
      
      
      )