#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
### YOUR CODE HERE for part 1h

import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F

class Highway(nn.Module):

    '''Implementation of a highway network'''

    def __init__(self, word_dimension):
        '''
        Initializes the network

        :param word_dimension (int): The input dimension. Note in the highway network this is also the output dimension.
        '''

        super(Highway, self).__init__()
        self.word_dimension = word_dimension
        self.proj = nn.Linear(in_features = self.word_dimension, out_features = self.word_dimension)
        self.gate = nn.Linear(in_features = self.word_dimension, out_features = self.word_dimension)


    def forward(self, inputs):
        '''
        A forward pass

        :param inputs (torch tensor): Inputs to the forward passes. It should be of shape word_dimension
        :return: x_highway (torch tensor): The gated output of the highway layer
        '''
        x_proj = self.proj(inputs)
        x_proj = F.relu(x_proj)

        x_gate = self.gate(inputs)
        x_gate = F.sigmoid(x_gate)

        x_highway = x_proj * x_gate + (1 - x_gate) * inputs # '*' is elem-wise multiplication

        return x_highway

### END YOUR CODE

# Sanity checks
def sanity_check_highway():
    # Print shapes, print intermediate values and ensure calculations are what we think it is
    highway_test = Highway(word_dimension=3)
    print('This is the projection matrix:  ' , highway_test.proj.weight)
    print('The shape of the projection matrix is {}'.format(highway_test.proj.weight.shape))
    print('This is the bias of the projection: ', highway_test.proj.bias)

    print('This is the gate matrix: ', highway_test.gate.weight)
    print('The shape of the gate matrix is {}'.format(highway_test.gate.weight.shape))
    print('This is the bias of the gate: ', highway_test.gate.bias)

    testing_inputs =  torch.rand(size = [3]) # Size is word_dimension
    print('This is the testing input: ', testing_inputs)

    output_highway =highway_test.forward(testing_inputs)

    output_test = F.relu(highway_test.proj(testing_inputs)) * F.sigmoid(highway_test.gate(testing_inputs)) + (1-F.sigmoid(highway_test.gate(testing_inputs))) * testing_inputs

    print('The output of the highway layer  is : {}'.format(output_highway))
    print('The output of the manual calculations is: {}'.format(output_test))

    assert(torch.all(torch.eq(output_highway, output_test)))

# sanity_check_highway()





