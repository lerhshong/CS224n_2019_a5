#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i

import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F

class CNN(nn.Module):
    ''' Implementation of the CNN layer'''

    def __init__(self, char_dimension, max_word_length, out_channels, kernel_size = 5):
        '''
        Initializes the CNN layer
        :param char_dimension: The dimension of char embeddings. This is the input channel
        :param max_word_length: Maximum word length, controls kernel size of max pool layer
        :param out_channels: Output channels / number of filters - taken as word dimension here
        :param kernel_size: How big the kernel is
        '''
        super(CNN, self).__init__()
        # out_channels is number of filters (which we've taken as word_dimension)
        # in_channels is the shape of the input, the x_reshape
        # kernel_size is the cnn layer kernel size. The max pool kernel is auto-calculated

        self.cnn_layer = nn.Conv1d(in_channels = char_dimension, out_channels = out_channels,
                                   kernel_size = kernel_size, padding = 0, stride = 1)
        self.max_pool_layer = nn.MaxPool1d(kernel_size = max_word_length - kernel_size + 1) # Max over second layer

    def forward(self,  inputs):
        '''
        Pushes an input through the CNN/Max pool layer
        :param inputs: Inputs to the CNN, x_reshape in the handouts. This should have shape (char_dimension, max_word_length)
        :return:
        '''
        # Inputs should have dimension (char_dimension, k)
        x_conv = self.cnn_layer(inputs) # (word_dimension, max_word_length - k + 1)
        # print('Just applied cnn layer, here is the shape of the output {}'.format(x_conv.shape))
        x_conv = F.relu(x_conv) # (word_dimension, max_word_length - k + 1)
        # print('Just applied RELU, here is the shape of the output {}'.format(x_conv.shape))
        # x_max_pooled = self.max_pool_layer(x_conv)
        # print('Just applied max_pool, here is the shape of the output {}'.format(x_max_pooled.shape))
        x_conv_out = self.max_pool_layer(x_conv).squeeze() # Needs to be squeezed to kill off extra dimension

        return x_conv_out

### END YOUR CODE

def sanity_check_cnn():
    char_dimension = 4
    word_dimension = 3
    max_word_length = 6
    cnn_kernel_size = 2
    cnn_test = CNN(char_dimension=char_dimension, max_word_length = max_word_length, out_channels = word_dimension,
                   kernel_size = cnn_kernel_size)

    test_inputs = torch.rand(1, char_dimension, max_word_length) # 1 is batch_size

    print('The testing inputs are:')
    print('-' * 80)
    print(test_inputs)

    print('The cnn layer weight is:')
    print('-' * 80)
    print(cnn_test.cnn_layer.weight.shape)
    print(cnn_test.cnn_layer.weight)

    print('The cnn layer bias is:')
    print('-' * 80)
    print(cnn_test.cnn_layer.bias.shape)
    print(cnn_test.cnn_layer.bias)

    x_conv_out = cnn_test.forward(inputs = test_inputs)

    print('The result of the convolutional forward step is:')
    print('-'*80)
    print(x_conv_out)
    print('With shape {}'.format(x_conv_out.shape))

    # This doesn't work because we still have ReLUs and MaxPool later.

    # x_manual_conv = 0
    # for i in range(char_dimension):
    #     for j in range(cnn_kernel_size):
    #         x_manual_conv += test_inputs[0][i][j] * cnn_test.cnn_layer.weight[0][i][j]
    # x_manual_conv += cnn_test.cnn_layer.bias[0]
    # print('The first convolution result is: ')
    # print('-' * 80)
    # print(x_manual_conv)

# sanity_check_cnn()