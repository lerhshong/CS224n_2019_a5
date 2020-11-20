#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn
import torch
import torch.nn.utils
import torch.nn.functional as F

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output. That is, e_word
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        pad_token_idx = vocab.char2id['<pad>']
        # num_embeddings = how many 'idxs' we have that is used to code characters. Something like char_vocab_size.
        self.char_dimension = 50
        self.word_dimension = embed_size
        self.max_word_length = 21  # Copied over from utils
        self.dropout_prob = 0.3
        self.embed_size = embed_size
        self.embeddings = nn.Embedding(num_embeddings=len(vocab.id2char), embedding_dim = self.char_dimension,
                                       padding_idx=pad_token_idx)

        self.cnn_layer = CNN(char_dimension = self.char_dimension, max_word_length=self.max_word_length,
                        out_channels = self.word_dimension)
        self.highway_layer = Highway(self.word_dimension)
        self.dropout_layer = nn.Dropout(p=self.dropout_prob)

        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        # Note that the input is of shape (sentence_length, batch_size, max_word_length). Presumably this is the output of
        # to_input_tensor_char method from vocab class.

        sentence_length = input.shape[0]
        batch_size = input.shape[1]

        # Input has shape (sentence_length, batch_size, max_word_length)
        x_emb = self.embeddings(input) # (sentence_length, batch_size, max_word_length, char_dim)

        x_reshape = x_emb.view(-1, self.max_word_length, self.char_dimension) # (sentence_length * batch_size, max_word_length, char_dim)
        x_reshape = x_reshape.permute([0,2,1]) # (sentence_length * batch_size, char_dim, max_word_length)

        x_conv_out = self.cnn_layer.forward(x_reshape) # Apply CNN (sentence_length * batch_size, word_dim)
        x_highway = self.highway_layer.forward(x_conv_out) # (sentence_length * batch_size, word_dim)
        x_word_emb = self.dropout_layer(x_highway) # (sentence_length * batch_size, word_dim)
        x_final = x_word_emb.reshape(sentence_length, batch_size, self.word_dimension)

        return x_final
        ### END YOUR CODE

