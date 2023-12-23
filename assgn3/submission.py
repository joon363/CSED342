#!/usr/bin/python

import random
import collections
import math
import sys
from collections import Counter
from util import *

SEED = 4312

############################################################
# Problem 2: binary classification
############################################################

############################################################
# Problem 2a: feature extraction


def extractWordFeatures(x):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x:
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_ANSWER (our solution is 6 lines of code, but don't worry if you deviate from this)
    
    # split words
    words = x.split()
    word_dict = {}
    # count words
    for word in words:
        word_dict[word] = word_dict.get(word, 0) + 1
    return word_dict

    # END_YOUR_ANSWER


############################################################
# Problem 2b: stochastic gradient descent


def learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta):
    """
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Note:
    1. only use the trainExamples for training!
    You can call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    2. don't shuffle trainExamples and use them in the original order to update weights.
    3. don't use any mini-batch whose size is more than 1
    """
    weights = {}  # feature => weight

    def sigmoid(n):
        return 1 / (1 + math.exp(-n))

    # BEGIN_YOUR_ANSWER (our solution is 14 lines of code, but don't worry if you deviate from this)
    
    # original gradient: 
    # -{(1 - (1+y)/2) * log(1 - sigmoid(phi dot w)) + (1+y)/2 * log(sigmoid(phi dot w)))}
    # derive:
    # -{ (1+y)/2 - sigmoid(phi dot w) } * phi
    # first part: gradiant_vector -> get -{ (1+y)/2 - sigmoid(phi dot w) }
    # second part: increment -> perform *phi

    def gradiant_vector(phi,y):
        return -((1+y)/2 - sigmoid(dotProduct(phi,weights)))

    #gradient descent
    for _ in range(numIters):
        for x,y in trainExamples:
            phi = featureExtractor(x)
            grad = gradiant_vector(phi,y)
            increment(weights, -eta*grad,phi)
        
    # END_YOUR_ANSWER
    return weights


############################################################
# Problem 2c: bigram features


def extractBigramFeatures(x):
    """
    Extract unigram and bigram features for a string x, where bigram feature is a tuple of two consecutive words. In addition, you should consider special words '<s>' and '</s>' which represent the start and the end of sentence respectively. You can exploit extractWordFeatures to extract unigram features.

    For example:
    >>> extractBigramFeatures("I am what I am")
    {('am', 'what'): 1, 'what': 1, ('I', 'am'): 2, 'I': 2, ('what', 'I'): 1, 'am': 2, ('<s>', 'I'): 1, ('am', '</s>'): 1}
    """
    # BEGIN_YOUR_ANSWER (our solution is 5 lines of code, but don't worry if you deviate from this)
    
    # formatting
    x_tag = "<s> "+ x + " </s>"
    words = x.split()
    words_tag = x_tag.split()

    # two words
    two_words = [ (word1, word2) for word1, word2 in zip(words_tag[:-1], words_tag[1:])]
    all_words = words+two_words
    # use counter function for all words
    phi = dict(Counter(all_words))

    # END_YOUR_ANSWER
    return phi
