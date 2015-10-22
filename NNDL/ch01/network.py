#!/usr/bin/env python
# coding=utf-8
import numpy as np

class NetWork(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.size = sizes
        self.biases = [np.ramdom.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.ramdom.randn(y, x) for x,y in zip(sizes[:-1], sizes[1:])]
