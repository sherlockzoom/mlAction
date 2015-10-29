# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 16:41:18 2015
More Examples: logistic regression
@author: knight
"""

import numpy
import theano
import theano.tensor as T
rng = numpy.random

N = 400
feats = 784
D = (rng.rand(N, feats), rng.randint(size=N, low=0, high=2))
traing_steps = 10000

x = T.matrix("x")
y = T.vector("y")

w = theano.shared(rng.randn(feats), name='w')
b = theano.shared(0., name='b')

print 'inital model:'
print w.get_value(), b.get_value()

p_1 = 1/(1 + T.exp(-T.dot(x, w) - b))
prediction = p_1 > 0.5
xent = -y*T.log(p_1) - (1-y)*T.log(1-p_1)
cost = xent.mean() + 0.01*(w**2).sum()
gw,gb = T.grad(cost, [w,b])

# compile

train = theano.function(
    inputs = [x,y],
    outputs=[prediction, xent],
    updates=((w, w - 0.1*gw), (b, b - 0.1*gw))
)

predict = theano.function(inputs=[x], outputs=prediction)

for i in range(traing_steps):
    pred, err = train(D[0], D[1])

print "Final model:" 
print w.get_value(), b.get_value()
print "target values for D: ", D[1]
print "prediction on D: ", predict(D[0])
