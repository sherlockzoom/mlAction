#!/usr/bin/env 
from theano.tensor.nnet import conv
import theano
from theano import tensor as T
import numpy
import pylab
from PIL import Image
from theano.tensor.signal import downsample
from logsitic_sgd import LogisticRegression,load_data
from mlp import HiddenLayer

rng = numpy.random.RandomState(23455)
# print rng
input = T.tensor4(name='input')
w_shp = (2, 3, 9, 9)
W_bound = numpy.sqrt(3*9*9)
W = theano.shared(numpy.asarray(
        rng.uniform(
            low=-1.0/W_bound,
            high=1.0/W_bound,
            size=w_shp),
        dtype=input.dtype), name='W')
b_shp = (2,)
b=theano.shared(numpy.asarray(
        rng.uniform(low=-.5, high=.5, size=b_shp),
        dtype=input.dtype), name='b')

conv_out = conv.conv2d(input, W)

output = T.nnet.sigmoid(conv_out + b.dimshuffle('x', 0, 'x', 'x'))

f = theano.function([input], output)

img = Image.open(open('./3wolfmoon.jpg'))
img = numpy.asarray(img, dtype='float64')/256.

img_ = img.swapaxes(0, 2).swapaxes(1, 2).reshape(1, 3, 639, 516)
filtered_img = f(img_)
pylab.subplot(1, 3, 1); pylab.axis('off'); pylab.imshow(img)
pylab.gray()

pylab.subplot(1, 3, 2); pylab.axis('off');pylab.imshow(filtered_img[0, 0, :, :])
pylab.subplot(1, 3, 3); pylab.axis('off');pylab.imshow(filtered_img[0, 1, :, :])
pylab.show()

class LeNetConvPoolLayer(object):
    """docstring for LeNetConvPoolLayer"""
    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2,2)):
        # super(LeNetConvPoolLayer, self).__init__()
        assert image_shape[1] == filter_shape[1]
        self.input = input
        fan_in = numpy.prod(filter_shape[1:])
        fan_out = (filter_shape[0]*numpy.prod(filter_shape[2:])/
                numpy.prod(poolsize))
        W_bound = numpy.sqrt(6./(fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        conv_out = conv.conv2d(input=input, 
                                filters=self.W, 
                                filter_shape=filter_shape,
                                image_shape=image_shape)
        
        pooled_out = downsample.max_pool_2d(input=conv_out, 
                                            ds=poolsize,
                                            ignore_border=True)

        self.out = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.params = [self.W, self.b]
        x = T.matrix('x')
        y = T.ivector('x')
        print '... building the model'
        layer0_input = x.reshape((batch_size, 1, 28, 28))
        layer0 = LeNetConvPoolLayer(rng, 
                                    input=layer0_input, 
                                    filter_shape=(nkerns[0], 1, 5, 5), 
                                    image_shape=(batch_size, 1, 28, 28),
                                    poolsize=(2,2)
                                    )
        layer1 = LeNetConvPoolLayer(rng, 
                                    input=layer0.output, 
                                    filter_shape=(nkers[1], nkerns[0], 5, 5), 
                                    image_shape=(batch_size, nkers[0], 12, 12),
                                    poolsize=(2,2)
                                    )
        layer2_input = layer1.output.flatten(2)
        layer2 = HiddenLayer(
            rng,
            input=layer2_input,
            n_in=nkerns[1]*4*4,
            n_out=500,
            activation=T.tanh
            )
        layer3 = LogisticRegression(input=layer2_output, n_in=500, n_out=10)
        cost = layer3.negative_log_likelihood(y)
        test_model = theano.function(
            [index],
            layer3.errors(y),
            givens={
                x:test_set_x[index*batch_size: (index+1)*batch_size],
                y:test_set_y[index*batch_size: (index+1)*batch_size]
            }
        )

        validate_model = theano.function(
            [index],
            layer3.errors(y),
            givens={
                x:valid_set_x[index*batch_size: (index+1)*batch_size],
                y:valid_set_y[index*batch_size: (index+1)*batch_size]
            }
        )
        params = layer3.params + layer2.params + layer1.params + layer0.params
        grads = T.grad(cost, params)
        updates = [
            (params_i, params_i - learning_rate*grad_i)
            for params_i, grad_i in zip(params, grads)
        ]
        train_model = theano.function(
            [index],
            cost,
            updats=updates,
            givens={
                x: train_set_x[index*batch_size: (index+1)*batch_size],
                y: train_set_y[index*batch_size: (index+1)*batch_size]
            }
        )

        