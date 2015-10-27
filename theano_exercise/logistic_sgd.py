# -*- coding=utf-8 -*-
import theano.tensor as T
import theano
import numpy

import cPickle
import gzip
import os
import sys
import timeit

class LogisticRegression(object):
    """多类别逻辑回归"""
    def __init__(self, input, n_in, n_out):
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        # 权重矩阵初始化为0,大小为(n_in, n_out)
        self.W = theano.shared(
            value = numpy.zeros(
                (n_in, n_out),
                dtype = theano.config.floatX
            ),
            name = 'W',
            borrow = True
        )
        # 初始化偏置向量为0
        self.b = theano.shared(
            value = numpy.zeros(
                (n_out,),
                dtype = theano.config.floatX
            ),
            name = 'b',
            borrow = True
        )
        # 调用softmax
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        # 计算最大概率的类 argmax
        self.y_pred = T.argmax(self.p_y_given_x, axis = 1)
        # 模型参数
        self.params = [self.W, self.b]
        # 模型输入
        self.input = input

    def negative_log_likelihood(self, y):
        """返回负的log似然值"""
        # start-snippet-2
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """
        # 检查y和y_pred 是否相同维度
        if y.ndim != self.y_pred.ndim:  
            raise TypeError(  
                'y should have the same shape as self.y_pred',  
                ('y', y.type, 'y_pred', self.y_pred.type)  
            )  
        # 检查y是否有正确数据类型
        if y.dtype.startswith('int'):
            # T.neq 返回0/1 ，1代表错误的预测即不相等
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

def load_data(dataset):
    ''' 加载数据'''
    # 下载MNIST数据 如果不存在
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
	        dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print 'Downloading data from %s '%origin
        urllib.urlretrieve(origin, dataset)

    print '... loading data'
    # load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    def shared_dataset(data_xy, borrow=True):
	    """ Function that loads the dataset into shared variables
	    The reason we store our dataset in shared variables is to allow
	    Theano to copy it into the GPU memory (when code is run on GPU).
	    Since copying data into the GPU is slow, copying a minibatch everytime
	    is needed (the default behaviour if the data is not in a shared
	    variable) would lead to a large decrease in performance.
	    """
	    data_x, data_y = data_xy
	    shared_x = theano.shared(numpy.asarray(data_x, dtype = theano.config.floatX), borrow = borrow)
	    shared_y = theano.shared(numpy.asarray(data_y, dtype = theano.config.floatX), borrow = borrow)
	    return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    return rval

def sgd_optimization_mnist(learning_rate=0.13, n_epochs=1000, dataset='mnist.pkl.gz', batch_size=600):
    """随机梯度优化
        This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz
    """
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # 计算minibatch数量
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]/batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]/batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]/batch_size
    print '... building the model'
    # index to a [mini]batch
    index = T.lscalar()

    x = T.matrix('x')
    y = T.ivector('y')
    # 构建逻辑回归类
    # 每个MNIST图像大小是28*28
    classifier = LogisticRegression(input=x, n_in=28*28, n_out=10)
    # 损失最小等于负log似然值
    cost = classifier.negative_log_likelihood(y)
    # 编译Theano 函数计算错误率
    test_model = theano.function(
        inputs = [index],
        outputs = classifier.errors(y),
        givens = {
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
    	inputs = [index],
    	outputs=classifier.errors(y),
    	givens={
    		x: valid_set_x[index*batch_size: (index + 1 )*batch_size],
    		y: valid_set_y[index*batch_size: (index + 1 )*batch_size]
    	}
    )
    # 计算梯度
    g_w = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)
    # updates 更新
    updates = [(classifier.W, classifier.W - learning_rate * g_w),
            (classifier.b, classifier.b - learning_rate * g_b)]

    train_model = theano.function(
        inputs = [index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index*batch_size: (index + 1)*batch_size],
            y: train_set_y[index*batch_size: (index + 1)*batch_size]
        }
    )

    print '... training the model'
    # 提前停止参数更新
    patience = 5000
    patience_increase = 2
    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches, patience / 2)
    # inf: 无穷
    best_validation_loss = numpy.inf
    test_score = 0
    start_time = timeit.default_timer()
    done_looping = False
    epoch = 0
    while(epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)
            # 迭代次数
            iter = (epoch - 1) * n_train_batches + minibatch_index
            if (iter + 1) % validation_frequency == 0:
                validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print (
                    'epoch %i, minibatch %i/%i, validation error%f %%'%
                    (
                    	epoch, 
                    	minibatch_index+1, 
                    	n_train_batches,
                    	this_validation_loss*100.
                    )
                )
         # if we got the best validation score until now
        if this_validation_loss < best_validation_loss:
            #improve patience if loss improvement is good enough
            if this_validation_loss < best_validation_loss *  \
               improvement_threshold:
                patience = max(patience, iter * patience_increase)

            best_validation_loss = this_validation_loss
            # test it on the test set

            test_losses = [test_model(i)
                           for i in xrange(n_test_batches)]
            test_score = numpy.mean(test_losses)

            print(
                (
                    '     epoch %i, minibatch %i/%i, test error of'
                    ' best model %f %%'
                ) %
                (
                    epoch,
                    minibatch_index + 1,
                    n_train_batches,
                    test_score * 100.
                )
            )

            # save the best model
            with open('best_model.pkl', 'w') as f:
                cPickle.dump(classifier, f)

        if patience <= iter:
            done_looping = True
            break

    end_time = timeit.default_timer()
    print(
        (
            'Optimization complete with best validation score of %f %%,'
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., test_score * 100.)
    )
    print 'The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.1fs' % ((end_time - start_time)))

def predict():
    """
    An example of how to load a trained model and use it
    to predict labels.
    """

    # load the saved model
    classifier = cPickle.load(open('best_model.pkl'))

    # compile a predictor function
    predict_model = theano.function(
        inputs=[classifier.input],
        outputs=classifier.y_pred)

    # We can test it on some examples from test test
    dataset='mnist.pkl.gz'
    datasets = load_data(dataset)
    test_set_x, test_set_y = datasets[2]
    test_set_x = test_set_x.get_value()

    predicted_values = predict_model(test_set_x[:10])
    print ("Predicted values for the first 10 examples in test set:")
    print predicted_values


if __name__ == '__main__':
    sgd_optimization_mnist()
