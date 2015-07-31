import os
import sys

import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

import pylab
from PIL import Image
from scipy import signal

class regressionLayer(object):
    def __init__(self,rng=None,input=None,n_in=121,n_out=1,W=None,b=None,activation=T.tanh):
        if W is None:
            initial_W = numpy.asarray(rng.uniform(low=-(1.0/(n_in+n_out)),high=(1.0/(n_in+n_out)),size=(n_in,n_out)),dtype=theano.config.floatX)
            W = theano.shared(value=initial_W,name='W',borrow=True)
        if b is None:
            initial_b = numpy.zeros((n_out,),dtype=theano.config.floatX)
            b = theano.shared(value=initial_b,name='b',borrow=True)
        self.W = W
        self.b = b
        self.input = input
        if activation == None:
            self.output = T.dot(self.input,self.W)+self.b
        else:
            self.output = activation(T.dot(self.input,self.W)+self.b)
        self.params = [self.W,self.b]

class daLayer(object):
    def __init__(self,rng=None,theano_rng=None,input=None,n_in=121,n_hidden=400,W=None,bhid=None,activation=None):
        self.n_visible = n_in
        self.n_hidden = n_hidden
        if not theano_rng:
            theano_rng = RandomStreams(rng.randint(2**30))
        if W is None:
            initial_W = numpy.asarray(rng.uniform(low=-4*numpy.sqrt(6./(n_hidden+n_in)),high=4.*numpy.sqrt(6./(n_hidden+n_in)),size=(n_in,n_hidden)),dtype=theano.config.floatX)
            W = theano.shared(value=initial_W,name='W',borrow=True)
        if bhid is None:
            bhid = theano.shared(value=numpy.zeros(n_hidden,dtype=theano.config.floatX),name='bvis',borrow=True)
        initial_W_prime = numpy.asarray(rng.uniform(low=-4*numpy.sqrt(6./(n_hidden+n_in)),high=4.*numpy.sqrt(6./(n_hidden+n_in)),size=(n_hidden,n_in)),dtype=theano.config.floatX)
        W_prime = theano.shared(value=initial_W_prime,name='W_prime',borrow=True)
        bvis = theano.shared(value=numpy.zeros(n_in,dtype=theano.config.floatX),name='bhid',borrow=True)
        self.W = W
        self.bvis = bvis
        self.W_prime = W_prime
        self.bhid = bhid
        self.theano_rng = theano_rng
        if input is None:
            self.x = T.matrix('x')
        else:
            self.x = input
        self.params = [self.W,self.W_prime,self.bvis,self.bhid]
        if activation is None:
            z = T.dot(self.x,self.W)+self.bhid
            self.output = T.dot(z,self.W_prime)+self.bvis
        else:
            z = activation(T.dot(self.x,self.W)+self.bhid)
            self.output = activation(T.dot(z,self.W_prime)+self.bvis)
    def get_cost_updates(self,learning_rate):
        cost = T.mean((self.output-self.x)**2)
        gparams = T.grad(cost,self.params)
        updates = [(param,param-learning_rate*gparam)for param,gparam in zip(self.params,gparams)]
        return (cost,updates)

class SdA(object):
    def __init__(self,rng=None,theano_rng=None,n_in=121,hidden_layers_sizes=[400,400,400],n_hidden=6,n_out=1):
        self.dA_layers = []
        self.sigmoid_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)
        assert self.n_layers > 0
        if not theano_rng:
            theano_rng = RandomStreams(rng.randint(2**30))
        self.x = T.matrix('x')
        self.y = T.matrix('y')
        for i in xrange(self.n_layers):
            if i == 0:
                input_size = n_in
                layer_input = self.x
            else:
                input_size = hidden_layers_sizes[i-1]
                layer_input = self.sigmoid_layers[-1].output
            sigmoid_layer = regressionLayer(rng=rng,input=layer_input,n_in=input_size,n_out=hidden_layers_sizes[i],activation=T.tanh)
            self.sigmoid_layers.append(sigmoid_layer)
            self.params.extend(sigmoid_layer.params)
            dA_layer = daLayer(rng=rng,theano_rng=theano_rng,input=layer_input,n_in=input_size,n_hidden=hidden_layers_sizes[i],W=sigmoid_layer.W,bhid=sigmoid_layer.b,activation=T.tanh)
            self.dA_layers.append(dA_layer)
        self.reg_layer1 = regressionLayer(rng=rng,input=self.sigmoid_layers[-1].output,n_in=hidden_layers_sizes[-1],n_out=n_hidden)
        self.reg_layer2 = regressionLayer(rng=rng,input=self.reg_layer1.output,n_in=n_hidden,n_out=n_out)
        self.params.extend(self.reg_layer1.params)
        self.params.extend(self.reg_layer2.params)
        self.output = self.reg_layer2.output
        self.errors = T.mean((self.output-self.y)**2)
    def pretraining_functions(self,data_x,batch_size):
        index = T.lscalar('index')
        learning_rate = T.scalar('lr')
        batch_begin = index*batch_size
        batch_end = batch_begin+batch_size
        pretrain_fns = []
        for dA in self.dA_layers:
            cost,updates = dA.get_cost_updates(learning_rate)
            fn = theano.function(inputs=[index,theano.Param(learning_rate,default=0.1)],outputs=cost,updates=updates,givens={self.x:data_x[batch_begin:batch_end]})
            pretrain_fns.append(fn)
        return pretrain_fns

    def finetune_functions(self,datasets,batch_size,learning_rate):
        train_x,train_y = datasets
        n_train_batches = train_x.get_value(borrow=True).shape[0]/batch_size
        index = T.lscalar('index')
        gparams = T.grad(self.errors,self.params)
        updates = [(param,param-learning_rate*gparam)for param,gparam in zip(self.params,gparams)]
        train_fn = theano.function(inputs=[index],outputs=self.errors,updates=updates,givens={self.x:train_x[index*batch_size:(index+1)*batch_size],self.y:train_y[index*batch_size:(index+1)*batch_size]})
        return train_fn

def enlarged(image,neighbors):
    height, width = image.shape
    enlarged = numpy.zeros((height+2*neighbors,width+2*neighbors))
    
    enlarged[:neighbors,:neighbors] = image[0,0]
    enlarged[:neighbors,-neighbors:] = image[0,-1]
    enlarged[-neighbors:,:neighbors] = image[-1,0]
    enlarged[-neighbors:,-neighbors:] = image[-1,-1]
    
    enlarged[:neighbors,neighbors:-neighbors] = image[0,:]
    enlarged[neighbors:-neighbors, :neighbors] = image[:,0][numpy.newaxis].T
    enlarged[-neighbors:,neighbors:-neighbors] = image[-1,:]
    enlarged[neighbors:-neighbors,-neighbors:] = image[:,-1][numpy.newaxis].T
    
    enlarged[neighbors:-neighbors,neighbors:-neighbors] = image
    
    return enlarged

def sub_image(image,size=5):
    subimages = []
    height,width = image.shape
    for i in xrange(size,height-size):
        for j in xrange(size,width-size):
            subimages.append(image[i-size:i+size+1,j-size:j+size+1].flatten())
    return subimages

def denoise_filter(image):
    image_background = signal.medfilt2d(image,11)
    image_mask = image<image_background-0.1
    image_filter = numpy.where(image_mask,image,1.0)
    return image_filter

def load_train_data(fname):
    data_x = numpy.asarray(Image.open('train/'+fname),dtype='float32')/255.
    data_y = numpy.asarray(Image.open('train_cleaned/'+fname),dtype='float32')/255.
    data_x = denoise_filter(data_x)
    data_x = enlarged(data_x,5)
    data_set_x = sub_image(data_x,5)
    data_set_y = []
    data_set_y.append(data_y.flatten())
    data_set_x = theano.shared(numpy.asarray(data_set_x,dtype=theano.config.floatX),borrow=True)
    data_set_y = theano.shared(numpy.asarray(data_set_y,dtype=theano.config.floatX).T,borrow=True)
    return (data_set_x,data_set_y)

def test_SdA(finetune_lr=0.1,pretraining_epochs=10,pretrain_lr=0.001,training_epochs=20,batch_size=540):
    train_x,train_y = load_train_data('101.png')
    n_train_batches = train_x.get_value(borrow=True).shape[0]/batch_size
    rng = numpy.random.RandomState(89677)
    print '... building the model'
    sda = SdA(rng=rng,n_in=121,hidden_layers_sizes=[1000,1000,1000],n_hidden=6,n_out=1)
    print '... getting the pretraining functions'
    pretraining_fns = sda.pretraining_functions(data_x=train_x,batch_size=batch_size)
    print '... pre-training the model'
    for i in xrange(sda.n_layers):
        for epoch in xrange(pretraining_epochs):
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(numpy.sqrt(pretraining_fns[i](batch_index)))
            print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
            print numpy.mean(c)
    print '... getting the finetuning functions'
    train_fn = sda.finetune_functions(datasets=(train_x,train_y),batch_size=batch_size,learning_rate=finetune_lr)
    print '... finetuning the model'
    for i in xrange(training_epochs):
        c= []
        for batch_index in xrange(n_train_batches):
            c.append(numpy.sqrt(train_fn(batch_index)))
        print 'Trainging Error of epoch %d' % (i+1),numpy.mean(c)

def test_dA(learning_rate=0.1,n_epochs=10,batch_size=540):
    data_x,data_y = load_train_data('101.png')
    rng = numpy.random.RandomState(89677)
    index = T.lscalar('index')
    x = T.matrix('x')
    dA = daLayer(rng=rng,input=x,n_in=121,n_hidden=400,activation=T.tanh)
    n_train_batches = data_x.get_value(borrow=True).shape[0]/batch_size
    cost,updates = dA.get_cost_updates(learning_rate)
    train_model = theano.function(inputs=[index],outputs=cost,updates=updates,givens={x:data_x[index*batch_size:(index+1)*batch_size]})
    for epoch in xrange(n_epochs):
        c = []
        for batch_index in xrange(n_train_batches):
            c.append(numpy.sqrt(train_model(batch_index)))
        print 'Training erro of %d epoch: ' %(epoch+1),numpy.mean(c)

if __name__ == '__main__':
    test_SdA()












