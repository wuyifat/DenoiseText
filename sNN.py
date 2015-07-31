import numpy
import theano
import theano.tensor as T
from PIL import Image
import os
import pylab
import cPickle
from scipy import signal

class Layer(object):
    def __init__(self, rng, inp, n_in, n_out, activation=T.tanh):
        W_values = numpy.asarray(rng.uniform(low=-numpy.sqrt(6.0 / (n_in + n_out)),
                               high=numpy.sqrt(6.0 / (n_in + n_out)),
                               size=(n_in, n_out)),dtype=theano.config.floatX)
        self.W = theano.shared(value=W_values, name='W', borrow=True)

        b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name='b', borrow=True)

        self.input = inp
        if activation == None:
            self.output = T.dot(self.input, self.W) + self.b
        else:
            self.output = activation(T.dot(self.input, self.W) + self.b)
        self.params = [self.W, self.b]


class Model(object):
    def __init__(self, rng, inp=None, n_in=121, n_hidden=[6,3], n_out=1):
        self.layer1 = Layer(rng=rng,inp=inp,n_in=n_in,n_out=n_hidden[0],activation=T.tanh)
        self.layer2 = Layer(rng=rng,inp=self.layer1.output,n_in=n_hidden[0],n_out=n_hidden[1],activation=T.tanh)
        self.layer3 = Layer(rng=rng,inp=self.layer2.output,n_in=n_hidden[1],n_out=n_out,activation=T.tanh)

        self.input = inp
        self.output = self.layer3.output
        self.params = self.layer1.params+self.layer2.params+self.layer3.params

    def cost(self,y):
        return T.mean((self.output-y)**2)

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

def load_test_data(fname):
    data_x = numpy.asarray(Image.open('test/'+fname),dtype='float32')/255.
    data_x = denoise_filter(data_x)
    data_x = enlarged(data_x,5)
    data_set_x = sub_image(data_x,5)
    data_set_x = theano.shared(numpy.asarray(data_set_x,dtype=theano.config.floatX),borrow=True)
    return data_set_x

def test_mlp(batch_size=540,learning_rate=0.1,n_epochs=10):
    print '... getting the data path'
    file_list = os.listdir('train')[1:]
    file_test = os.listdir('test')[1:]
    file_train = file_list[0:1]
    file_valid = file_list[1:1]
    rng = numpy.random.RandomState(12345)

    x = T.matrix('x')
    y = T.matrix('y')
    index = T.lscalar('index')
    
    print '... building the model'
    model = Model(rng=rng,inp=x,n_in=121,n_hidden=[16,3],n_out=1)

    gparams = T.grad(model.cost(y), model.params)
    updates = [(param, param - learning_rate*gparam)
               for param, gparam in zip(model.params, gparams)]
    print '... training the model'
    for i in xrange(len(file_train)):
        print '... training the %d-th image: ' % (i+1)
        train_set_x,train_set_y = load_train_data(file_train[i])
        train_model = theano.function(inputs=[index],outputs=model.cost(y),updates=updates,givens={x:train_set_x[index:(index+1)*batch_size],y:train_set_y[index:(index+1)*batch_size]})
        n_train_batches = train_set_x.get_value(borrow=True).shape[0]/batch_size
        for epoch in range(n_epochs):
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(numpy.sqrt(train_model(batch_index)))
            print 'Epoch %d, training error: '% epoch,numpy.mean(c)
    print '... validating the model'
    for i in xrange(len(file_valid)):
        print '... validating the %d-th image: ' % (i+1)
        c = []
        valid_set_x,valid_set_y = load_train_data(file_valid[i])
        valid_model = theano.function(inputs=[index],outputs=model.cost(y),givens={x:valid_set_x[index:(index+1)*batch_size],y:valid_set_y[index:(index+1)*batch_size]})
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]/batch_size
        for batch_index in xrange(n_valid_batches):
            c.append(numpy.sqrt(valid_model(batch_index)))
        print 'Validating Error: ', numpy.mean(c)
    print '... testing the model'
    test_model = theano.function(inputs=[x],outputs=model.output)
    for i in xrange(len(file_test[0:1])):
        test_set_x = load_test_data(file_test[i])
        test_set_y_pred = []
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]/batch_size
        test_model = theano.function(inputs=[index],outputs=model.output,givens={x:test_set_x[index:(index+1)*batch_size]})
        for batch_index in xrange(n_test_batches):
            y_pred = test_model(batch_index)
            array = numpy.zeros((batch_size,))
            for j in xrange(batch_size):
                array[j] = y_pred[j][0]
            test_set_y_pred.append(array)
        test_set_y_pred = (numpy.asarray(test_set_y_pred))*255.
        img = Image.fromarray(numpy.uint8(test_set_y_pred))
        pylab.imshow(img,cmap=pylab.gray())
        pylab.show()


if __name__ == '__main__':
    test_mlp()
