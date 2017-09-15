import numpy as np
import math

# now, batch=1
class Linear():

    def __init__(self, input_dim, output_dim):
        '''
        Augument:
            input_dim: int
            output_dim: int
        '''
        self.input_dim = input_dim
        self.output_dim = output_dim
        # W: input_dim+1, output_dim
        # initilize
        self.W = (2*np.random.rand(input_dim+1, output_dim)-1)\
                *math.sqrt(6)/math.sqrt(input_dim+output_dim)
        self.W[-1,:]=0
        self.X = None
        self.grad_W = None

    def forward(self, X):
        '''
        Augument:
            X: [batch_size, input_dim] Float
        Return: 
            O = X * W:[batch_size, ouput_dim] Float
        '''
        # Check dimention
        self.batch_size, input_dim = X.shape
        assert input_dim == self.input_dim
        # transfer X to (X, 1)
        self.X = np.concatenate((X, np.ones((batch_size,1))), axis = 1)
        #print 'X shape', X.shape
        #print 'W shape', self.W.shape
        self.output = self.X.dot(self.W)
        return self.output

    def backward(self, grad_in):
        '''
        Augument:
            grad_in: [batch_size, output_dim] Float
        Return: 
            grad_out = [batch_size, input_dim] Float
        '''
        batch_size, output_dim = grad_in.shape
        assert batch_size == self.batch_size
        assert output_dim == self.output_dim
        self.grad_W = self.X.T.dot(grad_in)
        grad_out = grad_in.dot(self.W.T)
        return grad_out

class Softmax_Cross_Entropy():

    def __init__(self):
        self.Softmax = None
        self.Labels = None

    def forward(self, X, Labels):
        '''
        Augument:
            X: [batch_size, input_dim] Float
            Labels: [batch_size, input_dim] one-hot 
        Return:
            Cross_Entropy_Loss: Float
        '''
        self.Labels = Labels
        # Softmax Part
        batch_size, input_dim = X.shape
        X_exp = np.exp(X)
        #print 'X_exp:', X_exp
        self.Softmax = np.transpose(X_exp.T/np.sum(X_exp, axis=1))
        #print 'Softmax_output', Softmax_output

        # Cross Entropy Loss
        LogSoftmax = np.log(self.Softmax)
        #print 'LogSoftmax', LogSoftmax
        loss = - np.sum(LogSoftmax * Labels, axis = 1)
        #print 'loss:', loss
        return np.sum(loss)

    def backward(self):
        '''
        Return:
            grad_out: [batch_size, input_dim] Float
        '''
        grad_out = self.Softmax - self.Labels 
        return grad_out

class Sigmoid():
    def __init__(self):
        self.output = None
        self.batch_size = None
        self.output_dim = None 

    def forward(self, X):
        '''
        Augument:
            X: [batch_size, input_dim] Float
        Return:
            Output: [batch_size, input_dim] Float
        '''
        self.batch_size, self.input_dim = X.shape
        self.output = 1.0/(1.0+np.exp(-X))
        return self.output

    def backward(self, grad_in):
        '''
        Augument:
            grad_in: [batch_size, input_dim] Float
        Return:
            grad_out: [batch_size, input_dim] Float
        '''
        batch_size, output_dim = grad_in.shape
        assert batch_size == self.batch_size
        assert output_dim == self.output_dim
        grad_out = grad_in * self.output * (1-self.output)
        return grad_out


