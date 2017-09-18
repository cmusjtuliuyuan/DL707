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
        # W: input_dim+1, output_dim
        # initilize
        self.W = (2*np.random.rand(input_dim+1, output_dim)-1)\
                *math.sqrt(6)/math.sqrt(input_dim+output_dim)
        self.W[-1,:]=0
        self.X = None
        self.grad_W = .0
        self.grad_W_pre = .0

    def forward(self, X):
        '''
        Augument:
            X: [batch_size, input_dim] Float
        Return: 
            O = X * W:[batch_size, ouput_dim] Float
        '''
        # transfer X to (X, 1)
        self.X = np.concatenate((X, np.ones((X.shape[0],1))), axis = 1)
        self.output = self.X.dot(self.W)
        return self.output

    def backward(self, grad_in):
        '''
        Augument:
            grad_in: [batch_size, output_dim] Float
        Return: 
            grad_out = [batch_size, input_dim] Float
        '''
        self.grad_W = self.X.T.dot(grad_in)
        grad_out = grad_in.dot(self.W.T)
        return grad_out[:, : -1]

    def update(self, learning_rt, momentum = .0, alpha = .0):
        self.grad_W += momentum * self.grad_W_pre
        self.grad_W_pre = self.grad_W
        self.W = self.W - learning_rt*self.grad_W - alpha * self.W

class Softmax_Cross_Entropy():

    def __init__(self):
        self.Softmax = None
        self.Labels = None

    def forward(self, X):
        '''
        Augument:
            X: [batch_size, input_dim] Float
            Labels: [batch_size, input_dim] one-hot 
        Return:
            LogSoftmax: [batch_size, input_dim] Float
        '''
        # Softmax Part
        X_exp = np.exp(X)
        self.Softmax = np.transpose(X_exp.T/np.sum(X_exp, axis=1))
        # Cross Entropy Loss
        LogSoftmax = np.log(self.Softmax)

        return LogSoftmax

    def backward(self, Labels):
        '''
        Return:
            grad_out: [batch_size, input_dim] Float
        '''
        grad_out = self.Softmax - Labels 
        return grad_out

    def get_loss(self, X, Labels):
        # TODO delete one
        LogSoftmax = self.forward(X)
        loss = - np.sum(LogSoftmax * Labels, axis = 1)
        return np.sum(loss)

class Sigmoid():
    def __init__(self):
        self.output = None

    def forward(self, X):
        '''
        Augument:
            X: [batch_size, input_dim] Float
        Return:
            Output: [batch_size, input_dim] Float
        '''
        self.output = 1.0/(1.0+np.exp(-X))
        return self.output

    def backward(self, grad_in):
        '''
        Augument:
            grad_in: [batch_size, input_dim] Float
        Return:
            grad_out: [batch_size, input_dim] Float
        '''
        batch_size, dim = grad_in.shape
        grad_out = grad_in * self.output * (1-self.output)
        return grad_out

class NN_3_layer():
    def __init__(self, hidden_dim = 100):
        self.layer1 = Linear(784, hidden_dim)
        self.act = Sigmoid()
        self.layer2 = Linear(hidden_dim, 10)
        self.loss = Softmax_Cross_Entropy()

    def get_NLL_loss(self, X, Labels):
        '''
        Augument:
            X: [batch_size, input_dim] Float
            Labels: [batch_size, input_dim] one-hot 
        Return: 
            loss: float
        '''
        o1 = self.layer1.forward(X)
        h1 = self.act.forward(o1)
        o2 = self.layer2.forward(h1)
        loss = self.loss.get_loss(o2, Labels)
        return loss

    def backward(self, Labels, learning_rt, momentum = .0, alpha = .0):
        '''
        Augument:
            learning_rt: float
        '''
        grad_out_loss = self.loss.backward(Labels)
        grad_out_layer2 = self.layer2.backward(grad_out_loss)
        self.layer2.update(learning_rt, momentum, alpha)
        grad_out_act = self.act.backward(grad_out_layer2)
        grad_out_layer1 = self.layer1.backward(grad_out_act)
        self.layer1.update(learning_rt, momentum, alpha)
        return grad_out_layer1

    def get_IC_loss(self, X, Labels):
        '''
        Augument:
            X: [batch_size, input_dim] Float
            Labels: [batch_size, input_dim] one-hot 
        Return:
            loss: int
        '''
        batch_size, dim = Labels.shape
        o1 = self.layer1.forward(X)
        h1 = self.act.forward(o1)
        o2 = self.layer2.forward(h1)
        predict = np.argmax(self.loss.forward(o2), axis=1)
        predict_one_hot = np.zeros((batch_size, dim))
        predict_one_hot[np.arange(batch_size), predict] = 1

        loss = batch_size - np.sum(predict_one_hot*Labels)

        return loss