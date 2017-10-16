import numpy as np
from nn import *
from RBM import *

BATCH_SIZE = 32
train_data = np.genfromtxt('data/digitstrain.txt', delimiter=',')
valid_data = np.genfromtxt('data/digitsvalid.txt', delimiter=',')


class Autoencoder():
    def __init__(self, hidden_dim, Denoise = False):
        self.hidden_dim = hidden_dim
        self.Denoise = Denoise
        if Denoise:
            self.drop_out = Drop_out(0.1)
        self.encoder = Linear(784, hidden_dim)
        self.act1 = Sigmoid()
        self.decoder = Linear(hidden_dim, 784)
        self.act2 = Sigmoid()
        self.loss = MSE()

    def forward(self, X):
        '''
        Augument:
            X: [batch_size, input_dim]
        Return:
            h2: [batch_size, input_dim]
        '''
        if self.Denoise:
            X = self.drop_out.forward(X)
        o1 = self.encoder.forward(X)
        h1 = self.act1.forward(o1)
        o2 = self.decoder.forward(h1)
        h2 = self.act2.forward(o2)
        return h2

    def get_loss(self, X):
        o = self.forward(X)
        loss = self.loss.get_loss(o, X)
        return loss

    def backward(self,learning_rt, momentum = .0, alpha = .0):
        grad_out_loss = self.loss.backward()
        grad_out_act2 = self.act2.backward(grad_out_loss)
        grad_out_decoder = self.decoder.backward(grad_out_act2)
        self.decoder.update(learning_rt, momentum, alpha)
        grad_out_act1 = self.act1.backward(grad_out_decoder)
        grad_out_encoder = self.encoder.backward(grad_out_act1)
        self.encoder.update(learning_rt, momentum, alpha)
        return grad_out_encoder

def train_one_epoch(model, data, learning_rt=0.01, train = True):

    def train_one_batch(model, batch_data, learning_rt, train = True):
        batch_size = len(batch_data)
        batch_data = np.array(batch_data)
        X = batch_data[:,:-1]
        loss = model.get_loss(X)
        if train:
            model.backward(learning_rt)
        return np.sum(loss)

    batch_data = []
    loss_sum = .0
    for i, index in enumerate(np.random.permutation(len(data))):
        batch_data.append(data[index])
        if len(batch_data) == BATCH_SIZE:
            loss_sum += train_one_batch(model, batch_data, learning_rt, train)
            batch_data = []

    if len(batch_data) != 0:
        loss_sum += train_one_batch(model, batch_data, learning_rt, train)
    return loss_sum/len(data)


def problem_e_f(Denoise = False):
    train_loss_no_pretrain, valid_loss_no_pretrain, _ = get_loss_one_time(NLL=False)
    model = Autoencoder(hidden_dim = 100, Denoise = Denoise)
    for i in range(50):
        print 'Epoch number:', i
        train_one_epoch(model, train_data, 0.1)
    for i in range(10):
        print 'Epoch number:', 20+i
        train_one_epoch(model, train_data, 0.01)

    fig = plot_10X10_figure(model.encoder.W[:-1,:].T)
    if Denoise:
        fig.savefig('Denoise_Autoencoder_W_visulize.png')
    else:
        fig.savefig('Autoencoder_W_visulize.png')
    train_loss_pretrain, valid_loss_pretrain, _ = get_loss_one_time(NLL=False, W = model.encoder.W[:-1,:].T)

    # Plot
    fig = plt.figure()
    plt.plot(valid_loss_no_pretrain,'g-',label='valid_IC_ratio_no_pretrain')
    plt.plot(valid_loss_pretrain,'r-',label='valid_IC_ratio_pretrain')

    plt.xlabel('epoches')
    plt.ylabel('loss')
    plt.title('incorrect classification_error')

    plt.grid(True)
    plt.legend()
    if Denoise:
        fig.savefig('Denoise_Autoencoder_compare.png')
    else:
        fig.savefig('Autoencoder_compare.png')

#problem_e_f(Denoise = True)
def problem_g(hidden_dim):
    Train_loss_array = []
    Valid_loss_array = []
    model = Autoencoder(hidden_dim = hidden_dim)
    for i in range(500):
        print 'Epoch number:', i
        Train_loss_array.append(train_one_epoch(model, train_data, train=False))
        Valid_loss_array.append(train_one_epoch(model, valid_data, train=False))
        train_one_epoch(model, train_data, 0.01)

    # Plot
    fig = plt.figure()
    plt.plot(Train_loss_array,'g-',label='Train Mean Square Error')
    plt.plot(Valid_loss_array,'r-',label='Valid Mean Square Error')

    plt.xlabel('epoches')
    plt.ylabel('loss')
    plt.title('Mean Squre Error, hidden_dim=%d'%(hidden_dim,))
    plt.ylim(0,25)
    plt.grid(True)
    plt.legend()

    fig.savefig('problem_g_hidden%d_%f.png'%(hidden_dim, min(Valid_loss_array)))

#problem_g(100)




