import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math
from nn import NN_3_layer, Sigmoid

BATCH_SIZE = 32
train_data = np.genfromtxt('data/digitstrain.txt', delimiter=',')
valid_data = np.genfromtxt('data/digitsvalid.txt', delimiter=',')

def get_bernoulli_sample(p):
    h = np.random.rand(*p.shape)
    return (h<p).astype(float)

def sigm(x):
    return 1/(1+np.exp(-x))

def plot_10X10_figure(samples):
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(10, 10)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r') 
    return fig


class RBM():
    def __init__(self, visible_dim = 784, hidden_dim = 100, k=1):
        # W : [hidden_dim, visible_dim]
        # c : [visible_dim]
        # b : [hidden_dim]
        self.W = (2*np.random.rand(hidden_dim, visible_dim)-1)\
                 *math.sqrt(6)/math.sqrt(visible_dim+hidden_dim)
        self.c = np.zeros(visible_dim)
        self.b = np.zeros(hidden_dim)
        self.k = k

    def _get_P_h_given_x(self, x):
        '''
        Augument:
            x:  [batch_size, visible_dim]
        Return:
            h(x) := p(h=1|x)
                [batch_size, hidden_dim]
        '''
        b_plus_Wx = np.expand_dims(self.b, axis=0) + self.W.dot(x.T).T
        return sigm(b_plus_Wx)

    def _get_P_x_given_h(self, h):
        '''
        Augument:
            h:  [batch_size, hidden_dim]
        Return:
            P(x=1|h): [batch_size, visible_dim]
        '''
        c_plus_hW = np.expand_dims(self.c, axis=0) + h.dot(self.W)
        return sigm(c_plus_hW)

    def _get_hx_x(self, x, h_x):
        '''
        Augument:
            x:  [batch_size, visible_dim]
            h_x: [batch_size, hidden_dim]
        Return:
            result := h(x)x^T = p(h=1|x)x^T
                [batch_size, hidden_dim, visible_dim]
        '''
        result = [ np.outer(h_x_sample, x_sample)#[np.expand_dims(h_x_sample, axis=1).dot(np.expand_dims(x_sample, axis=0))\
            for h_x_sample, x_sample in zip(h_x, x)]
        return np.asarray(result)

    def get_Gibbs_sample(self, x, k):
        '''
        Augument:
            x:  [batch_size, visible_dim]
        Return:
            x_tilda: [batch_size, visible_dim]
        '''        
        p_h_x = self._get_P_h_given_x(x)
        h_tilda = np.random.binomial(n=1, p=p_h_x) #get_bernoulli_sample(p_h_x)

        for i in range(k-1):
            # sample x_tilda
            p_x_h = self._get_P_x_given_h(h_tilda)
            x_tilda = np.random.binomial(n=1, p=p_x_h) #get_bernoulli_sample(p_x_h)
            # sample h_tilda
            p_h_x = self._get_P_h_given_x(x_tilda)
            h_tilda = np.random.binomial(n=1, p=p_h_x) #get_bernoulli_sample(p_h_x)

        # sample x_tilda
        p_x_h = self._get_P_x_given_h(h_tilda)
        x_tilda = np.random.binomial(n=1, p=p_x_h) #get_bernoulli_sample(p_x_h)
        return x_tilda, p_x_h

    def _get_grad(self, x, x_tilda):
        '''
        Augument:
            x:  [batch_size, visible_dim]
            x_tilda:  [batch_size, visible_dim]
        Return:
            result: list(grad_W, grad_b, grad_c)
            grad_W := h(x)x^T - h(x_tilda)x_tilda^T [batch_size, hidden_dim, visible_dim]
            grad_b := h(x) - h(x_tilda) [batch_size, hidden_dim]
            grad_c := x - x_tilda [batch_size, visible_dim]
        '''
        h_x = self._get_P_h_given_x(x)
        h_x_tilda = self._get_P_h_given_x(x_tilda)
        grad_W = self._get_hx_x(x, h_x) - self._get_hx_x(x_tilda, h_x_tilda)
        grad_b = h_x - h_x_tilda
        grad_c = x - x_tilda
        return [grad_W, grad_b, grad_c]

    def fit(self, x, learning_rt):
        '''
        Augument:
            x:  [batch_size, visible_dim]
        '''
        x_tilda,_ = self.get_Gibbs_sample(x, self.k)
        grad_W, grad_b, grad_c = self._get_grad(x, x_tilda)
        self.W += learning_rt * np.mean(grad_W, axis=0)
        self.b += learning_rt * np.mean(grad_b, axis=0)
        self.c += learning_rt * np.mean(grad_c, axis=0)

    def visulize_W(self):
        return plot_10X10_figure(self.W) 

    def evaluate(self, x):
        '''
        Augument:
            x:  [batch_size, visible_dim]
        Return:
            cross entropy loss
        '''
        p_h_x = self._get_P_h_given_x(x)
        h_tilda = get_bernoulli_sample(p_h_x)
        p_x_h = self._get_P_x_given_h(h_tilda)
        loss = -x*np.log(p_x_h) - (1-x)*np.log(1-p_x_h)
        loss = np.sum(loss)
        return loss


def train_or_evaluate_one_epoch(model, data, train = True, learning_rt=0.01):

    def train_or_evaluate_one_batch(model, batch_data, train, learning_rt):
        batch_size = len(batch_data)
        batch_data = np.array(batch_data)
        X = batch_data[:,:-1]
        if train:        
            model.fit(X,learning_rt)
        else:
            loss = model.evaluate(X)
            return loss

    batch_data = []
    loss_sum = .0
    for i, index in enumerate(np.random.permutation(len(data))):
        batch_data.append(data[index])
        if len(batch_data) == BATCH_SIZE:
            if train:
                train_or_evaluate_one_batch(model, batch_data, train, learning_rt)
            else:
                loss_sum += train_or_evaluate_one_batch(model, batch_data, train, learning_rt)
            batch_data = []

    if len(batch_data) != 0:
        if train:
            train_or_evaluate_one_batch(model, batch_data, train, learning_rt)
        else:
            loss_sum += train_or_evaluate_one_batch(model, batch_data, train, learning_rt)
    return loss_sum/len(data)

def problem_a_b(k=1):
    model = RBM(k=1)
    train_loss_array=[]
    valid_loss_array=[]
    for i in range(30):
        # evaluate
        train_loss = train_or_evaluate_one_epoch(model, train_data, train = False)
        valid_loss = train_or_evaluate_one_epoch(model, valid_data, train = False)
        train_loss_array.append(train_loss)
        valid_loss_array.append(valid_loss)
        print 'Epoch number:', i, 'Train loss:', train_loss, 'Valid loss:', valid_loss
        # train
        train_or_evaluate_one_epoch(model, train_data, train = True, learning_rt=0.1)

    fig = model.visulize_W()
    fig.savefig('k=%d_W_visulization.png'%(k,))
    # plot cross_entropy_loss
    fig = plt.figure()
    plt.plot(train_loss_array,'g-',label='train_cross_entropy')
    plt.plot(valid_loss_array,'r-',label='valid_cross_entropy')

    plt.xlabel('epoches')
    plt.ylabel('loss')
    plt.title('cross entropy loss, k=%d'%(k,))

    plt.ylim(0,500)

    plt.grid(True)
    plt.legend()
    fig.savefig('cross_entropy_loss_k=%d.png'%(k,))


problem_a_b(k=1)
'''
problem_a_b(k=5)
problem_a_b(k=10)
problem_a_b(k=20)
'''
def problem_c():
    model = RBM(k=10)
    for i in range(20):
        print 'Epoch number:', i
        train_or_evaluate_one_epoch(model, train_data, train = True, learning_rt=0.1)
    for i in range(20):
        print 'Epoch number:', 20+i
        train_or_evaluate_one_epoch(model, train_data, train = True, learning_rt=0.01)
    _, generated_samples = model.get_Gibbs_sample(np.random.rand(100,784), 10000)
    fig = plot_10X10_figure(generated_samples)
    fig.savefig('generated_samples.png')

#problem_c()

'''
Problem d:
'''
def train_or_evaluate_epoch(model, data, Train = True, 
                        learning_rt = 0.1, momentum = .0, NLL = True, alpha = .0):
    # Train = True means the model will be trained, otherwise, only evaluate it
    # NLL = True means it will output the cross_entropy_loss, otherwise, it will output 
    #  incorrect classification ratio.

    def train_or_evaluate_batch(model, sentences, Train = True, 
                        learning_rt = 0.1, momentum =.0, NLL = True, alpha = .0):
        batch_size = len(sentences)
        batch_data = np.array(sentences)
        X = batch_data[:,:-1]
        Labels = np.zeros((batch_size, 10))
        Labels[np.arange(batch_size), batch_data[:, -1].astype(int)]=1

        if NLL:
            loss = model.get_NLL_loss(X, Labels)
        else:
            loss = model.get_IC_loss(X, Labels)

        if Train:
            model.backward(Labels, learning_rt, momentum, alpha)

        return loss


    sentences = []
    sum_loss = 0
    for i, index in enumerate(np.random.permutation(len(data))):
        # Prepare batch dataset
        sentences.append(data[index])
        if len(sentences) == BATCH_SIZE:
            # Train the model
            loss = train_or_evaluate_batch(model, sentences, Train, learning_rt, momentum, NLL, alpha)
            #print loss
            sum_loss += loss
            # Clear old batch
            sentences = []

    if len(sentences) != 0:
        loss = train_or_evaluate_batch(model, sentences, Train, learning_rt, momentum, NLL, alpha)
        sum_loss += loss
    average_loss = sum_loss * 1.0 / len(data) 
    return average_loss

def get_loss_one_time(learning_rt = 0.1, momentum = .0, NLL=True, hidden_dim = 100, alpha = .0, act_fun = Sigmoid, W=None):
    model = NN_3_layer(hidden_dim)
    if W != None:
        model.layer1.W[:-1,:]=W.T
    Train_loss = []
    Valid_loss = []

    for i in range(100):
        '''
        First Evaluate:
        '''
        train_loss = train_or_evaluate_epoch(model, train_data, Train = False, NLL = NLL)
        # print 'train_loss:', train_loss
        valid_loss = train_or_evaluate_epoch(model, valid_data, Train = False, NLL = NLL)
        Train_loss.append(train_loss)
        Valid_loss.append(valid_loss)
        print 'Epoch num:', i, 'train_loss:', train_loss, 'valid_loss:', valid_loss

        '''
        Second Train:
        '''
        train_or_evaluate_epoch(model, train_data, Train = True,
                             learning_rt = learning_rt, momentum = momentum, NLL = NLL, alpha = alpha)
    return Train_loss, Valid_loss, model

def problem_d():
    train_loss_no_pretrain, valid_loss_no_pretrain, _ = get_loss_one_time(NLL=False)
    model = RBM(k=1)
    for i in range(20):
        print 'Epoch number:', i
        train_or_evaluate_one_epoch(model, train_data, train = True, learning_rt=0.1)
    train_loss_pretrain, valid_loss_pretrain, _ = get_loss_one_time(NLL=False, W = model.W)

    # Plot
    fig = plt.figure()
    plt.plot(valid_loss_no_pretrain,'g-',label='valid_IC_ratio_no_pretrain')
    plt.plot(valid_loss_pretrain,'r-',label='valid_IC_ratio_pretrain')

    plt.xlabel('epoches')
    plt.ylabel('loss')
    plt.title('incorrect classification_error')

    plt.grid(True)
    plt.legend()
    fig.savefig('problem_d.png')

#problem_d()


