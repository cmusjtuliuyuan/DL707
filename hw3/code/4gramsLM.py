import numpy as np
import math
from loader import get_train_dataset,get_val_dataset
import matplotlib.pyplot as plt
import cPickle
import optparse

BATCH_SIZE = 512
LEARNING_RATE = 0.1

optparser = optparse.OptionParser()
optparser.add_option(
    "--type", default='Linear',
    type='str', help="Whether use tanh or not"
)
optparser.add_option(
    "--hidden", default=128,
    type='int', help="Hidden dimension"
)
optparser.add_option(
    "--save", default='a.model',
    type='str', help="where to store the model"
)
optparser.add_option(
    "--load", default='a.model',
    type='str', help="which model to load"
)
optparser.add_option(
    "--epoch", default=100,
    type='int', help="Number of Epoch to train"
)
def rand_init(dim1, dim2):
    return (2*np.random.rand(dim1, dim2)-1)\
                *math.sqrt(3.0/(dim1+dim2))

class LanuageModel():
    def __init__(self, embed_dim=16, vocab_size=8000, hidden_dim=128, Linear=True):
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.Linear = Linear
        # Embedding layer
        self.C = rand_init(self.vocab_size, self.embed_dim)
        # Embedding to hidden
        self.W1 = rand_init(self.embed_dim, self.hidden_dim)
        self.W2 = rand_init(self.embed_dim, self.hidden_dim)
        self.W3 = rand_init(self.embed_dim, self.hidden_dim)
        self.b_hidden = rand_init(1, self.hidden_dim)
        # Hidden to output
        self.W_out = rand_init(self.hidden_dim, self.vocab_size)
        self.b_out = rand_init(1, self.vocab_size)
        #self.S = None


    def forward(self, w1, w2, w3):
        '''
        w1, w2, w3 are list of int of size:(batch_size,)
        '''
        # Embedding layer
        # C1, C2, C3: [batch_size, embed_dim]
        self.C1 = self.C[w1,:]
        self.C2 = self.C[w2,:]
        self.C3 = self.C[w3,:]
        # Embedding to Hidden
        # A:[batch_size, hidden_dim]
        A = self.C1.dot(self.W1)+self.C2.dot(self.W2)+self.C3.dot(self.W3)+self.b_hidden
        if self.Linear:
            self.H = A
        else:
            self.H = np.tanh(A)
        # Hidden to Output
        # O:[batch, voc_size]
        O = self.H.dot(self.W_out)+self.b_out
        # Softmax Layer
        # S:[batch, voc_size]
        O_exp = np.exp(O)
        self.S = np.transpose(O_exp.T/np.sum(O_exp, axis=1))
        return self.S

    def fit_batch(self, w1, w2, w3, w4):
    	batch_size = w1.shape[0]
        self.forward(w1,w2,w3)
        Labels = np.zeros((w4.shape[0], self.vocab_size))
        Labels[np.arange(w4.shape[0]), w4] = 1
        # d_loss_d_O: [batch_size, vocab_size]
        d_loss_d_O = self.S - Labels
        # d_loss_d_H: [batch_size, hidden_dim]
        d_loss_d_H = d_loss_d_O.dot(self.W_out.T)
        d_loss_d_W_out = self.H.T.dot(d_loss_d_O)
        #np.array([np.outer(H,dO) for dO, H in zip(d_loss_d_O, self.H)])
        d_loss_d_b_out = d_loss_d_O
        if self.Linear:
            d_loss_d_A = d_loss_d_H
        else:
            d_loss_d_A = d_loss_d_H*(1 - self.H*self.H)
        d_loss_d_c1 = d_loss_d_A.dot(self.W1.T)
        d_loss_d_c2 = d_loss_d_A.dot(self.W2.T)
        d_loss_d_c3 = d_loss_d_A.dot(self.W3.T)
        d_loss_d_W1 = self.C1.T.dot(d_loss_d_A)#np.array([np.outer(c,h) for c, h in zip(self.C1, d_loss_d_A)])
        d_loss_d_W2 = self.C2.T.dot(d_loss_d_A)#np.array([np.outer(c,h) for c, h in zip(self.C2, d_loss_d_A)])
        d_loss_d_W3 = self.C3.T.dot(d_loss_d_A)#np.array([np.outer(c,h) for c, h in zip(self.C3, d_loss_d_A)])
        d_loss_d_b_hidden = d_loss_d_A

        # updata parameters
        self.C[w1,:] -= LEARNING_RATE*np.mean(d_loss_d_c1, axis=0)
        self.C[w2,:] -= LEARNING_RATE*np.mean(d_loss_d_c2, axis=0)
        self.C[w3,:] -= LEARNING_RATE*np.mean(d_loss_d_c3, axis=0)
        self.W1 -= LEARNING_RATE*d_loss_d_W1/batch_size
        self.W2 -= LEARNING_RATE*d_loss_d_W2/batch_size
        self.W3 -= LEARNING_RATE*d_loss_d_W3/batch_size
        self.b_hidden -= LEARNING_RATE*np.mean(d_loss_d_b_hidden, axis=0, keepdims=True)
        self.W_out -= LEARNING_RATE*d_loss_d_W_out/batch_size
        self.b_out -= LEARNING_RATE*np.mean(d_loss_d_b_out, axis=0, keepdims=True)
        # calculate loss
        loss = np.mean([-np.log(s[i]) for s, i in zip(self.S, w4)])
        return loss

    def get_loss_batch(self, w1, w2, w3, w4):
        S = self.forward(w1,w2,w3)  
        loss = np.mean([-np.log(s[i]) for s, i in zip(S, w4)])
        return loss

    def get_perplexity_batch(self, w1, w2, w3, w4):
        S = self.forward(w1,w2,w3) 
        perplexity = np.mean([np.log2(s[i]) for s, i in zip(S, w4)])
        return perplexity 

    def loop_data(self, X, fn):
        batch_data = []
        sum_out = 0
        for i, index in enumerate(np.random.permutation(len(X))):
            # Prepare batch dataset
            batch_data.append(X[index])
            if len(batch_data) == BATCH_SIZE:
                # Train the model
                batch_data = np.array(batch_data)
                w1, w2, w3, w4 = batch_data[:,0],batch_data[:,1],batch_data[:,2],batch_data[:,3]
                out = fn(w1,w2,w3,w4)
                #print out
                sum_out += out*len(batch_data)
                batch_data = []
        average_out = sum_out/len(X)
        return average_out

    def fit(self, X):
        self.loop_data(X, self.fit_batch)

    def get_loss(self, X):
        loss = self.loop_data(X, self.get_loss_batch)
        return loss

    def get_perplexity(self, X):
        perplexity = self.loop_data(X, self.get_perplexity_batch)
        return 2**(-perplexity)

def train(model, train_data, valid_data, epoch_num):
    train_loss = [model.get_loss(train_data),]
    valid_loss = [model.get_loss(valid_data),]
    perplexity = [model.get_perplexity(valid_data),]

    for i in range(epoch_num):
        print 'EPOCH_NUM:', i
        model.fit(train_data)
        tl = model.get_loss(train_data)
        vl = model.get_loss(valid_data)
        per = model.get_perplexity(valid_data)
        train_loss.append(tl)
        valid_loss.append(vl)
        perplexity.append(per)
        print 'train loss:',tl,'valid loss:', vl, 'perplexity', per
    return train_loss, valid_loss, perplexity

def predict(model, dictionary, str, length):
    #Convert str to index:
    word_to_id = dictionary['word_to_id']
    str_list = str.split()
    w1 = [word_to_id[str_list[0]],]
    w2 = [word_to_id[str_list[1]],]
    w3 = [word_to_id[str_list[2]],]
    #Predict
    sent_list = [w1, w2, w3]
    for _ in range(length):
        S = model.forward(sent_list[-3], sent_list[-2], sent_list[-1])
        new_w = np.argmax(S, axis=1).tolist()
        sent_list.append(new_w)
    #Convert index to word
    id_to_word = dictionary['id_to_word']
    sen_len = len(sent_list)
    num_sent = len(sent_list[0])
    for i in range(num_sent):
        for j in range(sen_len):
            print id_to_word[sent_list[j][i]],
        print

def distance(model, dictionary, word1, word2):
    word_to_id = dictionary['word_to_id']
    index1 = word_to_id['word1']
    index2 = word_to_id['word2']
    e1, e2 = model.C[index1,:], model.C[index2,:]
    distance = np.linalg.norm(e1-e2)
    return distance


def plot_loss_perplexity(train_loss, valid_loss, perplexity,hidden_dim, prefix):
    # Plot Loss
    fig = plt.figure()
    plt.plot(train_loss,'g-',label='Train data')
    plt.plot(valid_loss,'r-',label='Valid data')
    plt.xlabel('epoches')
    plt.ylabel('loss')
    plt.title('Cross Entropy Loss, hidden_dim=%d'%(hidden_dim,))
    plt.grid(True)
    plt.legend()
    fig.savefig('%s:loss_hidden%d.png'%(prefix,hidden_dim))
    # Plot perplexity
    fig = plt.figure()
    plt.plot(perplexity,'g-',label='Valid data')
    plt.xlabel('epoches')
    plt.ylabel('perplexity')
    plt.title('Perplexity of Valid data, hidden_dim=%d'%(hidden_dim,))
    plt.grid(True)
    plt.legend()
    fig.savefig('%s:perplexity_hidden%d.png'%(prefix,hidden_dim))

def main():
    opts = optparser.parse_args()[0]
    
    hidden_dim = opts.hidden
    EPOCH_NUM = opts.epoch
    if opts.type == 'Linear':
        Linear, prefix = True, 'Linear'
    else:
        Linear, prefix = False, 'NoLinear'

    traindataset, dictionary = get_train_dataset('train.txt')
    valdataset = get_val_dataset('val.txt', dictionary)
    model = LanuageModel(hidden_dim=hidden_dim, Linear=Linear)
    train_loss, valid_loss, perplexity = train(model, traindataset, valdataset,opts.epoch)
    #plot_loss_perplexity(train_loss, valid_loss, perplexity, model.hidden_dim, prefix)
    with open(opts.save, 'wb') as f:
        cPickle.dump(model, f)
    '''
    _, dictionary = get_train_dataset('train.txt')
    with open(opts.load, 'rb') as f:
        model = cPickle.load(f)
    predict(model, dictionary,'the company said',10)
    predict(model, dictionary,'new york stock',10)
    predict(model, dictionary,'the company said',10)
    predict(model, dictionary,'president and chief',10)
    #predict(model, dictionary,'life in the',10)
    #predict(model, dictionary,'see you then',10)
    '''

if __name__ == '__main__':
    main()



