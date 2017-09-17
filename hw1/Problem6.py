import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from nn import NN_3_layer

BATCH_SIZE = 32
train_data = np.genfromtxt('data/digitstrain.txt', delimiter=',')
valid_data = np.genfromtxt('data/digitsvalid.txt', delimiter=',')

def train_or_evaluate_epoch(model, data, Train = True, 
                        learning_rt = 0.1, momentum = .0, NLL = True):
    # Train = True means the model will be trained, otherwise, only evaluate it
    # NLL = True means it will output the cross_entropy_loss, otherwise, it will output 
    #  incorrect classification ratio.

    def train_or_evaluate_batch(model, sentences, Train = True, 
                        learning_rt = 0.1, momentum =.0, NLL = True):
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
            model.backward(Labels, learning_rt, momentum)

        return loss        


    sentences = []
    sum_loss = 0
    for i, index in enumerate(np.random.permutation(len(data))):
        # Prepare batch dataset
        sentences.append(data[index])
        if len(sentences) == BATCH_SIZE:
            # Train the model
            loss = train_or_evaluate_batch(model, sentences, Train, learning_rt, momentum, NLL)
            #print loss
            sum_loss += loss
            # Clear old batch
            sentences = []

    if len(sentences) != 0:
        loss = train_or_evaluate_batch(model, sentences, Train, learning_rt, momentum, NLL)
        sum_loss += loss
    average_loss = sum_loss * 1.0 / len(data) 
    return average_loss

'''
    Problem a:
'''
def get_loss_one_time(learning_rt = 0.1, momentum = .0, NLL=True):
    model = NN_3_layer()
    Train_loss = []
    Valid_loss = []

    for i in range(201):
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
                             learning_rt = learning_rt, momentum = momentum, NLL = NLL)

    return Train_loss, Valid_loss, model

def plot_loss_average(info = 'cross_entropy_loss', ymax = 1, NLL=True):
    Train_loss = np.zeros(201)
    Valid_loss = np.zeros(201)
    for _ in range(5):
        train_loss, valid_loss, _ = get_loss_one_time(NLL=NLL)
        Train_loss += 0.2*np.array(train_loss)
        Valid_loss += 0.2*np.array(valid_loss)

    '''
    Plot the loss
    '''
    plt.figure()
    plt.plot(Train_loss,"g-",label="traindata")
    plt.plot(Valid_loss,"r-.",label="validdata")

    plt.xlabel("epoches")
    plt.ylabel("error")
    plt.title(info)

    plt.ylim(0,ymax)

    plt.grid(True)
    plt.legend()
    return plt

def plot_visulizing_parameter(samples, num):
    fig = plt.figure(figsize=(num, num))
    gs = gridspec.GridSpec(num, num)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

# Problem a
'''
fig = plot_loss_average(info = 'cross_entropy_loss', ymax = 1, NLL=True)
fig.savefig('problem_a.png')
'''
# Problem b
'''
fig = plot_loss_average(info = 'incorrect classification ratio', ymax = 0.5, NLL=False)
fig.savefig('problem_b.png')
'''
# Problem c
'''
_, _, model = get_loss_one_time(NLL=True)
fig = plot_visulizing_parameter(np.transpose(model.layer1.W[:-1,:]), 10)
fig.savefig('problem_c.png')
'''

# Problem 



