import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from nn import NN_3_layer, NN_4_layer, NN_4_BN_layer
from nn import Sigmoid, ReLU, Tanh

BATCH_SIZE = 32
train_data = np.genfromtxt('data/digitstrain.txt', delimiter=',')
valid_data = np.genfromtxt('data/digitsvalid.txt', delimiter=',')
test_data = np.genfromtxt('data/digitstest.txt', delimiter=',')


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

'''
    Problem a and b:
'''
def get_loss_one_time(learning_rt = 0.1, momentum = .0, NLL=True, hidden_dim = 100, alpha = .0, act_fun = Sigmoid):
    model = NN_3_layer(hidden_dim)
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
                             learning_rt = learning_rt, momentum = momentum, NLL = NLL, alpha = alpha)

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
    fig = plt.figure()
    plt.plot(Train_loss,"g-",label="traindata")
    plt.plot(Valid_loss,"r-.",label="validdata")

    plt.xlabel("epoches")
    plt.ylabel("error")
    plt.title(info)

    plt.ylim(0,ymax)

    plt.grid(True)
    plt.legend()
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

'''
    Problem c
'''
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
# Problem c
'''
_, _, model = get_loss_one_time(NLL=True)
fig = plot_visulizing_parameter(np.transpose(model.layer1.W[:-1,:]), 10)
fig.savefig('problem_c.png')
'''

'''
    Problem d
'''
def plot_problem_d_learning_rt(info, ymax, NLL):

    _, Valid_loss_001 , _ = get_loss_one_time(learning_rt = 0.01, momentum = .0, NLL=NLL)
    _, Valid_loss_01 , _ = get_loss_one_time(learning_rt = 0.1, momentum = .0, NLL=NLL)
    _, Valid_loss_02 , _ = get_loss_one_time(learning_rt = 0.2, momentum = .0, NLL=NLL)
    _, Valid_loss_05 , _ = get_loss_one_time(learning_rt = 0.5, momentum = .0, NLL=NLL)

    fig = plt.figure()
    plt.plot(Valid_loss_001,"g-",label="learning_rt = 0.01")
    plt.plot(Valid_loss_01,"r-",label="learning_rt = 0.1")
    plt.plot(Valid_loss_02,"m-",label="learning_rt = 0.2")
    plt.plot(Valid_loss_05,"k-",label="learning_rt = 0.5")

    plt.xlabel("epoches")
    plt.ylabel("error")
    plt.title(info)

    plt.ylim(0,ymax)

    plt.grid(True)
    plt.legend()
    return fig

def plot_problem_d_momentum(info, ymax, NLL):

    _, Valid_loss_0 , _ = get_loss_one_time(learning_rt = 0.1, momentum = .0, NLL=NLL)
    _, Valid_loss_5 , _ = get_loss_one_time(learning_rt = 0.1, momentum = .5, NLL=NLL)
    _, Valid_loss_9 , _ = get_loss_one_time(learning_rt = 0.1, momentum = .9, NLL=NLL)

    fig = plt.figure()
    plt.plot(Valid_loss_0,"g-",label="momentum = 0.0")
    plt.plot(Valid_loss_5,"r-",label="momentum = 0.5")
    plt.plot(Valid_loss_9,"m-",label="momentum = 0.9")

    plt.xlabel("epoches")
    plt.ylabel("error")
    plt.title(info)

    plt.ylim(0,ymax)

    plt.grid(True)
    plt.legend()
    return fig

# Problem d
'''
fig = plot_problem_d_learning_rt('cross_entropy_loss', 2, NLL=True)
fig.savefig('problem_d_learning_rt_cross_entropy.png')

fig = plot_problem_d_learning_rt('incorrect classification ratio', 1, NLL=False)
fig.savefig('problem_d_learning_rt_IC.png')


fig = plot_problem_d_momentum('cross_entropy_loss', 3, NLL=True)
fig.savefig('problem_d_momentum_cross_entropy.png')
fig = plot_problem_d_momentum('incorrect classification ratio', 1, NLL=False)
fig.savefig('problem_d_momentum_IC.png')
'''

'''
    Problem e
'''
def plot_problem_e():
    Train_loss_20, Valid_loss_20 , _ = get_loss_one_time(learning_rt = 0.01, momentum = .5, hidden_dim = 20)
    Train_loss_100, Valid_loss_100 , _ = get_loss_one_time(learning_rt = 0.01, momentum = .5, hidden_dim = 100)
    Train_loss_200, Valid_loss_200 , _ = get_loss_one_time(learning_rt = 0.01, momentum = .5, hidden_dim = 200)
    Train_loss_500, Valid_loss_500 , _ = get_loss_one_time(learning_rt = 0.01, momentum = .5, hidden_dim = 500)

    fig = plt.figure()
    p1 = plt.subplot(211)
    p2 = plt.subplot(212)

    p1.plot(Train_loss_20,"g-",label="hidden_dim = 20")
    p1.plot(Train_loss_100,"r-",label="hidden_dim = 100")
    p1.plot(Train_loss_200,"m-",label="hidden_dim = 200")
    p1.plot(Train_loss_500,"k-",label="hidden_dim = 500")

    p1.set_xlabel("epoches")
    p1.set_ylabel("error")
    p1.set_title("Train: cross_entropy_loss")

    p1.set_ylim(0,0.8)

    p1.grid(True)
    p1.legend()

    p2.plot(Valid_loss_20,"g-",label="hidden_dim = 20")
    p2.plot(Valid_loss_100,"r-",label="hidden_dim = 100")
    p2.plot(Valid_loss_200,"m-",label="hidden_dim = 200")
    p2.plot(Valid_loss_500,"k-",label="hidden_dim = 500")

    p2.set_xlabel("epoches")
    p2.set_ylabel("error")
    p2.set_title("Valid: cross_entropy_loss")

    p2.set_ylim(0.2,0.8)

    p2.grid(True)
    p2.legend()
    return fig

# Problem e
'''
fig = plot_problem_e()
fig.savefig('problem_e.png')
'''

# Problem f
def plot_problem_f_l2_reg():

    _, Valid_loss_0 , _ = get_loss_one_time(learning_rt = 0.01, alpha = 0.)
    _, Valid_loss_5 , _ = get_loss_one_time(learning_rt = 0.01, alpha = 0.0001)
    _, Valid_loss_10 , _ = get_loss_one_time(learning_rt = 0.01, alpha = 0.0005)
    _, Valid_loss_15 , _ = get_loss_one_time(learning_rt = 0.01, alpha = 0.001)

    fig = plt.figure()
    plt.plot(Valid_loss_0,"g-",label="weight decay = 0")
    plt.plot(Valid_loss_5,"r-",label="weight decay = 0.0001")
    plt.plot(Valid_loss_10,"m-",label="weight decay = 0.0005")
    plt.plot(Valid_loss_15,"k-",label="weight decay = 0.001")

    plt.xlabel("epoches")
    plt.ylabel("error")
    plt.title('cross_entropy_loss')

    plt.ylim(0,1)

    plt.grid(True)
    plt.legend()
    return fig

def plot_problem_f_or_g_result(model, epoch_num, learning_rt, momentum, alpha):

    for i in range(epoch_num):
        train_or_evaluate_epoch(model, train_data, True,
                             learning_rt, momentum, True, alpha)
    
    fig = plot_visulizing_parameter(np.transpose(model.layer1.W[:-1,:]), 10)
    
    train_loss_NLL = train_or_evaluate_epoch(model, train_data, Train = False, NLL = True)
    valid_loss_NLL = train_or_evaluate_epoch(model, valid_data, Train = False, NLL = True)
    test_loss_NLL= train_or_evaluate_epoch(model, test_data, Train = False, NLL = True)
    train_loss_IC = train_or_evaluate_epoch(model, train_data, Train = False, NLL = False)
    valid_loss_IC = train_or_evaluate_epoch(model, valid_data, Train = False, NLL = False)
    test_loss_IC= train_or_evaluate_epoch(model, test_data, Train = False, NLL = False)
    print 'train_loss_NLL', train_loss_NLL, 'valid_loss_NLL', valid_loss_NLL, 'test_loss_NLL', test_loss_NLL
    print 'train_loss_IC', train_loss_IC, 'valid_loss_IC', valid_loss_IC, 'test_loss_IC', test_loss_IC

    return fig

'''
# Problem f
fig = plot_problem_f_l2_reg()
fig.savefig('problem_f_find_l2.png')
#train_loss_NLL 0.0302798907306 valid_loss_NLL 0.274775637271 test_loss_NLL 0.325125106534
#train_loss_IC 0.0 valid_loss_IC 0.084 test_loss_IC 0.0923333333333
model = NN_3_layer()
fig = plot_problem_f_or_g_result(model, 180, 0.01, 0, 0.001)
fig.savefig('problem_f_visualization.png')
'''

'''
    Problem g
'''
'''
model = NN_4_layer()
#train_loss_NLL 0.324571271092 valid_loss_NLL 0.499887242168 test_loss_NLL 0.552144295303
#train_loss_IC 0.101 valid_loss_IC 0.149 test_loss_IC 0.166666666667
fig = plot_problem_f_or_g_result(model, 140, 0.01, 0, .0)
fig.savefig('problem_g_visualization.png')
'''

'''
    Problem h
'''
'''
model = NN_4_BN_layer()
#Epoch num: 7
#train_loss_NLL 0.115590739458 valid_loss_NLL 0.359999605125 test_loss_NLL 0.528242543282
#train_loss_IC 0.036 valid_loss_IC 0.118 test_loss_IC 0.138
for i in range(30):

    print 'Epoch num:', i
    # Train the model
    train_or_evaluate_epoch(model, train_data, Train = True,
                         learning_rt = 0.1, momentum = 0, NLL = True, alpha = 0)

    train_loss_NLL = train_or_evaluate_epoch(model, train_data, Train = False, NLL = True)
    valid_loss_NLL = train_or_evaluate_epoch(model, valid_data, Train = False, NLL = True)
    test_loss_NLL= train_or_evaluate_epoch(model, test_data, Train = False, NLL = True)
    train_loss_IC = train_or_evaluate_epoch(model, train_data, Train = False, NLL = False)
    valid_loss_IC = train_or_evaluate_epoch(model, valid_data, Train = False, NLL = False)
    test_loss_IC= train_or_evaluate_epoch(model, test_data, Train = False, NLL = False)
    print 'train_loss_NLL', train_loss_NLL, 'valid_loss_NLL', valid_loss_NLL, 'test_loss_NLL', test_loss_NLL
    print 'train_loss_IC', train_loss_IC, 'valid_loss_IC', valid_loss_IC, 'test_loss_IC', test_loss_IC
'''


'''
    Problem i
'''
def plot_problem_i():
    Train_loss_Sigmoid, Valid_loss_Sigmoid , _ = get_loss_one_time(learning_rt = 0.01, momentum = .5, hidden_dim = 100, act_fun = Sigmoid)
    Train_loss_ReLU, Valid_loss_ReLU , _ = get_loss_one_time(learning_rt = 0.01, momentum = .5, hidden_dim = 100, act_fun = ReLU)
    Train_loss_Tanh, Valid_loss_Tanh , _ = get_loss_one_time(learning_rt = 0.01, momentum = .5, hidden_dim = 100, act_fun = Tanh)

    fig = plt.figure()
    p1 = plt.subplot(211)
    p2 = plt.subplot(212)

    p1.plot(Train_loss_Sigmoid,"g-",label="Sigmoid")
    p1.plot(Train_loss_ReLU,"r-",label="ReLU")
    p1.plot(Train_loss_Tanh,"m-",label="Tanh")

    p1.set_xlabel("epoches")
    p1.set_ylabel("error")
    p1.set_title("Train: cross_entropy_loss")

    p1.set_ylim(0,0.8)

    p1.grid(True)
    p1.legend()

    p2.plot(Valid_loss_Sigmoid,"g-",label="Sigmoid")
    p2.plot(Valid_loss_ReLU,"r-",label="ReLU")
    p2.plot(Valid_loss_Tanh,"m-",label="Tanh")

    p2.set_xlabel("epoches")
    p2.set_ylabel("error")
    p2.set_title("Valid: cross_entropy_loss")

    p2.set_ylim(0.2,0.8)

    p2.grid(True)
    p2.legend()
    return fig
'''
# Problem i
fig = plot_problem_i()
fig.savefig('problem_i.png')
'''
