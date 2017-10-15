import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math

BATCH_SIZE = 32
train_data = np.genfromtxt('data/digitstrain.txt', delimiter=',')
valid_data = np.genfromtxt('data/digitsvalid.txt', delimiter=',')

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
        return 1/(1+np.exp(-b_plus_Wx))

    def _get_P_x_given_h(self, h):
        '''
        Augument:
            h:  [batch_size, hidden_dim]
        Return:
            P(x=1|h): [batch_size, visible_dim]
        '''
        c_plus_hW = np.expand_dims(self.c, axis=0) + h.dot(self.W)
        return 1/(1+np.exp(-c_plus_hW))

    def _get_hx_x(self, x, h_x):
        '''
        Augument:
            x:  [batch_size, visible_dim]
            h_x: [batch_size, hidden_dim]
        Return:
            result := h(x)x^T = p(h=1|x)x^T
                [batch_size, hidden_dim, visible_dim]
        '''
        result = [np.expand_dims(h_x_sample, axis=1).dot(np.expand_dims(x_sample, axis=0))\
            for h_x_sample, x_sample in zip(h_x, x)]
        return np.asarray(result)

    def _Gibbs_sample(self, x):
        '''
        Augument:
            x:  [batch_size, visible_dim]
        Return:
            x_tilda: [batch_size, visible_dim]
        '''        
        def get_bernoulli_sample(p):
            h = np.random.rand(*p.shape)
            return (h<p).astype(float)
        p_h_x = self._get_P_h_given_x(x)
        h_tilda = get_bernoulli_sample(p_h_x)

        for i in range(self.k-1):
            # sample x_tilda
            p_x_h = self._get_P_x_given_h(h_tilda)
            x_tilda = get_bernoulli_sample(p_x_h)
            # sample h_tilda
            p_h_x = self._get_P_h_given_x(x_tilda)
            h_tilda = get_bernoulli_sample(p_h_x)

        # sample x_tilda
        p_x_h = self._get_P_x_given_h(h_tilda)
        x_tilda = get_bernoulli_sample(p_x_h)
        return x_tilda

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
        print np.linalg.norm(grad_c)
        return [grad_W, grad_b, grad_c]

    def fit(self, x, learning_rt):
        '''
        Augument:
            x:  [batch_size, visible_dim]
        '''
        x_tilda = self._Gibbs_sample(x)
        # TODO
        grad_W, grad_b, grad_c = self._get_grad(x, x_tilda)
        self.W += learning_rt * np.mean(grad_W, axis=0)
        self.b += learning_rt * np.mean(grad_b, axis=0)
        self.c += learning_rt * np.mean(grad_c, axis=0)

    def visulize_W(self):
        fig = plt.figure(figsize=(10, 10))
        gs = gridspec.GridSpec(10, 10)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(self.W):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(28, 28), cmap='Greys_r') 
        
        fig.savefig('problem_e.png')

def train_one_epoch(model, data, learning_rt):

    def train_batch(model, batch_data, learning_rt):
        batch_size = len(batch_data)
        batch_data = np.array(batch_data)
        X = batch_data[:,:-1]        
        model.fit(X,learning_rt)

    batch_data = []
    for i, index in enumerate(np.random.permutation(len(data))):
        batch_data.append(data[index])
        if len(batch_data) == BATCH_SIZE:
            train_batch(model, batch_data, learning_rt)
            batch_data = []

    if len(batch_data) != 0:
        train_batch(model, batch_data, learning_rt)


model = RBM()
for _ in range(5):
    train_one_epoch(model, train_data, learning_rt=0.1)
for _ in range(5):
    train_one_epoch(model, train_data, learning_rt=0.01)
model.visulize_W()

