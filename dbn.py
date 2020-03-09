import numpy as np
from rbm import RBM_dtod, RBM_ctod


class DBN:

    __slots__ = ('layers', 'loss')

    def __init__(self, input_size, config):

        if config[0]['type'] == 'dtod':
            aux = RBM_dtod()
        elif config[0]['type'] == 'ctod':
            aux = RBM_ctod()
        aux.init_rand_prior(input_size, config[0]['hidden'])
        self.layers = [aux]
        for c in range(1, len(config)):
            if config[0]['type'] == 'dtod':
                aux = RBM_dtod()
            elif config[0]['type'] == 'ctod':
                aux = RBM_ctod()
            aux.init_rand_prior(config[c - 1]['hidden'], config[c]['hidden'])
            self.layers.append(aux)

        self.loss = None

    def visible_to_hidden(self, data):
        for l in self.layers:
            data = l.visible_to_hidden(data)
        return data

    def hidden_to_visible(self, data):
        for l in reversed(self.layers):
            data = l.hidden_to_visible(data)
        return data

    def get_param(self, data, batch_size, epsilon, epoch, gibbs_steps):
        for e in range(epoch):
            for l in self.layers:
                l.get_param_CD(data, batch_size, epsilon, 1, gibbs_steps)
                data = l.visible_to_hidden(data)

            hid = self.visible_to_hidden(data)
            data_reconstruct = self.hidden_to_visible(hid)
            t = np.sqrt(np.sum((np.square(data - data_reconstruct))) / np.prod(data.shape))
            self.loss = t
