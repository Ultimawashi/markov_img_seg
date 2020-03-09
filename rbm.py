import numpy as np
from utils import sigmoid_np


class RBM_dtod:

    __slots__ = ('a', 'b', 'w', 'loss', 'losstype')

    def __init__(self, losstype='cross_entropy', a=None, b=None, w=None):
        assert losstype in ['cross_entropy', 'mean_square_error'], 'please choose a loss function between cross_entropy and mean_square_error'
        self.a = a
        self.b = b
        self.w = w
        self.losstype = losstype
        self.loss = None

    def init_rand_prior(self,visible_size, hidden_size):
        self.a = np.zeros((1, visible_size))
        self.b = np.zeros((1, hidden_size))
        self.w = np.random.normal(0, 0.1, (visible_size, hidden_size))

    def visible_to_hidden(self, data, tir=False):
        hidden = sigmoid_np((data @ self.w) + np.tile(self.b, (data.shape[0], 1)))
        if tir:
            aux = np.random.rand(hidden.shape[0], hidden.shape[1])
            hidden = 1 * (aux <= hidden) + 0 * (aux > hidden)
        return hidden

    def hidden_to_visible(self, data, tir=False):
        visible = sigmoid_np((data @ self.w.transpose()) + np.tile(self.a, (data.shape[0], 1)))
        if tir:
            aux = np.random.rand(visible.shape[0], visible.shape[1])
            visible = 1 * (aux <= visible) + 0 * (aux > visible)
        return visible

    def get_param_CD(self, data, batch_size, epsilon, epoch, gibbs_steps=1):

        v = np.stack([data] + [np.zeros(data.shape)] * gibbs_steps, axis=2)
        h = np.zeros((v.shape[0], self.w.shape[1], v.shape[1]))
        n = v.shape[0]

        for e in range(epoch):
            np.random.shuffle(v[:, :, 0])

            for batch in range(0, n, batch_size):

                for t in range(gibbs_steps):
                    h[batch: min(n, batch + batch_size - 1), :, t] = self.visible_to_hidden(v[batch: min(n,
                                                                                                         batch + batch_size - 1),
                                                                                            :, t], tir=False)
                    aux = np.random.rand(h[batch: min(n, batch + batch_size - 1), :, t].shape[0],
                                         h[batch: min(n, batch + batch_size - 1), :, t].shape[1])

                    hidden = 1 * (aux <= h[batch: min(n, batch + batch_size - 1), :, t]) + 0 * (
                                aux > h[batch: min(n, batch + batch_size - 1), :, t])
                    v[batch: min(n, batch + batch_size - 1), :, t + 1] = self.hidden_to_visible(hidden, tir=True)
                    h[batch: min(n, batch + batch_size - 1), :, t + 1] = self.visible_to_hidden(v[batch: min(n,
                                                                                                             batch + batch_size - 1),
                                                                                                :, t + 1], tir=False)
                pos = h[batch:min(n, batch + batch_size - 1), :, 0].transpose() @ v[
                                                                                  batch:min(n, batch + batch_size - 1),
                                                                                  :, 0]
                neg = h[batch:min(n, batch + batch_size - 1), :, gibbs_steps].transpose() @ v[batch:min(n,
                                                                                                        batch + batch_size - 1),
                                                                                            :, gibbs_steps]
                dw = (pos - neg).T

                da = np.sum(v[batch:min(n, batch + batch_size - 1), :, 0] - v[batch: min(n,
                                                                                         batch + batch_size - 1), :,
                                                                            gibbs_steps], axis=0)
                db = np.sum(h[batch:min(n, batch + batch_size - 1), :, 0] - h[batch: min(n,
                                                                                         batch + batch_size - 1), :,
                                                                            gibbs_steps], axis=0)
                self.w = self.w + (epsilon / batch_size) * dw
                self.a = self.a + (epsilon / batch_size) * da
                self.b = self.b + (epsilon / batch_size) * db

            hid = self.visible_to_hidden(data, tir=True)
            if self.losstype == 'mean_square_error':
                data_reconstruct = self.hidden_to_visible(hid, tir=True)
                self.loss = np.sqrt(np.sum((np.square(data - data_reconstruct))) / np.prod(data.shape))
            elif self.losstype == 'cross_entropy':
                data_reconstruct = self.hidden_to_visible(hid, tir=False)
                self.loss = - np.sum(data*np.log(data_reconstruct))
            print({'epoch':e, 'error':self.loss})

    def get_param_persist_CD(self, data, batch_size, epsilon, epoch, gibbs_steps=1):

        v = np.stack([data] + [np.zeros(data.shape)] * gibbs_steps, axis=2)
        h = np.zeros((v.shape[0], self.w.shape[1], v.shape[1]))
        n = v.shape[0]

        for e in range(epoch):
            np.random.shuffle(v[:, :, 0])

            for batch in range(0, n, batch_size):

                for t in range(gibbs_steps):
                    h[batch: min(n, batch + batch_size - 1), :, t] = self.visible_to_hidden(v[batch: min(n,
                                                                                                         batch + batch_size - 1),
                                                                                            :, t], tir=False)
                    aux = np.random.rand(h[batch: min(n, batch + batch_size - 1), :, t].shape[0],
                                         h[batch: min(n, batch + batch_size - 1), :, t].shape[1])

                    hidden = 1 * (aux <= h[batch: min(n, batch + batch_size - 1), :, t]) + 0 * (
                            aux > h[batch: min(n, batch + batch_size - 1), :, t])
                    v[batch: min(n, batch + batch_size - 1), :, t + 1] = self.hidden_to_visible(hidden, tir=True)
                    h[batch: min(n, batch + batch_size - 1), :, t + 1] = self.visible_to_hidden(v[batch: min(n,
                                                                                                             batch + batch_size - 1),
                                                                                                :, t + 1], tir=False)
                pos = h[batch:min(n, batch + batch_size - 1), :, 0].transpose() @ v[
                                                                                  batch:min(n, batch + batch_size - 1),
                                                                                  :, 0]
                neg = h[batch:min(n, batch + batch_size - 1), :, gibbs_steps].transpose() @ v[batch:min(n,
                                                                                                        batch + batch_size - 1),
                                                                                            :, gibbs_steps]
                dw = (pos - neg).T

                da = np.sum(v[batch:min(n, batch + batch_size - 1), :, 0] - v[batch: min(n,
                                                                                         batch + batch_size - 1), :,
                                                                            gibbs_steps], axis=0)
                db = np.sum(h[batch:min(n, batch + batch_size - 1), :, 0] - h[batch: min(n,
                                                                                         batch + batch_size - 1), :,
                                                                            gibbs_steps], axis=0)
                self.w = self.w + (epsilon / batch_size) * dw
                self.a = self.a + (epsilon / batch_size) * da
                self.b = self.b + (epsilon / batch_size) * db
                h[batch: min(n, batch + batch_size - 1), :, 0] = h[batch: min(n, batch + batch_size - 1), :, gibbs_steps]
                v[batch: min(n, batch + batch_size - 1),:, 0] = v[batch: min(n, batch + batch_size - 1),:, gibbs_steps]

            hid = self.visible_to_hidden(data, tir=True)
            if self.losstype == 'mean_square_error':
                data_reconstruct = self.hidden_to_visible(hid, tir=True)
                self.loss = np.sqrt(np.sum((np.square(data - data_reconstruct))) / np.prod(data.shape))
            elif self.losstype == 'cross_entropy':
                data_reconstruct = self.hidden_to_visible(hid, tir=False)
                self.loss = - np.sum(data * np.log(data_reconstruct))
            print({'epoch': e, 'error': self.loss})

    def gen_visible(self, gibbs_steps, nb_im):
        v = np.zeros((nb_im, self.w.shape[0]))
        for t in range(gibbs_steps):
            h = self.visible_to_hidden(v, tir=True)
            v = self.hidden_to_visible(h, tir=True)
        return v

    def gen_hidden(self, gibbs_steps, nb_im):
        v = np.zeros((nb_im, self.w.shape[0]))
        for t in range(gibbs_steps):
            h = self.visible_to_hidden(v, tir=True)
            v = self.hidden_to_visible(h, tir=True)
        return h

    def gen_data(self, gibbs_steps, nb_im):
        v = np.zeros((nb_im, self.w.shape[0]))
        h = np.zeros((nb_im, self.w.shape[1]))
        for t in range(gibbs_steps):
            h = self.visible_to_hidden(v, tir=True)
            v = self.hidden_to_visible(h, tir=True)
        return v,h

class RBM_ctod:

    __slots__ = ('a', 'b', 'w', 'loss', 'losstype')

    def __init__(self, losstype='mean_square_error', a=None, b=None, w=None):
        assert losstype in ['mean_square_error'], 'please choose the loss function as the mean_square_error (default value)'
        self.a = a
        self.b = b
        self.w = w
        self.losstype = losstype
        self.loss = None

    def init_rand_prior(self,visible_size, hidden_size):
        self.a = np.zeros((1, visible_size))
        self.b = np.zeros((1, hidden_size))
        self.w = np.random.normal(0, 0.1, (visible_size, hidden_size))

    def visible_to_hidden(self, data, tir=False):
        hidden = sigmoid_np((data @ self.w) + np.tile(self.b, (data.shape[0], 1)))
        if tir:
            aux = np.random.rand(hidden.shape[0], hidden.shape[1])
            hidden = 1 * (aux <= hidden) + 0 * (aux > hidden)
        return hidden

    def hidden_to_visible(self, data, tir=False):
        visible = np.random.normal((data @ self.w.T) + np.tile(self.a, (data.shape[0], 1)))
        return visible

    def get_param_CD(self, data, batch_size, epsilon, epoch, gibbs_steps=1):

        v = np.stack([data] + [np.zeros(data.shape)] * gibbs_steps, axis=2)
        h = np.zeros((v.shape[0], self.w.shape[1], v.shape[1]))
        n = v.shape[0]

        for e in range(epoch):
            np.random.shuffle(v[:, :, 0])

            for batch in range(0, n, batch_size):

                for t in range(gibbs_steps):
                    h[batch: min(n, batch + batch_size - 1), :, t] = self.visible_to_hidden(v[batch: min(n,
                                                                                                         batch + batch_size - 1),
                                                                                            :, t], tir=False)
                    aux = np.random.rand(h[batch: min(n, batch + batch_size - 1), :, t].shape[0],
                                         h[batch: min(n, batch + batch_size - 1), :, t].shape[1])

                    hidden = 1 * (aux <= h[batch: min(n, batch + batch_size - 1), :, t]) + 0 * (
                                aux > h[batch: min(n, batch + batch_size - 1), :, t])
                    v[batch: min(n, batch + batch_size - 1), :, t + 1] = self.hidden_to_visible(hidden, tir=True)
                    h[batch: min(n, batch + batch_size - 1), :, t + 1] = self.visible_to_hidden(v[batch: min(n,
                                                                                                             batch + batch_size - 1),
                                                                                                :, t + 1], tir=False)
                pos = h[batch:min(n, batch + batch_size - 1), :, 0].transpose() @ v[
                                                                                  batch:min(n, batch + batch_size - 1),
                                                                                  :, 0]
                neg = h[batch:min(n, batch + batch_size - 1), :, gibbs_steps].transpose() @ v[batch:min(n,
                                                                                                        batch + batch_size - 1),
                                                                                            :, gibbs_steps]
                dw = (pos - neg).T

                da = np.sum(v[batch:min(n, batch + batch_size - 1), :, 0] - v[batch: min(n,
                                                                                         batch + batch_size - 1), :,
                                                                            gibbs_steps], axis=0)
                db = np.sum(h[batch:min(n, batch + batch_size - 1), :, 0] - h[batch: min(n,
                                                                                         batch + batch_size - 1), :,
                                                                            gibbs_steps], axis=0)
                self.w = self.w + (epsilon / batch_size) * dw
                self.a = self.a + (epsilon / batch_size) * da
                self.b = self.b + (epsilon / batch_size) * db

            hid = self.visible_to_hidden(data, tir=True)
            if self.losstype == 'mean_square_error':
                data_reconstruct = self.hidden_to_visible(hid, tir=True)
                self.loss = np.sqrt(np.sum((np.square(data - data_reconstruct))) / np.prod(data.shape))
            print({'epoch':e, 'error':self.loss})

    def get_param_persist_CD(self, data, batch_size, epsilon, epoch, gibbs_steps=1):

        v = np.stack([data] + [np.zeros(data.shape)] * gibbs_steps, axis=2)
        h = np.zeros((v.shape[0], self.w.shape[1], v.shape[1]))
        n = v.shape[0]

        for e in range(epoch):
            np.random.shuffle(v[:, :, 0])

            for batch in range(0, n, batch_size):

                for t in range(gibbs_steps):
                    h[batch: min(n, batch + batch_size - 1), :, t] = self.visible_to_hidden(v[batch: min(n,
                                                                                                         batch + batch_size - 1),
                                                                                            :, t], tir=False)
                    aux = np.random.rand(h[batch: min(n, batch + batch_size - 1), :, t].shape[0],
                                         h[batch: min(n, batch + batch_size - 1), :, t].shape[1])

                    hidden = 1 * (aux <= h[batch: min(n, batch + batch_size - 1), :, t]) + 0 * (
                            aux > h[batch: min(n, batch + batch_size - 1), :, t])
                    v[batch: min(n, batch + batch_size - 1), :, t + 1] = self.hidden_to_visible(hidden, tir=True)
                    h[batch: min(n, batch + batch_size - 1), :, t + 1] = self.visible_to_hidden(v[batch: min(n,
                                                                                                             batch + batch_size - 1),
                                                                                                :, t + 1], tir=False)
                pos = h[batch:min(n, batch + batch_size - 1), :, 0].transpose() @ v[
                                                                                  batch:min(n, batch + batch_size - 1),
                                                                                  :, 0]
                neg = h[batch:min(n, batch + batch_size - 1), :, gibbs_steps].transpose() @ v[batch:min(n,
                                                                                                        batch + batch_size - 1),
                                                                                            :, gibbs_steps]
                dw = (pos - neg).T

                da = np.sum(v[batch:min(n, batch + batch_size - 1), :, 0] - v[batch: min(n,
                                                                                         batch + batch_size - 1), :,
                                                                            gibbs_steps], axis=0)
                db = np.sum(h[batch:min(n, batch + batch_size - 1), :, 0] - h[batch: min(n,
                                                                                         batch + batch_size - 1), :,
                                                                            gibbs_steps], axis=0)
                self.w = self.w + (epsilon / batch_size) * dw
                self.a = self.a + (epsilon / batch_size) * da
                self.b = self.b + (epsilon / batch_size) * db
                h[batch: min(n, batch + batch_size - 1), :, 0] = h[batch: min(n, batch + batch_size - 1), :, gibbs_steps]
                v[batch: min(n, batch + batch_size - 1),:, 0] = v[batch: min(n, batch + batch_size - 1),:, gibbs_steps]

            hid = self.visible_to_hidden(data, tir=True)
            if self.losstype == 'mean_square_error':
                data_reconstruct = self.hidden_to_visible(hid, tir=True)
                self.loss = np.sqrt(np.sum((np.square(data - data_reconstruct))) / np.prod(data.shape))
            elif self.losstype == 'cross_entropy':
                data_reconstruct = self.hidden_to_visible(hid, tir=False)
                self.loss = - np.sum(data * np.log(data_reconstruct))
            print({'epoch': e, 'error': self.loss})

    def gen_visible(self, gibbs_steps, nb_im):
        v = np.zeros((nb_im, self.w.shape[0]))
        for t in range(gibbs_steps):
            h = self.visible_to_hidden(v, tir=True)
            v = self.hidden_to_visible(h, tir=True)
        return v

    def gen_hidden(self, gibbs_steps, nb_im):
        v = np.zeros((nb_im, self.w.shape[0]))
        for t in range(gibbs_steps):
            h = self.visible_to_hidden(v, tir=True)
            v = self.hidden_to_visible(h, tir=True)
        return h

    def gen_data(self, gibbs_steps, nb_im):
        v = np.zeros((nb_im, self.w.shape[0]))
        h = np.zeros((nb_im, self.w.shape[1]))
        for t in range(gibbs_steps):
            h = self.visible_to_hidden(v, tir=True)
            v = self.hidden_to_visible(h, tir=True)
        return v,h



