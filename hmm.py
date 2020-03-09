import numpy as np
import itertools
from utils import generate_semipos_sym_mat, np_multivariate_normal_pdf, calc_product, convertcls_vect, calc_matDS, \
    multinomial_rvs, convert_multcls_vectors, test_calc_cond_DS, calc_transDS, calc_vectDS, calc_cacheDS
from scipy.linalg import block_diag
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans


class HMC_ctod:
    __slots__ = ('p', 't', 'mu', 'sigma', 'nbc_x')

    def __init__(self, nbc_x, p=None, t=None, mu=None, sigma=None):
        self.p = p
        self.t = t
        self.mu = mu
        self.sigma = sigma
        self.nbc_x = nbc_x

    def init_data_prior(self, data, scale=1):
        nb_class = self.nbc_x
        self.p = np.array([1 / nb_class] * nb_class)
        a = np.full((nb_class, nb_class), 1 / (2 * (nb_class - 1)))
        a = a - np.diag(np.diag(a))
        self.t = np.diag(np.array([1 / 2] * nb_class)) + a
        M = np.mean(data, axis=0)
        Sig = np.cov(data, rowvar=False).reshape(data.shape[1], data.shape[1])
        self.mu = [None] * nb_class
        self.sigma = [None] * nb_class
        for l in range(nb_class):
            if l % 2 == 0:
                self.mu[l] = M - scale * ((l / (nb_class / 2)) * np.sum(Sig, axis=1))
            else:
                self.mu[l] = M + scale * ((l / (nb_class / 2)) * np.sum(Sig, axis=1))
            self.sigma[l] = Sig

        self.mu = np.array(self.mu)
        self.sigma = np.array(self.sigma)

    def init_kmeans(self, data):
        self.t = np.zeros((self.nbc_x, self.nbc_x))
        self.p = np.zeros((self.nbc_x,))
        self.mu = np.zeros((self.nbc_x, 1 * len(data[0])))
        self.sigma = np.zeros((self.nbc_x, 1 * len(data[0]), 1 * len(data[0])))

        kmeans = KMeans(n_clusters=self.nbc_x).fit(data)
        hidden = kmeans.labels_
        hiddenc = np.stack((hidden[:-1], hidden[1:]), axis=-1)
        aux = np.moveaxis(np.indices(self.t.shape), 0, -1)
        broadc = (len(aux.shape) - len(hiddenc.shape) + 1)

        c = (1 / (len(data) - 1)) * (
            np.all(hiddenc.reshape((hiddenc.shape[0],) + (1,) * broadc + hiddenc.shape[1:]) == aux, axis=-1).sum(
                axis=0)).reshape(self.t.shape)

        self.p = (1 / (len(data))) * (hidden[..., np.newaxis] == np.indices((self.p.shape[0],))).sum(axis=0)
        self.t = (c.T / self.p).T

        broadc = (len(self.mu.shape) - len(data.shape) + 1)
        self.mu = (((hidden[..., np.newaxis] == np.indices((self.mu.shape[0],))) * data).sum(axis=0) / (
                hidden[..., np.newaxis] == np.indices((self.mu.shape[0],))).sum(axis=0)).reshape(self.mu.shape)
        self.sigma = (((hidden[..., np.newaxis] == np.indices((self.mu.shape[0],))).reshape(
            ((hidden.shape[0],) + self.sigma.shape)) * np.einsum('...i,...j',
                                                                 (data.reshape((data.shape[0],) + (
                                                                     1,) * broadc + data.shape[1:]) - self.mu),
                                                                 (data.reshape(
                                                                     (data.shape[0],) + (
                                                                         1,) * broadc + data.shape[
                                                                                        1:]) - self.mu))).sum(
            axis=0)
                      / ((hidden[..., np.newaxis] == np.indices((self.mu.shape[0],))).sum(
                    axis=0)).reshape(self.sigma.shape))

    def give_param(self, c, mu, sigma):
        self.p = np.sum(c, axis=1)
        self.t = (c.T / self.p).T
        self.t[np.isnan(self.t)] = 0
        self.mu = mu
        self.sigma = sigma

    def seg_map(self, data):
        pass

    def seg_mpm(self, data):
        gaussians = np_multivariate_normal_pdf(data, self.mu, self.sigma)
        forward, backward = self.get_forward_backward(gaussians)
        p_apost = forward * backward
        p_apost = p_apost / (p_apost.sum(axis=1)[..., np.newaxis])
        return np.argmax(p_apost, axis=1)

    def simul_hidden_apost(self, backward, gaussians):
        res = np.zeros(len(backward), dtype=int)
        T = self.t
        aux = (gaussians[0] * self.p) * backward[0]
        test = np.random.multinomial(1, aux / np.sum(aux))
        res[0] = np.argmax(test)
        tapost = (
            (gaussians[1:, np.newaxis, :]
             * backward[1:, np.newaxis, :]
             * T[np.newaxis, :, :])
        )
        tapost = tapost / tapost.sum(axis=2)[..., np.newaxis]
        for i in range(1, len(res)):
            test = np.random.multinomial(1, tapost[i - 1, res[i - 1], :])
            res[i] = np.argmax(test)
        return res

    def simul_visible(self, hidden):
        res = np.zeros((len(hidden), self.mu.shape[-1]))
        for i in range(0, len(res)):
            res[i] = multivariate_normal.rvs(self.mu[hidden[i]], self.sigma[hidden[i]])
        return res

    def generate_sample(self, length):
        hidden = np.zeros(length, dtype=int)
        visible = np.zeros((length, self.mu.shape[-1]))
        T = self.t
        test = np.random.multinomial(1, self.p)
        hidden[0] = np.argmax(test)
        visible[0] = multivariate_normal.rvs(self.mu[hidden[0]], self.sigma[hidden[0]])
        for i in range(1, length):
            test = np.random.multinomial(1, T[hidden[i - 1], :])
            hidden[i] = np.argmax(test)
            visible[i] = multivariate_normal.rvs(self.mu[hidden[i]], self.sigma[hidden[i]])

        return hidden, visible

    def get_forward_backward(self, gaussians):
        forward = np.zeros((len(gaussians), self.t.shape[0]))
        backward = np.zeros((len(gaussians), self.t.shape[0]))
        backward[len(gaussians) - 1] = np.ones(self.t.shape[0])
        forward[0] = self.p * gaussians[0]
        forward[0] = forward[0] / (forward[0].sum())
        T = self.t
        for l in range(1, len(gaussians)):
            k = len(gaussians) - 1 - l
            forward[l] = gaussians[l] * (forward[l - 1] @ T)
            forward[l] = forward[l] / forward[l].sum()
            backward[k] = T @ (backward[k + 1] * gaussians[k + 1])
            backward[k] = backward[k] / (backward[k].sum())

        return forward, backward

    def get_param_EM(self, data, iter, early_stopping=0):
        print({'iter': 0, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma})
        for q in range(iter):
            prev_p = self.p
            prev_t = self.t
            prev_mu = self.mu
            prev_sigma = self.sigma

            gaussians = np_multivariate_normal_pdf(data, self.mu, self.sigma)
            forward, backward = self.get_forward_backward(gaussians)
            T = self.t
            gamma = (
                    forward[:-1, :, np.newaxis]
                    * (gaussians[1:, np.newaxis, :]
                       * backward[1:, np.newaxis, :]
                       * T[np.newaxis, :, :])
            )
            gamma = gamma / (gamma.sum(axis=(1, 2))[..., np.newaxis, np.newaxis])
            psi = forward * backward
            psi = psi / (psi.sum(axis=1)[..., np.newaxis])
            self.t = np.transpose(np.transpose((gamma.sum(axis=0))) / (psi[:-1:].sum(axis=0)))
            self.p = (psi.sum(axis=0)) / psi.shape[0]
            self.mu = (((psi * data).sum(axis=0)) / (psi.sum(axis=0))).reshape(self.mu.shape)
            broadc = (len(self.mu.shape) - len(data.shape) + 1)
            self.sigma = (psi.reshape(((psi.shape[0],) + self.sigma.shape)) * np.einsum('...i,...j',
                                                                                        (data.reshape(
                                                                                            (data.shape[0],) + (
                                                                                                1,) * broadc + data.shape[
                                                                                                               1:]) - self.mu),
                                                                                        (data.reshape(
                                                                                            (data.shape[0],) + (
                                                                                                1,) * broadc + data.shape[
                                                                                                               1:]) - self.mu))).sum(
                axis=0) / ((psi.sum(axis=0)).reshape(self.sigma.shape))

            nb_param = np.prod(self.p.shape) + np.prod(self.t.shape) + np.prod(self.mu.shape) + np.prod(
                self.sigma.shape)
            norm_param = (1 / nb_param) * np.sqrt(
                ((self.p - prev_p) ** 2).sum() + ((self.t - prev_t) ** 2).sum() + ((self.mu - prev_mu) ** 2).sum() + (
                        (self.sigma - prev_sigma) ** 2).sum())
            print({'iter': q + 1, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma,
                   'diff_norm_param': norm_param})
            if norm_param < early_stopping:
                break

    def get_param_ICE(self, data, iter, Nb_simul, early_stopping=0):
        print({'iter': 0, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma})
        for q in range(iter):
            prev_p = self.p
            prev_t = self.t
            prev_mu = self.mu
            prev_sigma = self.sigma

            gaussians = np_multivariate_normal_pdf(data, self.mu, self.sigma)
            forward, backward = self.get_forward_backward(gaussians)
            hidden = np.stack([self.simul_hidden_apost(backward, gaussians) for n in range(Nb_simul)], axis=0)
            T = self.t
            gamma = (
                    forward[:-1, :, np.newaxis]
                    * (gaussians[1:, np.newaxis, :]
                       * backward[1:, np.newaxis, :]
                       * T[np.newaxis, :, :])
            )
            gamma = gamma / (gamma.sum(axis=(1, 2))[..., np.newaxis, np.newaxis])
            psi = forward * backward
            psi = psi / (psi.sum(axis=1)[..., np.newaxis])
            self.t = np.transpose(np.transpose((gamma.sum(axis=0))) / (psi[:-1:].sum(axis=0)))
            self.p = (psi.sum(axis=0)) / psi.shape[0]

            broadc = (len(self.mu.shape) - len(data.shape) + 1)
            self.mu = (((hidden[..., np.newaxis] == np.indices((self.mu.shape[0],))) * data).sum(axis=(0, 1)) /
                       (hidden[..., np.newaxis] == np.indices((self.mu.shape[0],))).sum(axis=(0, 1))).reshape(
                self.mu.shape)
            self.sigma = (
                    ((hidden[..., np.newaxis] == np.indices((self.mu.shape[0],))).reshape(
                        ((hidden.shape[0], hidden.shape[1]) + self.sigma.shape)) * np.einsum('...i,...j',
                                                                                             (data.reshape(
                                                                                                 (data.shape[0],) + (
                                                                                                     1,) * broadc + data.shape[
                                                                                                                    1:]) - self.mu),
                                                                                             (data.reshape(
                                                                                                 (data.shape[0],) + (
                                                                                                     1,) * broadc + data.shape[
                                                                                                                    1:]) - self.mu))).sum(
                        axis=(0, 1)) /
                    ((hidden[..., np.newaxis] == np.indices((self.mu.shape[0],))).sum(
                        axis=(0, 1))).reshape(self.sigma.shape))

            nb_param = np.prod(self.p.shape) + np.prod(self.t.shape) + np.prod(self.mu.shape) + np.prod(
                self.sigma.shape)
            norm_param = (1 / nb_param) * np.sqrt(
                ((self.p - prev_p) ** 2).sum() + ((self.t - prev_t) ** 2).sum() + ((self.mu - prev_mu) ** 2).sum() + (
                        (self.sigma - prev_sigma) ** 2).sum())
            print({'iter': q + 1, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma,
                   'diff_norm_param': norm_param})
            if norm_param < early_stopping:
                break

    def get_param_SEM(self, data, iter, early_stopping=0):
        print({'iter': 0, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma})
        for q in range(iter):
            prev_p = self.p
            prev_t = self.t
            prev_mu = self.mu
            prev_sigma = self.sigma

            gaussians = np_multivariate_normal_pdf(data, self.mu, self.sigma)
            forward, backward = self.get_forward_backward(gaussians)

            hidden = self.simul_hidden_apost(backward, gaussians)
            hiddenc = np.stack((hidden[:-1], hidden[1:]), axis=-1)
            aux = np.moveaxis(np.indices(self.t.shape), 0, -1)
            broadc = (len(aux.shape) - len(hiddenc.shape) + 1)
            c = (1 / (len(data) - 1)) * (
                np.all(hiddenc.reshape((hiddenc.shape[0],) + (1,) * broadc + hiddenc.shape[1:]) == aux, axis=-1).sum(
                    axis=0)).reshape(self.t.shape)

            self.p = (1 / (len(data))) * (hidden[..., np.newaxis] == np.indices((self.p.shape[0],))).sum(axis=0)
            self.t = (c.T / self.p).T

            broadc = (len(self.mu.shape) - len(data.shape) + 1)
            self.mu = (((hidden[..., np.newaxis] == np.indices((self.mu.shape[0],))) * data).sum(axis=0) / (
                    hidden[..., np.newaxis] == np.indices((self.mu.shape[0],))).sum(axis=0)).reshape(self.mu.shape)
            self.sigma = (((hidden[..., np.newaxis] == np.indices((self.mu.shape[0],))).reshape(
                ((hidden.shape[0],) + self.sigma.shape)) * np.einsum('...i,...j',
                                                                     (data.reshape((data.shape[0],) + (
                                                                         1,) * broadc + data.shape[1:]) - self.mu),
                                                                     (data.reshape(
                                                                         (data.shape[0],) + (
                                                                             1,) * broadc + data.shape[
                                                                                            1:]) - self.mu))).sum(
                axis=0)
                          / ((hidden[..., np.newaxis] == np.indices((self.mu.shape[0],))).sum(
                        axis=0)).reshape(self.sigma.shape))

            nb_param = np.prod(self.p.shape) + np.prod(self.t.shape) + np.prod(self.mu.shape) + np.prod(
                self.sigma.shape)
            norm_param = (1 / nb_param) * np.sqrt(
                ((self.p - prev_p) ** 2).sum() + ((self.t - prev_t) ** 2).sum() + ((self.mu - prev_mu) ** 2).sum() + (
                        (self.sigma - prev_sigma) ** 2).sum())
            print({'iter': q + 1, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma,
                   'diff_norm_param': norm_param})
            if norm_param < early_stopping:
                break

    def get_param_supervised(self, data, hidden):

        hiddenc = np.stack((hidden[:-1], hidden[1:]), axis=-1)
        aux = np.moveaxis(np.indices(self.t.shape), 0, -1)
        broadc = (len(aux.shape) - len(hiddenc.shape) + 1)
        c = (1 / (len(data) - 1)) * (
            np.all(hiddenc.reshape((hiddenc.shape[0],) + (1,) * broadc + hiddenc.shape[1:]) == aux, axis=-1).sum(
                axis=0)).reshape(self.t.shape)

        self.p = (1 / (len(data))) * (hidden[..., np.newaxis] == np.indices((self.p.shape[0],))).sum(axis=0)
        self.t = (c.T / self.p).T
        self.t[np.isnan(self.t)] = 0

        broadc = (len(self.mu.shape) - len(data.shape) + 1)
        self.mu = (((hidden[..., np.newaxis] == np.indices((self.mu.shape[0],))) * data).sum(axis=0) / (
                hidden[..., np.newaxis] == np.indices((self.mu.shape[0],))).sum(axis=0)).reshape(self.mu.shape)
        self.sigma = (((hidden[..., np.newaxis] == np.indices((self.mu.shape[0],))).reshape(
            ((hidden.shape[0],) + self.sigma.shape)) * np.einsum('...i,...j',
                                                                 (data.reshape((data.shape[0],) + (
                                                                     1,) * broadc + data.shape[1:]) - self.mu),
                                                                 (data.reshape(
                                                                     (data.shape[0],) + (
                                                                         1,) * broadc + data.shape[
                                                                                        1:]) - self.mu))).sum(
            axis=0)
                      / ((hidden[..., np.newaxis] == np.indices((self.mu.shape[0],))).sum(
                    axis=0)).reshape(self.sigma.shape))


class HSMC_ctod:
    __slots__ = ('p', 't', 'mu', 'sigma', 'nbc_x', 'nbc_u')

    def __init__(self, nbc_x, nbc_u, p=None, t=None, mu=None, sigma=None):
        self.p = p
        self.t = t
        self.mu = mu
        self.sigma = sigma
        self.nbc_x = nbc_x
        self.nbc_u = nbc_u

    def init_data_prior(self, data, scale=1):
        nb_class = self.nbc_x * self.nbc_u
        u = np.ones((self.nbc_x, self.nbc_x, self.nbc_u)) * (1 / self.nbc_u)
        x = np.array([1 / self.nbc_x] * self.nbc_x)
        a = np.full((self.nbc_x, self.nbc_x), 1 / (2 * (self.nbc_x - 1)))
        a = a - np.diag(np.diag(a))
        a = np.diag(np.array([1 / 2] * self.nbc_x)) + a
        self.p = (np.sum((a.T * x).T * u.T, axis=1)).T.flatten()
        b = np.repeat(np.eye(self.nbc_x, self.nbc_x, k=0), self.nbc_u, axis=0)
        idx = [i for i in range(b.shape[0]) if ((i + 1) % self.nbc_u == 0)]
        for i, e in enumerate(idx):
            b[e] = a[i]

        a = u
        ut = [[np.eye(self.nbc_u, k=1) for n1 in range(self.nbc_x)] for n2 in range(int(nb_class / self.nbc_u))]
        for i, e in enumerate(ut):
            for j, p in enumerate(e):
                p[-1] = a[i, j]
        ut = np.block(ut)
        for i in range(1, self.nbc_x + 1):
            ut[:, (i - 1) * self.nbc_u:i * self.nbc_u] = (
                    ut[:, (i - 1) * self.nbc_u:i * self.nbc_u].T * b[:, i - 1]).T
        self.t = ut
        M = np.mean(data, axis=0)
        Sig = np.cov(data, rowvar=False).reshape(data.shape[1], data.shape[1])
        self.mu = [None] * self.nbc_x
        self.sigma = [None] * self.nbc_x
        for l in range(self.nbc_x):
            if l % 2 == 0:
                self.mu[l] = M - scale * ((l / (self.nbc_x / 2)) * np.sum(Sig, axis=1))
            else:
                self.mu[l] = M + scale * ((l / (self.nbc_x / 2)) * np.sum(Sig, axis=1))
            self.sigma[l] = Sig

        self.mu = np.array(self.mu)
        self.sigma = np.array(self.sigma)

    def init_kmeans(self, data):
        kmeans = KMeans(n_clusters=self.nbc_x).fit(data)
        hidden = kmeans.labels_
        hiddenc = np.stack((hidden[:-1], hidden[1:]), axis=-1)
        aux = np.moveaxis(np.indices((self.nbc_x, self.nbc_x)), 0, -1)
        broadc = (len(aux.shape) - len(hiddenc.shape) + 1)
        self.mu = np.zeros((self.nbc_x, 1 * len(data[0])))
        self.sigma = np.zeros((self.nbc_x, 1 * len(data[0]), 1 * len(data[0])))
        c = (1 / (len(data) - 1)) * (
            np.all(hiddenc.reshape((hiddenc.shape[0],) + (1,) * broadc + hiddenc.shape[1:]) == aux, axis=-1).sum(
                axis=0)).reshape((self.nbc_x, self.nbc_x))

        u = np.ones((self.nbc_x, self.nbc_x, self.nbc_u)) * (1 / self.nbc_u)
        self.p = (np.sum(c * u.T, axis=1)).T.flatten()
        a = (c.T / np.sum(c, axis=1)).T
        b = np.repeat(np.eye(self.nbc_x, self.nbc_x, k=0), self.nbc_u, axis=0)
        idx = [i for i in range(b.shape[0]) if ((i + 1) % self.nbc_u == 0)]
        for i, e in enumerate(idx):
            b[e] = a[i]

        a = u
        ut = [[np.eye(self.nbc_u, k=1) for n1 in range(self.nbc_x)] for n2 in
              range(int((self.nbc_x * self.nbc_u) / self.nbc_u))]
        for i, e in enumerate(ut):
            for j, p in enumerate(e):
                p[-1] = a[i, j]
        ut = np.block(ut)
        for i in range(1, self.nbc_x + 1):
            ut[:, (i - 1) * self.nbc_u:i * self.nbc_u] = (
                    ut[:, (i - 1) * self.nbc_u:i * self.nbc_u].T * b[:, i - 1]).T
        self.t = ut

        broadc = (len(self.mu.shape) - len(data.shape) + 1)
        self.mu = (((hidden[..., np.newaxis] == np.indices((self.mu.shape[0],))) * data).sum(axis=0) / (
                hidden[..., np.newaxis] == np.indices((self.mu.shape[0],))).sum(axis=0)).reshape(self.mu.shape)
        self.sigma = (((hidden[..., np.newaxis] == np.indices((self.mu.shape[0],))).reshape(
            ((hidden.shape[0],) + self.sigma.shape)) * np.einsum('...i,...j',
                                                                 (data.reshape((data.shape[0],) + (
                                                                     1,) * broadc + data.shape[1:]) - self.mu),
                                                                 (data.reshape(
                                                                     (data.shape[0],) + (
                                                                         1,) * broadc + data.shape[
                                                                                        1:]) - self.mu))).sum(
            axis=0)
                      / ((hidden[..., np.newaxis] == np.indices((self.mu.shape[0],))).sum(
                    axis=0)).reshape(self.sigma.shape))

    def init_from_markov_chain(self, data, iter=100, early_stopping=10 ** -4):
        hmc = HMC_ctod(self.nbc_x)
        hmc.init_data_prior(data)
        hmc.get_param_EM(data, iter, early_stopping=early_stopping)
        c = (hmc.t.T * hmc.p).T
        u = np.ones((self.nbc_x, self.nbc_x, self.nbc_u)) * (1 / self.nbc_u)
        nb_class = self.nbc_x * self.nbc_u
        self.p = (np.sum(c * u.T, axis=1)).T.flatten()
        a = (c.T / np.sum(c, axis=1)).T
        b = np.repeat(np.eye(self.nbc_x, self.nbc_x, k=0), self.nbc_u, axis=0)
        idx = [i for i in range(b.shape[0]) if ((i + 1) % self.nbc_u == 0)]
        for i, e in enumerate(idx):
            b[e] = a[i]

        a = u
        ut = [[np.eye(self.nbc_u, k=1) for n1 in range(self.nbc_x)] for n2 in range(int(nb_class / self.nbc_u))]
        for i, e in enumerate(ut):
            for j, p in enumerate(e):
                p[-1] = a[i, j]
        ut = np.block(ut)
        for i in range(1, self.nbc_x + 1):
            ut[:, (i - 1) * self.nbc_u:i * self.nbc_u] = (
                    ut[:, (i - 1) * self.nbc_u:i * self.nbc_u].T * b[:, i - 1]).T
        self.t = ut
        self.mu = hmc.mu
        self.sigma = hmc.sigma

    def give_param(self, c, u, mu, sigma):
        nb_class = self.nbc_x * self.nbc_u
        self.p = (np.sum(c * u.T, axis=1)).T.flatten()
        a = (c.T / np.sum(c, axis=1)).T
        b = np.repeat(np.eye(self.nbc_x, self.nbc_x, k=0), self.nbc_u, axis=0)
        idx = [i for i in range(b.shape[0]) if ((i + 1) % self.nbc_u == 0)]
        for i, e in enumerate(idx):
            b[e] = a[i]

        a = u
        ut = [[np.eye(self.nbc_u, k=1) for n1 in range(self.nbc_x)] for n2 in range(int(nb_class / self.nbc_u))]
        for i, e in enumerate(ut):
            for j, p in enumerate(e):
                p[-1] = a[i, j]
        ut = np.block(ut)
        for i in range(1, self.nbc_x + 1):
            ut[:, (i - 1) * self.nbc_u:i * self.nbc_u] = (
                    ut[:, (i - 1) * self.nbc_u:i * self.nbc_u].T * b[:, i - 1]).T
        self.t = ut
        self.mu = mu
        self.sigma = sigma

    def seg_map(self, data):
        pass

    def seg_mpm(self, data):
        mu = np.repeat(self.mu, self.nbc_u, axis=0)
        sigma = np.repeat(self.sigma, self.nbc_u, axis=0)
        gaussians = np_multivariate_normal_pdf(data, mu, sigma)
        forward, backward = self.get_forward_backward(gaussians)
        p_apost = forward * backward
        p_apost = p_apost / (p_apost.sum(axis=1)[..., np.newaxis])
        p_apost_x = (p_apost.reshape((p_apost.shape[0], self.nbc_x, self.nbc_u))).sum(axis=2)
        return np.argmax(p_apost_x, axis=1)

    def seg_mpm_u(self, data):
        mu = np.repeat(self.mu, self.nbc_u, axis=0)
        sigma = np.repeat(self.sigma, self.nbc_u, axis=0)
        gaussians = np_multivariate_normal_pdf(data, mu, sigma)
        forward, backward = self.get_forward_backward(gaussians)
        p_apost = forward
        p_apost_u = (p_apost.reshape((p_apost.shape[0], self.nbc_x, self.nbc_u))).sum(axis=1)
        return np.argmax(p_apost_u, axis=1)

    def simul_hidden_apost(self, backward, gaussians):
        res = np.zeros(len(backward), dtype=int)
        T = self.t
        aux = (gaussians[0] * self.p) * backward[0]
        test = np.random.multinomial(1, aux / np.sum(aux))
        res[0] = np.argmax(test)
        tapost = (
            (gaussians[1:, np.newaxis, :]
             * backward[1:, np.newaxis, :]
             * T[np.newaxis, :, :])
        )
        tapost = tapost / tapost.sum(axis=2)[..., np.newaxis]
        for i in range(1, len(res)):
            test = np.random.multinomial(1, tapost[i - 1, res[i - 1], :])
            res[i] = np.argmax(test)
        return res

    def simul_visible(self, hidden):
        res = np.zeros((len(hidden), self.mu.shape[-1]))
        mu = np.repeat(self.mu, self.nbc_u, axis=0)
        sigma = np.repeat(self.sigma, self.nbc_u, axis=0)
        for i in range(0, len(hidden)):
            res[i] = multivariate_normal.rvs(mu[hidden[i]], sigma[hidden[i]])
        return np.array(res)

    def generate_sample(self, length, x_only=True):
        hidden = np.zeros(length, dtype=int)
        visible = np.zeros((length, self.mu.shape[-1]))
        T = self.t
        mu = np.repeat(self.mu, self.nbc_u, axis=0)
        sigma = np.repeat(self.sigma, self.nbc_u, axis=0)
        test = np.random.multinomial(1, self.p)
        hidden[0] = np.argmax(test)
        visible[0] = multivariate_normal.rvs(mu[hidden[0]], sigma[hidden[0]])
        for i in range(1, length):
            test = np.random.multinomial(1, T[hidden[i - 1], :])
            hidden[i] = np.argmax(test)
            visible[i] = multivariate_normal.rvs(mu[hidden[i]], sigma[hidden[i]])
        if x_only:
            hidden = convert_multcls_vectors(hidden, (self.nbc_u, self.nbc_x))[:, 1]

        return hidden, visible

    def get_forward_backward(self, gaussians):
        forward = np.zeros((len(gaussians), self.t.shape[0]))
        backward = np.zeros((len(gaussians), self.t.shape[0]))
        backward[len(gaussians) - 1] = np.ones(self.t.shape[0])
        forward[0] = self.p * gaussians[0]
        forward[0] = forward[0] / (forward[0].sum())
        T = self.t
        for l in range(1, len(gaussians)):
            k = len(gaussians) - 1 - l
            forward[l] = gaussians[l] * (forward[l - 1] @ T)
            forward[l] = forward[l] / forward[l].sum()
            backward[k] = (gaussians[k + 1] * backward[k + 1]) @ T.T
            backward[k] = backward[k] / (backward[k].sum())

        return forward, backward

    def get_forward_backward_supervised(self, gaussians, hiddenx):
        forward = np.zeros((len(gaussians), self.nbc_u))
        backward = np.zeros((len(gaussians), self.nbc_u))
        backward[len(gaussians) - 1] = np.ones(self.nbc_u)
        forward[0] = self.p[hiddenx[0] * self.nbc_u:(hiddenx[0] + 1) * self.nbc_u] * gaussians[0]
        forward[0] = forward[0] / (forward[0].sum())
        T = self.t
        for l in range(1, len(gaussians)):
            k = len(gaussians) - 1 - l
            forward[l] = gaussians[l] * (
                        forward[l - 1] @ T[hiddenx[l - 1] * self.nbc_u:(hiddenx[l - 1] + 1) * self.nbc_u,
                                         hiddenx[l] * self.nbc_u:(hiddenx[l] + 1) * self.nbc_u])
            forward[l] = forward[l] / forward[l].sum()
            backward[k] = (gaussians[k + 1] * backward[k + 1]) @ (
            T[hiddenx[k] * self.nbc_u:(hiddenx[k] + 1) * self.nbc_u,
            hiddenx[k + 1] * self.nbc_u:(hiddenx[k + 1] + 1) * self.nbc_u]).T
            backward[k] = backward[k] / (backward[k].sum())

        return forward, backward

    def get_param_EM(self, data, iter, early_stopping=0):
        print({'iter': 0, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma})
        for q in range(iter):
            prev_p = self.p
            prev_t = self.t
            prev_mu = self.mu
            prev_sigma = self.sigma
            T = self.t
            mu = np.repeat(self.mu, self.nbc_u, axis=0)
            sigma = np.repeat(self.sigma, self.nbc_u, axis=0)
            gaussians = np_multivariate_normal_pdf(data, mu, sigma)
            forward, backward = self.get_forward_backward(gaussians)
            gamma = (
                    forward[:-1, :, np.newaxis]
                    * (gaussians[1:, np.newaxis, :]
                       * backward[1:, np.newaxis, :]
                       * T[np.newaxis, :, :])
            )
            gamma = gamma / (gamma.sum(axis=(1, 2))[..., np.newaxis, np.newaxis])
            psi = forward * backward
            psi = psi / (psi.sum(axis=1)[..., np.newaxis])
            self.t = np.transpose(np.transpose((gamma.sum(axis=0))) / (psi[:-1:].sum(axis=0)))
            self.p = (psi.sum(axis=0)) / psi.shape[0]
            psi = (psi.reshape((psi.shape[0], self.nbc_x, self.nbc_u))).sum(axis=2)
            self.mu = (((psi * data).sum(axis=0)) / (psi.sum(axis=0))).reshape(self.mu.shape)
            broadc = (len(self.mu.shape) - len(data.shape) + 1)
            self.sigma = (psi.reshape(((psi.shape[0],) + self.sigma.shape)) * np.einsum('...i,...j',
                                                                                        (data.reshape(
                                                                                            (data.shape[0],) + (
                                                                                                1,) * broadc + data.shape[
                                                                                                               1:]) - self.mu),
                                                                                        (data.reshape(
                                                                                            (data.shape[0],) + (
                                                                                                1,) * broadc + data.shape[
                                                                                                               1:]) - self.mu))).sum(
                axis=0) / ((psi.sum(axis=0)).reshape(self.sigma.shape))

            nb_param = np.prod(self.p.shape) + np.prod(self.t.shape) + np.prod(self.mu.shape) + np.prod(
                self.sigma.shape)
            norm_param = (1 / nb_param) * np.sqrt(
                ((self.p - prev_p) ** 2).sum() + ((self.t - prev_t) ** 2).sum() + ((self.mu - prev_mu) ** 2).sum() + (
                        (self.sigma - prev_sigma) ** 2).sum())
            print({'iter': q + 1, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma,
                   'diff_norm_param': norm_param})
            if norm_param < early_stopping:
                break

    def get_param_ICE(self, data, iter, Nb_simul, early_stopping=0):
        print({'iter': 0, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma})
        for q in range(iter):
            prev_p = self.p
            prev_t = self.t
            prev_mu = self.mu
            prev_sigma = self.sigma

            T = self.t
            mu = np.repeat(self.mu, self.nbc_u, axis=0)
            sigma = np.repeat(self.sigma, self.nbc_u, axis=0)
            gaussians = np_multivariate_normal_pdf(data, mu, sigma)
            forward, backward = self.get_forward_backward(gaussians)
            hidden = np.stack([self.simul_hidden_apost(backward, gaussians) for n in range(Nb_simul)], axis=0)
            gamma = (
                    forward[:-1, :, np.newaxis]
                    * (gaussians[1:, np.newaxis, :]
                       * backward[1:, np.newaxis, :]
                       * T[np.newaxis, :, :])
            )
            gamma = gamma / (gamma.sum(axis=(1, 2))[..., np.newaxis, np.newaxis])
            psi = forward * backward
            psi = psi / (psi.sum(axis=1)[..., np.newaxis])
            self.t = np.transpose(np.transpose((gamma.sum(axis=0))) / (psi[:-1:].sum(axis=0)))
            self.p = (psi.sum(axis=0)) / psi.shape[0]

            broadc = (len(self.mu.shape) - len(data.shape) + 1)
            self.mu = (
                    (((hidden[..., np.newaxis] // self.nbc_u) == np.indices((self.mu.shape[0],))) * data).sum(
                        axis=(0, 1))
                    /
                    ((hidden[..., np.newaxis] // self.nbc_u) == np.indices((self.mu.shape[0],))).sum(
                        axis=(0, 1))).reshape(self.mu.shape)
            self.sigma = (
                    (((hidden[..., np.newaxis] // self.nbc_u) == np.indices((self.mu.shape[0],))).reshape(
                        ((hidden.shape[0], hidden.shape[1]) + self.sigma.shape)) * np.einsum('...i,...j',
                                                                                             (data.reshape(
                                                                                                 (data.shape[0],) + (
                                                                                                     1,) * broadc + data.shape[
                                                                                                                    1:]) - self.mu),
                                                                                             (data.reshape(
                                                                                                 (data.shape[0],) + (
                                                                                                     1,) * broadc + data.shape[
                                                                                                                    1:]) - self.mu))).sum(
                        axis=(0, 1)) /
                    (((hidden[..., np.newaxis] // self.nbc_u) == np.indices((self.mu.shape[0],))).sum(
                        axis=(0, 1))).reshape(self.sigma.shape))

            nb_param = np.prod(self.p.shape) + np.prod(self.t.shape) + np.prod(self.mu.shape) + np.prod(
                self.sigma.shape)
            norm_param = (1 / nb_param) * np.sqrt(
                ((self.p - prev_p) ** 2).sum() + ((self.t - prev_t) ** 2).sum() + ((self.mu - prev_mu) ** 2).sum() + (
                        (self.sigma - prev_sigma) ** 2).sum())
            print({'iter': q + 1, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma,
                   'diff_norm_param': norm_param})
            if norm_param < early_stopping:
                break

    def get_param_SEM(self, data, iter, early_stopping=0):
        print({'iter': 0, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma})
        for q in range(iter):
            prev_p = self.p
            prev_t = self.t
            prev_mu = self.mu
            prev_sigma = self.sigma

            mu = np.repeat(self.mu, self.nbc_u, axis=0)
            sigma = np.repeat(self.sigma, self.nbc_u, axis=0)
            gaussians = np_multivariate_normal_pdf(data, mu, sigma)
            forward, backward = self.get_forward_backward(gaussians)
            hidden = self.simul_hidden_apost(backward, gaussians)
            hiddenc = np.stack((hidden[:-1], hidden[1:]), axis=-1)
            aux = np.moveaxis(np.indices(self.t.shape), 0, -1)
            broadc = (len(aux.shape) - len(hiddenc.shape) + 1)
            c = (1 / (len(data) - 1)) * (
                np.all(hiddenc.reshape((hiddenc.shape[0],) + (1,) * broadc + hiddenc.shape[1:]) == aux, axis=-1).sum(
                    axis=0)).reshape(self.t.shape)

            self.p = (1 / (len(data))) * (hidden[..., np.newaxis] == np.indices((self.p.shape[0],))).sum(axis=0)
            self.t = (c.T / self.p).T
            self.t[np.isnan(self.t)] = 0

            broadc = (len(self.mu.shape) - len(data.shape) + 1)
            self.mu = ((((hidden[..., np.newaxis] // self.nbc_u) == np.indices((self.mu.shape[0],))) * data).sum(
                axis=0) / (
                               (hidden[..., np.newaxis] // self.nbc_u) == np.indices((self.mu.shape[0],))).sum(
                axis=0)).reshape(self.mu.shape)
            self.sigma = ((((hidden[..., np.newaxis] // self.nbc_u) == np.indices((self.mu.shape[0],))).reshape(
                ((hidden.shape[0],) + self.sigma.shape)) * np.einsum('...i,...j',
                                                                     (data.reshape((data.shape[0],) + (
                                                                         1,) * broadc + data.shape[1:]) - self.mu),
                                                                     (data.reshape(
                                                                         (data.shape[0],) + (
                                                                             1,) * broadc + data.shape[
                                                                                            1:]) - self.mu))).sum(
                axis=0)
                          / (((hidden[..., np.newaxis] // self.nbc_u) == np.indices((self.mu.shape[0],))).sum(
                        axis=0)).reshape(self.sigma.shape))

            nb_param = np.prod(self.p.shape) + np.prod(self.t.shape) + np.prod(self.mu.shape) + np.prod(
                self.sigma.shape)
            norm_param = (1 / nb_param) * np.sqrt(
                ((self.p - prev_p) ** 2).sum() + ((self.t - prev_t) ** 2).sum() + ((self.mu - prev_mu) ** 2).sum() + (
                        (self.sigma - prev_sigma) ** 2).sum())
            print({'iter': q + 1, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma,
                   'diff_norm_param': norm_param})
            if norm_param < early_stopping:
                break

    def get_param_supervised(self, data, hidden, iter=100, early_stopping=0):
        aux = np.moveaxis(np.indices((self.nbc_x, self.nbc_x)), 0, -1)
        hiddenc = np.stack((hidden[:-1], hidden[1:]), axis=-1)
        broadc = (len(self.mu.shape) - len(data.shape) + 1)
        self.mu = (((hidden[..., np.newaxis] == np.indices((self.mu.shape[0],))) * data).sum(axis=0) / (
                hidden[..., np.newaxis] == np.indices((self.mu.shape[0],))).sum(axis=0)).reshape(self.mu.shape)
        self.sigma = (((hidden[..., np.newaxis] == np.indices((self.mu.shape[0],))).reshape(
            ((hidden.shape[0],) + self.sigma.shape)) * np.einsum('...i,...j',
                                                                 (data.reshape((data.shape[0],) + (
                                                                     1,) * broadc + data.shape[1:]) - self.mu),
                                                                 (data.reshape(
                                                                     (data.shape[0],) + (
                                                                         1,) * broadc + data.shape[
                                                                                        1:]) - self.mu))).sum(
            axis=0)
                      / ((hidden[..., np.newaxis] == np.indices((self.mu.shape[0],))).sum(
                    axis=0)).reshape(self.sigma.shape))
        print({'iter': 0, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma})
        for q in range(iter):
            prev_p = self.p
            prev_t = self.t
            prev_mu = self.mu
            prev_sigma = self.sigma

            T = np.array([self.t[h[0] * self.nbc_u:(h[0] + 1) * self.nbc_u, h[1] * self.nbc_u:(h[1] + 1) * self.nbc_u] for l, h in enumerate(hiddenc)])
            mu = np.repeat(self.mu, self.nbc_u, axis=0)
            sigma = np.repeat(self.sigma, self.nbc_u, axis=0)
            gaussians = np_multivariate_normal_pdf(data, mu, sigma)
            gaussians = np.array([gaussians[l][h * self.nbc_u:(h + 1) * self.nbc_u] for l, h in enumerate(hidden)])
            forward, backward = self.get_forward_backward_supervised(gaussians, hidden)
            gamma = (
                    forward[:-1, :, np.newaxis]
                    * (gaussians[1:, np.newaxis, :]
                       * backward[1:, np.newaxis, :]
                       * T)
            )
            gamma = gamma / (gamma.sum(axis=(1, 2))[..., np.newaxis, np.newaxis])
            psi = forward * backward
            psi = psi / (psi.sum(axis=1)[..., np.newaxis])
            u = (1/gamma.shape[0])*gamma.sum(axis=0)
            pu = (1/psi.shape[0])*psi.sum(axis=0)

            broadc = (len(aux.shape) - len(hiddenc.shape) + 1)
            px = (((psi[:,:,np.newaxis] * (hidden[..., np.newaxis] == np.indices((self.nbc_x,)))[:,np.newaxis,:]).sum(axis=0)) / (psi[:,:,np.newaxis].sum(axis=0)))
            px[np.isnan(px)] = 0
            x = (gamma[:,np.newaxis,:,np.newaxis,:] * np.all(hiddenc.reshape((hiddenc.shape[0],) + (1,) * broadc + hiddenc.shape[1:]) == aux, axis=-1)[:,:,np.newaxis,:,np.newaxis]).sum(
                axis=0) / (gamma[:,np.newaxis,:,np.newaxis,:]).sum(axis=0)
            x[np.isnan(x)] = 0
            c = ((u[np.newaxis,:,np.newaxis,:]*x).reshape((self.nbc_x*self.nbc_u,self.nbc_x*self.nbc_u)))
            self.p = (pu[:, np.newaxis]*px).T.reshape((self.nbc_x*self.nbc_u,))
            self.t = (c.T/self.p).T
            self.t[np.isnan(self.t)] = 0

            nb_param = np.prod(self.p.shape) + np.prod(self.t.shape) + np.prod(self.mu.shape) + np.prod(
                self.sigma.shape)
            norm_param = (1 / nb_param) * np.sqrt(
                ((self.p - prev_p) ** 2).sum() + ((self.t - prev_t) ** 2).sum() + ((self.mu - prev_mu) ** 2).sum() + (
                        (self.sigma - prev_sigma) ** 2).sum())
            print({'iter': q + 1, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma,
                   'diff_norm_param': norm_param})
            if norm_param < early_stopping:
                break


class HEMC_ctod:
    __slots__ = ('p', 't', 'lx', 'mu', 'sigma', 'nbc_x', 'nbc_u')

    def __init__(self, nbc_x, p=None, t=None, mu=None, sigma=None):
        self.p = p
        self.t = t
        self.mu = mu
        self.sigma = sigma
        self.nbc_x = nbc_x
        self.nbc_u = (2 ** nbc_x) - 1

    def init_data_prior(self, data, scale=1):
        self.lx = np.stack([convertcls_vect(i + 1, (2,) * self.nbc_x) for i in range(self.nbc_u)], axis=0).T
        card = 1 / np.sum(self.lx, axis=0)
        a = np.full((self.nbc_u, self.nbc_u), 1 / (2 * (self.nbc_u - 1)))
        a = a - np.diag(np.diag(a))
        # self.p = card * np.array([1 / self.nbc_u] * self.nbc_u)
        # self.t = card * np.diag(np.array([1 / 2] * self.nbc_u)) + a
        p = np.array([1 / self.nbc_u] * self.nbc_u)
        t = np.diag(np.array([1 / 2] * self.nbc_u)) + a
        u = (t.T / p).T
        self.p = card * u.sum(axis=1)
        self.t = ((np.outer(card, card) * u).T / self.p).T

        M = np.mean(data, axis=0)
        Sig = np.cov(data, rowvar=False).reshape(data.shape[1], data.shape[1])
        self.mu = [None] * self.nbc_x
        self.sigma = [None] * self.nbc_x
        for l in range(self.nbc_x):
            if l % 2 == 0:
                self.mu[l] = M - scale * ((l / (self.nbc_x / 2)) * np.sum(Sig, axis=1))
            else:
                self.mu[l] = M + scale * ((l / (self.nbc_x / 2)) * np.sum(Sig, axis=1))
            self.sigma[l] = Sig

        self.mu = np.array(self.mu)
        self.sigma = np.array(self.sigma)

    def init_kmeans(self, data, perturbation_param=0.1):
        self.lx = np.stack([convertcls_vect(i + 1, (2,) * self.nbc_x) for i in range(self.nbc_u)], axis=0).T
        card = 1 / np.sum(self.lx, axis=0)
        self.t = np.zeros((self.nbc_u, self.nbc_u))
        self.p = np.zeros((self.nbc_u,))
        self.mu = np.zeros((self.nbc_x, 1 * len(data[0])))
        self.sigma = np.zeros((self.nbc_x, 1 * len(data[0]), 1 * len(data[0])))

        kmeans = KMeans(n_clusters=self.nbc_x).fit(data)
        hidden = kmeans.labels_
        hiddenc = np.stack((hidden[:-1], hidden[1:]), axis=-1)
        aux = np.moveaxis(np.indices((self.nbc_x, self.nbc_x)), 0, -1)
        broadc = (len(aux.shape) - len(hiddenc.shape) + 1)

        c = (1 / (len(data) - 1)) * (
            np.all(hiddenc.reshape((hiddenc.shape[0],) + (1,) * broadc + hiddenc.shape[1:]) == aux, axis=-1).sum(
                axis=0)).reshape((self.nbc_x, self.nbc_x))
        perturbation = np.random.uniform(0, 1, size=(self.nbc_u - self.nbc_x))
        perturbation = (perturbation / perturbation.sum()) * perturbation_param
        index1 = self.lx.T.sum(axis=1) == 1
        index2 = self.lx.T.sum(axis=1) != 1
        res = np.copy(self.lx.T)
        res[index1] = res[index1] * (1 - perturbation_param)
        res[index2] = (res[index2].T * perturbation).T
        u = res @ c @ res.T
        self.p = card * u.sum(axis=1)
        self.t = ((np.outer(card, card) * u).T / self.p).T

        broadc = (len(self.mu.shape) - len(data.shape) + 1)
        self.mu = (((hidden[..., np.newaxis] == np.indices((self.mu.shape[0],))) * data).sum(axis=0) / (
                hidden[..., np.newaxis] == np.indices((self.mu.shape[0],))).sum(axis=0)).reshape(self.mu.shape)
        self.sigma = (((hidden[..., np.newaxis] == np.indices((self.mu.shape[0],))).reshape(
            ((hidden.shape[0],) + self.sigma.shape)) * np.einsum('...i,...j',
                                                                 (data.reshape((data.shape[0],) + (
                                                                     1,) * broadc + data.shape[1:]) - self.mu),
                                                                 (data.reshape(
                                                                     (data.shape[0],) + (
                                                                         1,) * broadc + data.shape[
                                                                                        1:]) - self.mu))).sum(
            axis=0)
                      / ((hidden[..., np.newaxis] == np.indices((self.mu.shape[0],))).sum(
                    axis=0)).reshape(self.sigma.shape))

    def init_from_markov_chain(self, data, iter=100, early_stopping=10 ** -4, perturbation_param=0.03):
        hmc = HMC_ctod(self.nbc_x)
        hmc.init_data_prior(data)
        hmc.get_param_EM(data, iter, early_stopping=early_stopping)
        c = (hmc.t.T * hmc.p).T
        self.lx = np.stack([convertcls_vect(i + 1, (2,) * self.nbc_x) for i in range(self.nbc_u)], axis=0).T
        card = 1 / np.sum(self.lx, axis=0)
        perturbation = np.random.uniform(0, 1, size=(self.nbc_u - self.nbc_x))
        perturbation = (perturbation / perturbation.sum()) * perturbation_param
        index1 = self.lx.T.sum(axis=1) == 1
        index2 = self.lx.T.sum(axis=1) != 1
        res = np.copy(self.lx.T)
        res[index1] = res[index1] * (1 - perturbation_param)
        res[index2] = (res[index2].T * perturbation).T
        u = res @ c @ res.T
        self.p = card * u.sum(axis=1)
        self.t = ((np.outer(card, card) * u).T / self.p).T
        self.mu = hmc.mu
        self.sigma = hmc.sigma
        print(self.lx)

    def give_param(self, u, mu, sigma):
        self.lx = np.stack([convertcls_vect(i + 1, (2,) * self.nbc_x) for i in range(self.nbc_u)], axis=0).T
        card = 1 / np.sum(self.lx, axis=0)
        self.p = card * u.sum(axis=1)
        self.t = ((np.outer(card, card) * u).T / self.p).T
        self.t[np.isnan(self.t)] = 0
        self.mu = mu
        self.sigma = sigma

    def seg_map(self, data):
        pass

    def seg_mpm(self, data):
        mu = np.repeat(self.mu, self.nbc_u, axis=0)
        sigma = np.repeat(self.sigma, self.nbc_u, axis=0)
        gaussians = np_multivariate_normal_pdf(data, mu, sigma)
        forward, backward = self.get_forward_backward(gaussians)
        p_apost = forward
        p_apost_x = (p_apost.reshape((p_apost.shape[0], self.nbc_x, self.nbc_u))).sum(axis=2)
        return np.argmax(p_apost_x, axis=1)

    def seg_mpm_u(self, data):
        mu = np.repeat(self.mu, self.nbc_u, axis=0)
        sigma = np.repeat(self.sigma, self.nbc_u, axis=0)
        gaussians = np_multivariate_normal_pdf(data, mu, sigma)
        forward, backward = self.get_forward_backward(gaussians)
        p_apost = forward
        p_apost_u = (p_apost.reshape((p_apost.shape[0], self.nbc_x, self.nbc_u))).sum(axis=1)
        return np.argmax(p_apost_u, axis=1)

    def simul_hidden_apost(self, backward, gaussians, x_only=False):
        res = np.zeros(len(backward), dtype=int)
        T = calc_transDS(self.t, self.lx)
        T[np.isnan(T)] = 0
        aux = backward[0] / backward[0].sum()
        test = np.random.multinomial(1, aux)
        res[0] = np.argmax(test)
        tapost = (
                (backward[1:, np.newaxis, :] * gaussians[1:, np.newaxis, :]
                 * T[np.newaxis, :, :]) / backward[:-1, :, np.newaxis])
        tapost[np.isnan(tapost)] = 0
        tapost = tapost / tapost.sum(axis=2)[..., np.newaxis]
        tapost[np.isnan(tapost)] = 0
        for i in range(1, len(res)):
            test = np.random.multinomial(1, tapost[i - 1, res[i - 1], :])
            res[i] = np.argmax(test)
        if x_only:
            res = convert_multcls_vectors(res, (self.nbc_u, self.nbc_x))[:, 1]
        return res

    def simul_visible(self, hidden):
        res = np.zeros((len(hidden), self.mu.shape[-1]))
        mu = np.repeat(self.mu, self.nbc_u, axis=0)
        sigma = np.repeat(self.sigma, self.nbc_u, axis=0)
        for i in range(0, len(hidden)):
            res[i] = multivariate_normal.rvs(mu[hidden[i]], sigma[hidden[i]])
        return np.array(res)

    def generate_sample(self, length, x_only=True):
        hidden = np.zeros(length, dtype=int)
        visible = np.zeros((length, self.mu.shape[-1]))
        backward = self.get_backward_apri(length)
        mu = np.repeat(self.mu, self.nbc_u, axis=0)
        sigma = np.repeat(self.sigma, self.nbc_u, axis=0)
        T = calc_transDS(self.t, self.lx)
        T[np.isnan(T)] = 0
        tapri = (
                (backward[1:, np.newaxis, :]
                 * T[np.newaxis, :, :]) / backward[:-1, :, np.newaxis]
        )
        tapri[np.isnan(tapri)] = 0
        tapri = tapri / tapri.sum(axis=2)[..., np.newaxis]
        tapri[np.isnan(tapri)] = 0
        test = np.random.multinomial(1, backward[0] / backward[0].sum())
        hidden[0] = np.argmax(test)
        visible[0] = multivariate_normal.rvs(mu[hidden[0]], sigma[hidden[0]])
        for i in range(1, length):
            test = np.random.multinomial(1, tapri[i - 1, hidden[i - 1], :])
            hidden[i] = np.argmax(test)
            visible[i] = multivariate_normal.rvs(mu[hidden[i]], sigma[hidden[i]])
        if x_only:
            hidden = convert_multcls_vectors(hidden, (self.nbc_u, self.nbc_x))[:, 1]
        return hidden, visible

    def get_backward_apri(self, length):
        C = calc_matDS((self.t.T * self.p).T, self.lx)
        T = calc_transDS(self.t, self.lx)
        T[np.isnan(T)] = 0
        backward = np.zeros((length, T.shape[0]))
        backward[length - 1] = np.ones(T.shape[0])
        for l in reversed(range(0, length - 1)):
            if l == 0:
                phi = C
            else:
                phi = T
            backward[l] = phi @ (backward[l + 1])
            backward[l] = backward[l] / (backward[l].sum())

        return backward

    def get_gaussians(self, data):
        mu = np.repeat(self.mu, self.nbc_u, axis=0)
        sigma = np.repeat(self.sigma, self.nbc_u, axis=0)
        return np_multivariate_normal_pdf(data, mu, sigma)

    def get_forward_backward(self, gaussians):
        C = calc_matDS((self.t.T * self.p).T, self.lx)
        T = calc_transDS(self.t, self.lx)
        T[np.isnan(T)] = 0
        forward = np.zeros((len(gaussians), T.shape[0]))
        backward = np.zeros((len(gaussians), T.shape[0]))
        backward[len(gaussians) - 1] = np.ones(T.shape[0])
        for l in reversed(range(0, len(gaussians) - 1)):
            if l == 0:
                phi = ((C * gaussians[l + 1]).T * gaussians[l]).T
            else:
                phi = T * gaussians[l + 1]
            backward[l] = phi @ (backward[l + 1])
            backward[l] = backward[l] / (backward[l].sum())

        forward[0] = backward[0] / np.sum(backward[0])
        tapost = (
                (backward[1:, np.newaxis, :] * gaussians[1:, np.newaxis, :]
                 * T[np.newaxis, :, :]) / backward[:-1, :, np.newaxis])
        tapost[np.isnan(tapost)] = 0
        tapost = tapost / tapost.sum(axis=2)[..., np.newaxis]
        tapost[np.isnan(tapost)] = 0
        for k in range(1, len(gaussians)):
            forward[k] = (forward[k - 1] @ tapost[k - 1])

        return forward, backward

    def get_forward_backward_supervised(self, gaussians, hiddenx):
        C = calc_matDS((self.t.T * self.p).T, self.lx)
        T = calc_transDS(self.t, self.lx)
        T[np.isnan(T)] = 0
        forward = np.zeros((len(gaussians), self.nbc_u))
        backward = np.zeros((len(gaussians), self.nbc_u))
        backward[len(gaussians) - 1] = np.ones(self.nbc_u)
        for l in reversed(range(0, len(gaussians) - 1)):
            if l == 0:
                phi = (((((C * gaussians[l + 1]).T * gaussians[l]).T).reshape(
                    (self.nbc_x, self.nbc_u, self.nbc_x, self.nbc_u)).sum(axis=(0, 2))).T * self.lx[hiddenx[l]]).T * self.lx[hiddenx[l+1]]
            else:
                phi = (T * gaussians[l + 1]).reshape((self.nbc_x, self.nbc_u, self.nbc_x, self.nbc_u)).sum(
                    axis=(0, 2)) * self.lx[hiddenx[l+1]]
            backward[l] = phi @ (backward[l + 1])
            backward[l] = backward[l] / (backward[l].sum())

        forward[0] = backward[0] / np.sum(backward[0])
        Tprime = gaussians[1:, np.newaxis, :] * T[np.newaxis, :, :]
        Tprime = Tprime.reshape((Tprime.shape[0], self.nbc_x, self.nbc_u, self.nbc_x, self.nbc_u)).sum(axis=(1, 3))
        tapost = (
                (backward[1:, np.newaxis, :] * self.lx[hiddenx][1:, np.newaxis, :]
                 * Tprime) / backward[:-1, :, np.newaxis])
        tapost[np.isnan(tapost)] = 0
        tapost = tapost / tapost.sum(axis=2)[..., np.newaxis]
        tapost[np.isnan(tapost)] = 0
        for k in range(1, len(gaussians)):
            forward[k] = (forward[k - 1] @ tapost[k - 1])

        return forward, backward

    def get_param_EM(self, data, iter, early_stopping=0):
        print({'iter': 0, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma})
        for q in range(iter):
            prev_p = self.p
            prev_t = self.t
            prev_mu = self.mu
            prev_sigma = self.sigma

            T = calc_transDS(self.t, self.lx)
            T[np.isnan(T)] = 0
            mu = np.repeat(self.mu, self.nbc_u, axis=0)
            sigma = np.repeat(self.sigma, self.nbc_u, axis=0)
            card = 1 / np.sum(self.lx, axis=0)
            gaussians = np_multivariate_normal_pdf(data, mu, sigma)
            forward, backward = self.get_forward_backward(gaussians)
            tapost = (
                    (backward[1:, np.newaxis, :] * gaussians[1:, np.newaxis, :]
                     * T[np.newaxis, :, :]) / backward[:-1, :, np.newaxis])
            tapost[np.isnan(tapost)] = 0
            tapost = tapost / tapost.sum(axis=2)[..., np.newaxis]
            tapost[np.isnan(tapost)] = 0
            gamma = tapost * forward[:-1, :, np.newaxis]
            psi = forward
            self.p = (1 / psi.shape[0]) * card * (psi.reshape((psi.shape[0], self.nbc_x, self.nbc_u))).sum(axis=(0, 1))
            self.t = np.transpose(np.transpose((np.outer(card, card) *
                                                gamma.reshape((gamma.shape[0], self.nbc_x, self.nbc_u, self.nbc_x,
                                                               self.nbc_u)).sum(
                                                    axis=(0, 1, 3)))) / (
                                              (card) * (psi.reshape((psi.shape[0], self.nbc_x, self.nbc_u))).sum(
                                          axis=(0, 1))))

            psi = (psi.reshape((psi.shape[0], self.nbc_x, self.nbc_u))).sum(axis=2)
            self.mu = (((psi * data).sum(axis=0)) / (psi.sum(axis=0))).reshape(self.mu.shape)
            broadc = (len(self.mu.shape) - len(data.shape) + 1)
            self.sigma = (psi.reshape(((psi.shape[0],) + self.sigma.shape)) * np.einsum('...i,...j',
                                                                                        (data.reshape(
                                                                                            (data.shape[0],) + (
                                                                                                1,) * broadc + data.shape[
                                                                                                               1:]) - self.mu),
                                                                                        (data.reshape(
                                                                                            (data.shape[0],) + (
                                                                                                1,) * broadc + data.shape[
                                                                                                               1:]) - self.mu))).sum(
                axis=0) / ((psi.sum(axis=0)).reshape(self.sigma.shape))

            nb_param = np.prod(self.p.shape) + np.prod(self.t.shape) + np.prod(self.mu.shape) + np.prod(
                self.sigma.shape)
            norm_param = (1 / nb_param) * np.sqrt(
                ((self.p - prev_p) ** 2).sum() + ((self.t - prev_t) ** 2).sum() + ((self.mu - prev_mu) ** 2).sum() + (
                        (self.sigma - prev_sigma) ** 2).sum())
            print({'iter': q + 1, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma,
                   'diff_norm_param': norm_param})
            if norm_param < early_stopping:
                break

    def get_param_ICE(self, data, iter, Nb_simul, early_stopping=0):
        print({'iter': 0, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma})
        for q in range(iter):
            prev_p = self.p
            prev_t = self.t
            prev_mu = self.mu
            prev_sigma = self.sigma

            T = calc_transDS(self.t, self.lx)
            T[np.isnan(T)] = 0
            mu = np.repeat(self.mu, self.nbc_u, axis=0)
            sigma = np.repeat(self.sigma, self.nbc_u, axis=0)
            card = 1 / np.sum(self.lx, axis=0)
            card[card == np.inf] = 0
            gaussians = np_multivariate_normal_pdf(data, mu, sigma)
            forward, backward = self.get_forward_backward(gaussians)
            hidden = np.stack([self.simul_hidden_apost(backward, gaussians) for n in range(Nb_simul)], axis=0)
            tapost = (
                    (backward[1:, np.newaxis, :] * gaussians[1:, np.newaxis, :]
                     * T[np.newaxis, :, :]) / backward[:-1, :, np.newaxis])
            tapost[np.isnan(tapost)] = 0
            tapost = tapost / tapost.sum(axis=2)[..., np.newaxis]
            tapost[np.isnan(tapost)] = 0
            gamma = tapost * forward[:-1, :, np.newaxis]
            psi = forward
            self.p = (1 / psi.shape[0]) * card * (psi.reshape((psi.shape[0], self.nbc_x, self.nbc_u))).sum(axis=(0, 1))
            self.t = np.transpose(np.transpose((np.outer(card, card) *
                                                gamma.reshape((gamma.shape[0], self.nbc_x, self.nbc_u, self.nbc_x,
                                                               self.nbc_u)).sum(
                                                    axis=(0, 1, 3)))) / (
                                          (card) * (psi.reshape((psi.shape[0], self.nbc_x, self.nbc_u))).sum(
                                      axis=(0, 1))))

            broadc = (len(self.mu.shape) - len(data.shape) + 1)
            self.mu = (
                    (((hidden[..., np.newaxis] // self.nbc_u) == np.indices((self.mu.shape[0],))) * data).sum(
                        axis=(0, 1)) /
                    ((hidden[..., np.newaxis] // self.nbc_u) == np.indices((self.mu.shape[0],))).sum(
                        axis=(0, 1))).reshape(self.mu.shape)
            self.sigma = (
                    (((hidden[..., np.newaxis] // self.nbc_u) == np.indices((self.mu.shape[0],))).reshape(
                        ((hidden.shape[0], hidden.shape[1]) + self.sigma.shape)) * np.einsum('...i,...j',
                                                                                             (data.reshape(
                                                                                                 (data.shape[0],) + (
                                                                                                     1,) * broadc + data.shape[
                                                                                                                    1:]) - self.mu),
                                                                                             (data.reshape(
                                                                                                 (data.shape[0],) + (
                                                                                                     1,) * broadc + data.shape[
                                                                                                                    1:]) - self.mu))).sum(
                        axis=(0, 1))
                    /
                    (((hidden[..., np.newaxis] // self.nbc_u) == np.indices((self.mu.shape[0],))).sum(
                        axis=(0, 1))).reshape(self.sigma.shape))

            nb_param = np.prod(self.p.shape) + np.prod(self.t.shape) + np.prod(self.mu.shape) + np.prod(
                self.sigma.shape)
            norm_param = (1 / nb_param) * np.sqrt(
                ((self.p - prev_p) ** 2).sum() + ((self.t - prev_t) ** 2).sum() + ((self.mu - prev_mu) ** 2).sum() + (
                        (self.sigma - prev_sigma) ** 2).sum())
            print({'iter': q + 1, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma,
                   'diff_norm_param': norm_param})
            if norm_param < early_stopping:
                break

    def get_param_SEM(self, data, iter, early_stopping=0):
        print({'iter': 0, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma})
        for q in range(iter):
            prev_p = self.p
            prev_t = self.t
            prev_mu = self.mu
            prev_sigma = self.sigma

            T = calc_transDS((self.t.T / self.p).T, self.lx)
            T[np.isnan(T)] = 0
            mu = np.repeat(self.mu, self.nbc_u, axis=0)
            sigma = np.repeat(self.sigma, self.nbc_u, axis=0)
            card = 1 / np.sum(self.lx, axis=0)
            gaussians = np_multivariate_normal_pdf(data, mu, sigma)
            forward, backward = self.get_forward_backward(gaussians)
            hidden = self.simul_hidden_apost(backward, gaussians)
            hiddenc = np.stack((hidden[:-1], hidden[1:]), axis=-1)
            aux = np.moveaxis(np.indices(T.shape), 0, -1)
            broadc = (len(aux.shape) - len(hiddenc.shape) + 1)
            self.p = (1 / hidden.shape[0]) * card * (hidden[..., np.newaxis] == np.indices((T.shape[0],))).reshape(
                (hidden.shape[0], self.nbc_x, self.nbc_u)).sum(axis=(0, 1))
            c = (1 / hiddenc.shape[0]) * (np.outer(card, card) *
                                          np.all(hiddenc.reshape(
                                              (hiddenc.shape[0],) + (1,) * broadc + hiddenc.shape[1:]) == aux,
                                                 axis=-1).reshape(
                                              (hiddenc.shape[0], self.nbc_x, self.nbc_u, self.nbc_x, self.nbc_u)).sum(
                                              axis=(0, 1, 3)))
            self.t = (c.T / self.p).T
            self.t[np.isnan(self.t)] = 0

            broadc = (len(self.mu.shape) - len(data.shape) + 1)
            self.mu = ((((hidden[..., np.newaxis] // self.nbc_u) == np.indices((self.mu.shape[0],))) * data).sum(
                axis=0) / (
                               (hidden[..., np.newaxis] // self.nbc_u) == np.indices((self.mu.shape[0],))).sum(
                axis=0)).reshape(self.mu.shape)

            self.sigma = ((((hidden[..., np.newaxis] // self.nbc_u) == np.indices((self.mu.shape[0],))).reshape(
                ((hidden.shape[0],) + self.sigma.shape)) * np.einsum('...i,...j',
                                                                     (data.reshape((data.shape[0],) + (
                                                                         1,) * broadc + data.shape[1:]) - self.mu),
                                                                     (data.reshape(
                                                                         (data.shape[0],) + (
                                                                             1,) * broadc + data.shape[
                                                                                            1:]) - self.mu))).sum(
                axis=0)
                          / (((hidden[..., np.newaxis] // self.nbc_u) == np.indices((self.mu.shape[0],))).sum(
                        axis=0)).reshape(self.sigma.shape))

            nb_param = np.prod(self.p.shape) + np.prod(self.t.shape) + np.prod(self.mu.shape) + np.prod(
                self.sigma.shape)
            norm_param = (1 / nb_param) * np.sqrt(
                ((self.p - prev_p) ** 2).sum() + ((self.t - prev_t) ** 2).sum() + ((self.mu - prev_mu) ** 2).sum() + (
                        (self.sigma - prev_sigma) ** 2).sum())
            print({'iter': q + 1, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma,
                   'diff_norm_param': norm_param})
            if norm_param < early_stopping:
                break

    def get_param_supervised(self, data, hidden, iter=100, early_stopping=0):
        broadc = (len(self.mu.shape) - len(data.shape) + 1)
        self.mu = (((hidden[..., np.newaxis] == np.indices((self.mu.shape[0],))) * data).sum(axis=0) / (
                hidden[..., np.newaxis] == np.indices((self.mu.shape[0],))).sum(axis=0)).reshape(self.mu.shape)
        self.sigma = (((hidden[..., np.newaxis] == np.indices((self.mu.shape[0],))).reshape(
            ((hidden.shape[0],) + self.sigma.shape)) * np.einsum('...i,...j',
                                                                 (data.reshape((data.shape[0],) + (
                                                                     1,) * broadc + data.shape[1:]) - self.mu),
                                                                 (data.reshape(
                                                                     (data.shape[0],) + (
                                                                         1,) * broadc + data.shape[
                                                                                        1:]) - self.mu))).sum(
            axis=0)
                      / ((hidden[..., np.newaxis] == np.indices((self.mu.shape[0],))).sum(
                    axis=0)).reshape(self.sigma.shape))
        print({'iter': 0, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma})
        for q in range(iter):
            prev_p = self.p
            prev_t = self.t
            prev_mu = self.mu
            prev_sigma = self.sigma

            T = calc_transDS(self.t, self.lx)
            T[np.isnan(T)] = 0
            mu = np.repeat(self.mu, self.nbc_u, axis=0)
            sigma = np.repeat(self.sigma, self.nbc_u, axis=0)
            card = 1 / np.sum(self.lx, axis=0)
            gaussians = np_multivariate_normal_pdf(data, mu, sigma)
            forward, backward = self.get_forward_backward_supervised(gaussians, hidden)
            Tprime = gaussians[1:, np.newaxis, :] * T[np.newaxis, :, :]
            Tprime = Tprime.reshape((Tprime.shape[0], self.nbc_x, self.nbc_u, self.nbc_x, self.nbc_u)).sum(axis=(1, 3))
            tapost = (
                    (backward[1:, np.newaxis, :] * self.lx[hidden][1:, np.newaxis, :]
                     * Tprime) / backward[:-1, :, np.newaxis])
            tapost[np.isnan(tapost)] = 0
            tapost = tapost / tapost.sum(axis=2)[..., np.newaxis]
            tapost[np.isnan(tapost)] = 0
            gamma = tapost * forward[:-1, :, np.newaxis]
            psi = forward
            self.p = (1 / psi.shape[0]) * card * psi.sum(axis=0)
            self.t = np.transpose(np.transpose((np.outer(card, card) *
                                                gamma.sum(axis=0))) / (
                                          (card) * psi.sum(axis=0)))
            self.t[np.isnan(self.t)] = 0
            nb_param = np.prod(self.p.shape) + np.prod(self.t.shape) + np.prod(self.mu.shape) + np.prod(
                self.sigma.shape)
            norm_param = (1 / nb_param) * np.sqrt(
                ((self.p - prev_p) ** 2).sum() + ((self.t - prev_t) ** 2).sum() + ((self.mu - prev_mu) ** 2).sum() + (
                        (self.sigma - prev_sigma) ** 2).sum())
            print({'iter': q + 1, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma,
                   'diff_norm_param': norm_param})
            if norm_param < early_stopping:
                break


class HSEMC_ctod:
    __slots__ = ('p', 't', 'lx', 'mu', 'sigma', 'nbc_x', 'nbc_u1', 'nbc_u2')

    def __init__(self, nbc_x, nbc_u, p=None, t=None, mu=None, sigma=None):
        self.p = p
        self.t = t
        self.mu = mu
        self.sigma = sigma
        self.nbc_x = nbc_x
        self.nbc_u1 = (2 ** nbc_x) - 1
        self.nbc_u2 = nbc_u

    def init_data_prior(self, data, scale=1):
        self.lx = np.repeat(np.stack([convertcls_vect(i + 1, (2,) * self.nbc_x) for i in range(self.nbc_u1)], axis=0),
                            self.nbc_u2,
                            axis=0).T
        lxprime = np.stack([convertcls_vect(i + 1, (2,) * self.nbc_x) for i in range(self.nbc_u1)], axis=0).T
        card = 1 / np.sum(lxprime, axis=0)
        nb_class = self.nbc_u1 * self.nbc_u2
        u2 = np.ones((self.nbc_u1, self.nbc_u1, self.nbc_u2)) * (1 / self.nbc_u2)
        pu1 = np.array([1 / self.nbc_u1] * self.nbc_u1)
        a = np.full((self.nbc_u1, self.nbc_u1), 1 / (2 * (self.nbc_u1 - 1)))
        a = a - np.diag(np.diag(a))
        tu1 = np.diag(np.array([1 / 2] * self.nbc_u1)) + a
        cu1 = (tu1.T / pu1).T
        u1 = card * cu1.sum(axis=1)
        a = ((np.outer(card, card) * cu1).T / u1).T
        self.p = (np.sum((a.T * u1).T * u2.T, axis=1)).T.flatten()

        b = np.repeat(np.eye(self.nbc_u1, self.nbc_u1, k=0), self.nbc_u2, axis=0)
        idx = [i for i in range(b.shape[0]) if ((i + 1) % self.nbc_u2 == 0)]
        for i, e in enumerate(idx):
            b[e] = a[i]

        a = u2
        ut = [[np.eye(self.nbc_u2, k=1) for n1 in range(self.nbc_u1)] for n2 in range(int(nb_class / self.nbc_u2))]
        for i, e in enumerate(ut):
            for j, p in enumerate(e):
                p[-1] = a[i, j]
        ut = np.block(ut)
        for i in range(1, self.nbc_u1 + 1):
            ut[:, (i - 1) * self.nbc_u2:i * self.nbc_u2] = (
                    ut[:, (i - 1) * self.nbc_u2:i * self.nbc_u2].T * b[:, i - 1]).T
        self.t = ut
        M = np.mean(data, axis=0)
        Sig = np.cov(data, rowvar=False).reshape(data.shape[1], data.shape[1])
        self.mu = [None] * self.nbc_x
        self.sigma = [None] * self.nbc_x
        for l in range(self.nbc_x):
            if l % 2 == 0:
                self.mu[l] = M - scale * ((l / (self.nbc_x / 2)) * np.sum(Sig, axis=1))
            else:
                self.mu[l] = M + scale * ((l / (self.nbc_x / 2)) * np.sum(Sig, axis=1))
            self.sigma[l] = Sig

        self.mu = np.array(self.mu)
        self.sigma = np.array(self.sigma)

    def init_kmeans(self, data, perturbation_param=0.1):
        self.lx = np.repeat(np.stack([convertcls_vect(i + 1, (2,) * self.nbc_x) for i in range(self.nbc_u1)], axis=0),
                            self.nbc_u2,
                            axis=0).T
        kmeans = KMeans(n_clusters=self.nbc_x).fit(data)
        hidden = kmeans.labels_
        hiddenc = np.stack((hidden[:-1], hidden[1:]), axis=-1)
        aux = np.moveaxis(np.indices((self.nbc_x, self.nbc_x)), 0, -1)
        broadc = (len(aux.shape) - len(hiddenc.shape) + 1)
        self.mu = np.zeros((self.nbc_x, 1 * len(data[0])))
        self.sigma = np.zeros((self.nbc_x, 1 * len(data[0]), 1 * len(data[0])))
        c = (1 / (len(data) - 1)) * (
            np.all(hiddenc.reshape((hiddenc.shape[0],) + (1,) * broadc + hiddenc.shape[1:]) == aux, axis=-1).sum(
                axis=0)).reshape((self.nbc_x, self.nbc_x))

        lxprime = np.stack([convertcls_vect(i + 1, (2,) * self.nbc_x) for i in range(self.nbc_u1)], axis=0).T
        card = 1 / np.sum(lxprime, axis=0)
        perturbation = np.random.uniform(0, 1, size=(self.nbc_u1 - self.nbc_x))
        perturbation = (perturbation / perturbation.sum()) * perturbation_param
        index1 = lxprime.T.sum(axis=1) == 1
        index2 = lxprime.T.sum(axis=1) != 1
        res = lxprime.T
        res[index1] = res[index1] * (1 - perturbation_param)
        res[index2] = (res[index2].T * perturbation).T
        u = res @ c @ res.T

        u1 = card * u.sum(axis=1)
        a = ((np.outer(card, card) * u).T / u1).T
        u = np.ones((self.nbc_u1, self.nbc_u1, self.nbc_u2)) * (1 / self.nbc_u2)
        self.p = (np.sum((a.T * u1).T * u.T, axis=1)).T.flatten()
        b = np.repeat(np.eye(self.nbc_u1, self.nbc_u1, k=0), self.nbc_u2, axis=0)
        idx = [i for i in range(b.shape[0]) if ((i + 1) % self.nbc_u2 == 0)]
        for i, e in enumerate(idx):
            b[e] = a[i]

        a = u
        ut = [[np.eye(self.nbc_u2, k=1) for n1 in range(self.nbc_u1)] for n2 in
              range(int((self.nbc_u1 * self.nbc_u2) / self.nbc_u2))]
        for i, e in enumerate(ut):
            for j, p in enumerate(e):
                p[-1] = a[i, j]
        ut = np.block(ut)
        for i in range(1, self.nbc_x + 1):
            ut[:, (i - 1) * self.nbc_u2:i * self.nbc_u2] = (
                    ut[:, (i - 1) * self.nbc_u2:i * self.nbc_u2].T * b[:, i - 1]).T
        self.t = ut

        broadc = (len(self.mu.shape) - len(data.shape) + 1)
        self.mu = (((hidden[..., np.newaxis] == np.indices((self.mu.shape[0],))) * data).sum(axis=0) / (
                hidden[..., np.newaxis] == np.indices((self.mu.shape[0],))).sum(axis=0)).reshape(self.mu.shape)
        self.sigma = (((hidden[..., np.newaxis] == np.indices((self.mu.shape[0],))).reshape(
            ((hidden.shape[0],) + self.sigma.shape)) * np.einsum('...i,...j',
                                                                 (data.reshape((data.shape[0],) + (
                                                                     1,) * broadc + data.shape[1:]) - self.mu),
                                                                 (data.reshape(
                                                                     (data.shape[0],) + (
                                                                         1,) * broadc + data.shape[
                                                                                        1:]) - self.mu))).sum(
            axis=0)
                      / ((hidden[..., np.newaxis] == np.indices((self.mu.shape[0],))).sum(
                    axis=0)).reshape(self.sigma.shape))

    def init_from_markov_chain(self, data, iter=100, early_stopping=10 ** -4, perturbation_param=0.1):
        self.lx = np.repeat(np.stack([convertcls_vect(i + 1, (2,) * self.nbc_x) for i in range(self.nbc_u1)], axis=0),
                            self.nbc_u2,
                            axis=0).T
        hmc = HMC_ctod(self.nbc_x)
        hmc.init_data_prior(data)
        hmc.get_param_EM(data, iter, early_stopping=early_stopping)
        c = (hmc.t.T * hmc.p).T
        lxprime = np.stack([convertcls_vect(i + 1, (2,) * self.nbc_x) for i in range(self.nbc_u1)], axis=0).T
        card = 1 / np.sum(lxprime, axis=0)
        perturbation = np.random.uniform(0, 1, size=(self.nbc_u1 - self.nbc_x))
        perturbation = (perturbation / perturbation.sum()) * perturbation_param
        index1 = lxprime.T.sum(axis=1) == 1
        index2 = lxprime.T.sum(axis=1) != 1
        res = lxprime.T
        res[index1] = res[index1] * (1 - perturbation_param)
        res[index2] = (res[index2].T * perturbation).T
        u1 = res @ c @ res.T

        u2 = np.ones((self.nbc_u1, self.nbc_u1, self.nbc_u2)) * (1 / self.nbc_u2)
        nb_class = self.nbc_u1 * self.nbc_u2
        pu1 = (card * np.sum(u1, axis=1))
        a = ((np.outer(card, card) * u1).T / pu1).T
        self.p = (np.sum((a.T * pu1).T * u2.T, axis=1)).T.flatten()
        b = np.repeat(np.eye(self.nbc_u1, self.nbc_u1, k=0), self.nbc_u2, axis=0)
        idx = [i for i in range(b.shape[0]) if ((i + 1) % self.nbc_u2 == 0)]
        for i, e in enumerate(idx):
            b[e] = a[i]

        a = u2
        ut = [[np.eye(self.nbc_u2, k=1) for n1 in range(self.nbc_u1)] for n2 in range(int(nb_class / self.nbc_u2))]
        for i, e in enumerate(ut):
            for j, p in enumerate(e):
                p[-1] = a[i, j]
        ut = np.block(ut)
        for i in range(1, self.nbc_u1 + 1):
            ut[:, (i - 1) * self.nbc_u2:i * self.nbc_u2] = (
                    ut[:, (i - 1) * self.nbc_u2:i * self.nbc_u2].T * b[:, i - 1]).T
        self.t = ut
        self.t[np.isnan(self.t)] = 0

        self.mu = hmc.mu
        self.sigma = hmc.sigma

    def give_param(self, u1, u2, mu, sigma):
        self.lx = np.repeat(np.stack([convertcls_vect(i + 1, (2,) * self.nbc_x) for i in range(self.nbc_u1)], axis=0),
                            self.nbc_u2,
                            axis=0).T
        lxprime = np.stack([convertcls_vect(i + 1, (2,) * self.nbc_x) for i in range(self.nbc_u1)], axis=0).T
        card = 1 / np.sum(lxprime, axis=0)
        nb_class = self.nbc_u1 * self.nbc_u2
        pu1 = (card * np.sum(u1, axis=1))
        a = ((np.outer(card, card) * u1).T / pu1).T
        self.p = (np.sum((a.T * pu1).T * u2.T, axis=1)).T.flatten()
        b = np.repeat(np.eye(self.nbc_u1, self.nbc_u1, k=0), self.nbc_u2, axis=0)
        idx = [i for i in range(b.shape[0]) if ((i + 1) % self.nbc_u2 == 0)]
        for i, e in enumerate(idx):
            b[e] = a[i]

        a = u2
        ut = [[np.eye(self.nbc_u2, k=1) for n1 in range(self.nbc_u1)] for n2 in range(int(nb_class / self.nbc_u2))]
        for i, e in enumerate(ut):
            for j, p in enumerate(e):
                p[-1] = a[i, j]
        ut = np.block(ut)
        for i in range(1, self.nbc_u1 + 1):
            ut[:, (i - 1) * self.nbc_u2:i * self.nbc_u2] = (
                    ut[:, (i - 1) * self.nbc_u2:i * self.nbc_u2].T * b[:, i - 1]).T
        self.t = ut
        self.t[np.isnan(self.t)] = 0
        self.mu = mu
        self.sigma = sigma

    def seg_map(self, data):
        pass

    def seg_mpm(self, data):
        mu = np.repeat(self.mu, self.nbc_u1 * self.nbc_u2, axis=0)
        sigma = np.repeat(self.sigma, self.nbc_u1 * self.nbc_u2, axis=0)
        gaussians = np_multivariate_normal_pdf(data, mu, sigma)
        forward, backward = self.get_forward_backward(gaussians)
        p_apost = forward
        p_apost_x = (p_apost.reshape((p_apost.shape[0], self.nbc_x, self.nbc_u1 * self.nbc_u2))).sum(axis=2)
        return np.argmax(p_apost_x, axis=1)

    def simul_hidden_apost(self, backward, gaussians, x_only=False):
        res = np.zeros(len(backward), dtype=int)
        T = calc_transDS(self.t, self.lx)
        aux = backward[0] / backward[0].sum()
        test = np.random.multinomial(1, aux)
        res[0] = np.argmax(test)
        tapost = (
                (backward[1:, np.newaxis, :] * gaussians[1:, np.newaxis, :]
                 * T[np.newaxis, :, :]) / backward[:-1, :, np.newaxis])
        tapost[np.isnan(tapost)] = 0
        tapost = tapost / tapost.sum(axis=2)[..., np.newaxis]
        tapost[np.isnan(tapost)] = 0
        for i in range(1, len(res)):
            test = np.random.multinomial(1, tapost[i - 1, res[i - 1], :])
            res[i] = np.argmax(test)
        if x_only:
            res = convert_multcls_vectors(res, (self.nbc_u1 * self.nbc_u2, self.nbc_x))[:, 1]
        return res

    def simul_visible(self, hidden):
        res = np.zeros((len(hidden), self.mu.shape[-1]))
        mu = np.repeat(self.mu, self.nbc_u1 * self.nbc_u2, axis=0)
        sigma = np.repeat(self.sigma, self.nbc_u1 * self.nbc_u2, axis=0)
        for i in range(0, len(hidden)):
            res[i] = multivariate_normal.rvs(mu[hidden[i]], sigma[hidden[i]])
        return np.array(res)

    def generate_sample(self, length, x_only=True):
        hidden = np.zeros(length, dtype=int)
        visible = np.zeros((length, self.mu.shape[-1]))
        backward = self.get_backward_apri(length)
        mu = np.repeat(self.mu, self.nbc_u1 * self.nbc_u2, axis=0)
        sigma = np.repeat(self.sigma, self.nbc_u1 * self.nbc_u2, axis=0)
        T = calc_transDS(self.t, self.lx)
        tapri = (
                (backward[1:, np.newaxis, :]
                 * T[np.newaxis, :, :]) / backward[:-1, :, np.newaxis]
        )
        tapri[np.isnan(tapri)] = 0
        tapri = tapri / tapri.sum(axis=2)[..., np.newaxis]
        tapri[np.isnan(tapri)] = 0
        test = np.random.multinomial(1, backward[0] / backward[0].sum())
        hidden[0] = np.argmax(test)
        visible[0] = multivariate_normal.rvs(mu[hidden[0]], sigma[hidden[0]])
        for i in range(1, length):
            test = np.random.multinomial(1, tapri[i - 1, hidden[i - 1], :])
            hidden[i] = np.argmax(test)
            visible[i] = multivariate_normal.rvs(mu[hidden[i]], sigma[hidden[i]])
        if x_only:
            hidden = convert_multcls_vectors(hidden, (self.nbc_u1 * self.nbc_u2, self.nbc_x))[:, 1]
        return hidden, visible

    def get_backward_apri(self, length):
        C = calc_matDS((self.t.T * self.p).T, self.lx)
        T = calc_transDS(self.t, self.lx)
        backward = np.zeros((length, T.shape[0]))
        backward[length - 1] = np.ones(T.shape[0])
        for l in reversed(range(0, length - 1)):
            if l == 0:
                phi = C
            else:
                phi = T
            backward[l] = phi @ (backward[l + 1])
            backward[l] = backward[l] / (backward[l].sum())

        return backward

    def get_gaussians(self, data):
        mu = np.repeat(self.mu, self.nbc_u1 * self.nbc_u2, axis=0)
        sigma = np.repeat(self.sigma, self.nbc_u1 * self.nbc_u2, axis=0)
        return np_multivariate_normal_pdf(data, mu, sigma)

    def get_forward_backward(self, gaussians):
        C = calc_matDS((self.t.T * self.p).T, self.lx)
        T = calc_transDS(self.t, self.lx)
        forward = np.zeros((len(gaussians), T.shape[0]))
        backward = np.zeros((len(gaussians), T.shape[0]))
        backward[len(gaussians) - 1] = np.ones(T.shape[0])
        for l in reversed(range(0, len(gaussians) - 1)):
            if l == 0:
                phi = ((C * gaussians[l + 1]).T * gaussians[l]).T
            else:
                phi = T * gaussians[l + 1]
            backward[l] = phi @ (backward[l + 1])
            backward[l] = backward[l] / (backward[l].sum())

        forward[0] = backward[0] / np.sum(backward[0])
        tapost = (
                (backward[1:, np.newaxis, :] * gaussians[1:, np.newaxis, :]
                 * T[np.newaxis, :, :]) / backward[:-1, :, np.newaxis])
        tapost[np.isnan(tapost)] = 0
        tapost = tapost / tapost.sum(axis=2)[..., np.newaxis]
        tapost[np.isnan(tapost)] = 0
        for k in range(1, len(gaussians)):
            forward[k] = (forward[k - 1] @ tapost[k - 1])

        return forward, backward

    def get_forward_backward_supervised(self, gaussians, hiddenx):
        forward = np.zeros((len(gaussians), self.nbc_u1*self.nbc_u2))
        backward = np.zeros((len(gaussians), self.nbc_u1*self.nbc_u2))
        backward[len(gaussians) - 1] = np.ones(self.nbc_u1*self.nbc_u2)
        C = calc_matDS((self.t.T * self.p).T, self.lx)
        T = calc_transDS(self.t, self.lx)
        for l in reversed(range(0, len(gaussians) - 1)):
            if l == 0:
                phi = ((((C * gaussians[l + 1]).T * gaussians[l]).T).reshape(
                    (self.nbc_x, self.nbc_u1*self.nbc_u2, self.nbc_x, self.nbc_u1*self.nbc_u2)).sum(axis=(0, 2)) * self.lx[hiddenx[l]])
            else:
                phi = (T * gaussians[l + 1]).reshape((self.nbc_x, self.nbc_u1*self.nbc_u2, self.nbc_x, self.nbc_u1*self.nbc_u2)).sum(
                    axis=(0, 2)) * self.lx[hiddenx[l]]
            backward[l] = phi @ (backward[l + 1])
            backward[l] = backward[l] / (backward[l].sum())

        forward[0] = backward[0] / np.sum(backward[0])
        Tprime = gaussians[1:, np.newaxis, :] * T[np.newaxis, :, :]
        Tprime = Tprime.reshape((Tprime.shape[0], self.nbc_x, self.nbc_u1*self.nbc_u2, self.nbc_x, self.nbc_u1*self.nbc_u2)).sum(axis=(1, 3))
        tapost = (
                (backward[1:, np.newaxis, :] * self.lx[hiddenx][1:, np.newaxis, :]
                 * Tprime) / backward[:-1, :, np.newaxis])
        tapost[np.isnan(tapost)] = 0
        tapost = tapost / tapost.sum(axis=2)[..., np.newaxis]
        tapost[np.isnan(tapost)] = 0
        for k in range(1, len(gaussians)):
            forward[k] = (forward[k - 1] @ tapost[k - 1])

        return forward, backward

    def get_param_EM(self, data, iter, early_stopping=0):
        print({'iter': 0, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma})
        for q in range(iter):
            prev_p = self.p
            prev_t = self.t
            prev_mu = self.mu
            prev_sigma = self.sigma

            T = calc_transDS(self.t, self.lx)
            T[np.isnan(T)] = 0
            mu = np.repeat(self.mu, self.nbc_u1 * self.nbc_u2, axis=0)
            sigma = np.repeat(self.sigma, self.nbc_u1 * self.nbc_u2, axis=0)
            card = 1 / np.sum(self.lx, axis=0)
            card[card == np.inf] = 0
            gaussians = np_multivariate_normal_pdf(data, mu, sigma)
            forward, backward = self.get_forward_backward(gaussians)
            tapost = (
                    (backward[1:, np.newaxis, :] * gaussians[1:, np.newaxis, :]
                     * T[np.newaxis, :, :]) / backward[:-1, :, np.newaxis])
            tapost[np.isnan(tapost)] = 0
            tapost = tapost / tapost.sum(axis=2)[..., np.newaxis]
            tapost[np.isnan(tapost)] = 0
            gamma = tapost * forward[:-1, :, np.newaxis]
            psi = forward
            self.p = (1 / psi.shape[0]) * card * (
                psi.reshape((psi.shape[0], self.nbc_x, self.nbc_u1 * self.nbc_u2))).sum(axis=(0, 1))
            self.t = np.transpose(np.transpose((np.outer(card, card) *
                                                gamma.reshape((gamma.shape[0], self.nbc_x, self.nbc_u1 * self.nbc_u2,
                                                               self.nbc_x, self.nbc_u1 * self.nbc_u2)).sum(
                                                    axis=(0, 1, 3)))) / (
                                          (card) * (
                                      psi.reshape((psi.shape[0], self.nbc_x, self.nbc_u1 * self.nbc_u2))).sum(
                                      axis=(0, 1))))

            psi = (psi.reshape((psi.shape[0], self.nbc_x, self.nbc_u1 * self.nbc_u2))).sum(axis=2)
            self.mu = (((psi * data).sum(axis=0)) / (psi.sum(axis=0))).reshape(self.mu.shape)
            broadc = (len(self.mu.shape) - len(data.shape) + 1)
            self.sigma = (psi.reshape(((psi.shape[0],) + self.sigma.shape)) * np.einsum('...i,...j',
                                                                                        (data.reshape(
                                                                                            (data.shape[0],) + (
                                                                                                1,) * broadc + data.shape[
                                                                                                               1:]) - self.mu),
                                                                                        (data.reshape(
                                                                                            (data.shape[0],) + (
                                                                                                1,) * broadc + data.shape[
                                                                                                               1:]) - self.mu))).sum(
                axis=0) / ((psi.sum(axis=0)).reshape(self.sigma.shape))

            nb_param = np.prod(self.p.shape) + np.prod(self.t.shape) + np.prod(self.mu.shape) + np.prod(
                self.sigma.shape)

            norm_param = (1 / nb_param) * np.sqrt(
                ((self.p - prev_p) ** 2).sum() + ((self.t - prev_t) ** 2).sum() + ((self.mu - prev_mu) ** 2).sum() + (
                        (self.sigma - prev_sigma) ** 2).sum())

            if np.isnan(norm_param):
                self.p = prev_p
                self.t = prev_t
                self.mu = prev_mu
                self.sigma = prev_sigma

            print({'iter': q + 1, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma,
                   'diff_norm_param': norm_param})

            if norm_param < early_stopping:
                break

    def get_param_ICE(self, data, iter, Nb_simul, early_stopping=0):
        print({'iter': 0, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma})
        for q in range(iter):
            prev_p = self.p
            prev_t = self.t
            prev_mu = self.mu
            prev_sigma = self.sigma

            T = calc_transDS(self.t, self.lx)
            T[np.isnan(T)] = 0
            mu = np.repeat(self.mu, self.nbc_u1 * self.nbc_u2, axis=0)
            sigma = np.repeat(self.sigma, self.nbc_u1 * self.nbc_u2, axis=0)
            card = 1 / np.sum(self.lx, axis=0)
            card[card == np.inf] = 0
            gaussians = np_multivariate_normal_pdf(data, mu, sigma)
            forward, backward = self.get_forward_backward(gaussians)
            hidden = np.stack([self.simul_hidden_apost(backward, gaussians) for n in range(Nb_simul)], axis=0)
            tapost = (
                    (backward[1:, np.newaxis, :] * gaussians[1:, np.newaxis, :]
                     * T[np.newaxis, :, :]) / backward[:-1, :, np.newaxis])
            tapost[np.isnan(tapost)] = 0
            tapost = tapost / tapost.sum(axis=2)[..., np.newaxis]
            tapost[np.isnan(tapost)] = 0
            gamma = tapost * forward[:-1, :, np.newaxis]
            psi = forward
            self.p = (1 / psi.shape[0]) * card * (
                psi.reshape((psi.shape[0], self.nbc_x, self.nbc_u1 * self.nbc_u2))).sum(axis=(0, 1))
            self.t = np.transpose(np.transpose((np.outer(card, card) *
                                                gamma.reshape((gamma.shape[0], self.nbc_x, self.nbc_u1 * self.nbc_u2,
                                                               self.nbc_x, self.nbc_u1 * self.nbc_u2)).sum(
                                                    axis=(0, 1, 3)))) / (
                                          (card) * (
                                      psi.reshape((psi.shape[0], self.nbc_x, self.nbc_u1 * self.nbc_u2))).sum(
                                      axis=(0, 1))))

            broadc = (len(self.mu.shape) - len(data.shape) + 1)
            self.mu = (
                    (((hidden[..., np.newaxis] // (self.nbc_u1 * self.nbc_u2)) == np.indices(
                        (self.mu.shape[0],))) * data).sum(axis=(0, 1)) /
                    ((hidden[..., np.newaxis] // (self.nbc_u1 * self.nbc_u2)) == np.indices((self.mu.shape[0],))).sum(
                        axis=(0, 1))).reshape(self.mu.shape)
            self.sigma = (
                    (((hidden[..., np.newaxis] // (self.nbc_u1 * self.nbc_u2)) == np.indices(
                        (self.mu.shape[0],))).reshape(
                        ((hidden.shape[0], hidden.shape[1]) + self.sigma.shape)) * np.einsum('...i,...j',
                                                                                             (data.reshape(
                                                                                                 (data.shape[0],) + (
                                                                                                     1,) * broadc + data.shape[
                                                                                                                    1:]) - self.mu),
                                                                                             (data.reshape(
                                                                                                 (data.shape[0],) + (
                                                                                                     1,) * broadc + data.shape[
                                                                                                                    1:]) - self.mu))).sum(
                        axis=(0, 1))
                    /
                    (((hidden[..., np.newaxis] // (self.nbc_u1 * self.nbc_u2)) == np.indices((self.mu.shape[0],))).sum(
                        axis=(0, 1))).reshape(self.sigma.shape))

            nb_param = np.prod(self.p.shape) + np.prod(self.t.shape) + np.prod(self.mu.shape) + np.prod(
                self.sigma.shape)
            norm_param = (1 / nb_param) * np.sqrt(
                ((self.p - prev_p) ** 2).sum() + ((self.t - prev_t) ** 2).sum() + ((self.mu - prev_mu) ** 2).sum() + (
                        (self.sigma - prev_sigma) ** 2).sum())
            print({'iter': q + 1, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma,
                   'diff_norm_param': norm_param})
            if norm_param < early_stopping:
                break

    def get_param_SEM(self, data, iter, early_stopping=0):
        print({'iter': 0, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma})
        for q in range(iter):
            prev_p = self.p
            prev_t = self.t
            prev_mu = self.mu
            prev_sigma = self.sigma

            T = calc_transDS((self.t.T / self.p).T, self.lx)
            T[np.isnan(T)] = 0
            mu = np.repeat(self.mu, self.nbc_u1 * self.nbc_u2, axis=0)
            sigma = np.repeat(self.sigma, self.nbc_u1 * self.nbc_u2, axis=0)
            card = 1 / np.sum(self.lx, axis=0)
            gaussians = np_multivariate_normal_pdf(data, mu, sigma)
            forward, backward = self.get_forward_backward(gaussians)
            hidden = self.simul_hidden_apost(backward, gaussians)
            hiddenc = np.stack((hidden[:-1], hidden[1:]), axis=-1)
            aux = np.moveaxis(np.indices(T.shape), 0, -1)
            broadc = (len(aux.shape) - len(hiddenc.shape) + 1)
            self.p = (1 / hidden.shape[0]) * card * (hidden[..., np.newaxis] == np.indices((T.shape[0],))).reshape(
                (hidden.shape[0], self.nbc_x, self.nbc_u1 * self.nbc_u2)).sum(axis=(0, 1))
            c = (1 / hiddenc.shape[0]) * (np.outer(card, card) *
                                          np.all(hiddenc.reshape(
                                              (hiddenc.shape[0],) + (1,) * broadc + hiddenc.shape[1:]) == aux,
                                                 axis=-1).reshape(
                                              (hiddenc.shape[0], self.nbc_x, self.nbc_u1 * self.nbc_u2, self.nbc_x,
                                               self.nbc_u1 * self.nbc_u2)).sum(
                                              axis=(0, 1, 3)))
            self.t = (c.T / self.p).T
            self.t[np.isnan(self.t)] = 0

            broadc = (len(self.mu.shape) - len(data.shape) + 1)
            self.mu = ((((hidden[..., np.newaxis] // (self.nbc_u1 * self.nbc_u2)) == np.indices(
                (self.mu.shape[0],))) * data).sum(
                axis=0) / (
                               (hidden[..., np.newaxis] // (self.nbc_u1 * self.nbc_u2)) == np.indices(
                           (self.mu.shape[0],))).sum(
                axis=0)).reshape(self.mu.shape)

            self.sigma = ((((hidden[..., np.newaxis] // (self.nbc_u1 * self.nbc_u2)) == np.indices(
                (self.mu.shape[0],))).reshape(
                ((hidden.shape[0],) + self.sigma.shape)) * np.einsum('...i,...j',
                                                                     (data.reshape((data.shape[0],) + (
                                                                         1,) * broadc + data.shape[1:]) - self.mu),
                                                                     (data.reshape(
                                                                         (data.shape[0],) + (
                                                                             1,) * broadc + data.shape[
                                                                                            1:]) - self.mu))).sum(
                axis=0)
                          / (((hidden[..., np.newaxis] // (self.nbc_u1 * self.nbc_u2)) == np.indices(
                        (self.mu.shape[0],))).sum(
                        axis=0)).reshape(self.sigma.shape))

            nb_param = np.prod(self.p.shape) + np.prod(self.t.shape) + np.prod(self.mu.shape) + np.prod(
                self.sigma.shape)
            norm_param = (1 / nb_param) * np.sqrt(
                ((self.p - prev_p) ** 2).sum() + ((self.t - prev_t) ** 2).sum() + ((self.mu - prev_mu) ** 2).sum() + (
                        (self.sigma - prev_sigma) ** 2).sum())
            print({'iter': q + 1, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma,
                   'diff_norm_param': norm_param})
            if norm_param < early_stopping:
                break

    def get_param_supervised(self, data, hidden, iter=100, early_stopping=0):
        broadc = (len(self.mu.shape) - len(data.shape) + 1)
        self.mu = (((hidden[..., np.newaxis] == np.indices((self.mu.shape[0],))) * data).sum(axis=0) / (
                hidden[..., np.newaxis] == np.indices((self.mu.shape[0],))).sum(axis=0)).reshape(self.mu.shape)
        self.sigma = (((hidden[..., np.newaxis] == np.indices((self.mu.shape[0],))).reshape(
            ((hidden.shape[0],) + self.sigma.shape)) * np.einsum('...i,...j',
                                                                 (data.reshape((data.shape[0],) + (
                                                                     1,) * broadc + data.shape[1:]) - self.mu),
                                                                 (data.reshape(
                                                                     (data.shape[0],) + (
                                                                         1,) * broadc + data.shape[
                                                                                        1:]) - self.mu))).sum(
            axis=0)
                      / ((hidden[..., np.newaxis] == np.indices((self.mu.shape[0],))).sum(
                    axis=0)).reshape(self.sigma.shape))
        print({'iter': 0, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma})
        for q in range(iter):
            prev_p = self.p
            prev_t = self.t
            prev_mu = self.mu
            prev_sigma = self.sigma

            T = calc_transDS(self.t, self.lx)
            T[np.isnan(T)] = 0
            mu = np.repeat(self.mu, self.nbc_u1*self.nbc_u2, axis=0)
            sigma = np.repeat(self.sigma, self.nbc_u1*self.nbc_u2, axis=0)
            card = 1 / np.sum(self.lx, axis=0)
            gaussians = np_multivariate_normal_pdf(data, mu, sigma)
            forward, backward = self.get_forward_backward_supervised(gaussians, hidden)
            Tprime = gaussians[1:, np.newaxis, :] * T[np.newaxis, :, :]
            Tprime = Tprime.reshape((Tprime.shape[0], self.nbc_x, self.nbc_u1*self.nbc_u2, self.nbc_x, self.nbc_u1*self.nbc_u2)).sum(axis=(1, 3))
            tapost = (
                    (backward[1:, np.newaxis, :] * self.lx[hidden][1:, np.newaxis, :]
                     * Tprime) / backward[:-1, :, np.newaxis])
            tapost[np.isnan(tapost)] = 0
            tapost = tapost / tapost.sum(axis=2)[..., np.newaxis]
            tapost[np.isnan(tapost)] = 0
            gamma = tapost * forward[:-1, :, np.newaxis]
            psi = forward
            self.p = (1 / psi.shape[0]) * card * psi.sum(axis=0)
            self.t = np.transpose(np.transpose((np.outer(card, card) *
                                                gamma.sum(axis=0))) / (
                                          (card) * psi.sum(axis=0)))
            self.t[np.isnan(self.t)]=0

            nb_param = np.prod(self.p.shape) + np.prod(self.t.shape) + np.prod(self.mu.shape) + np.prod(
                self.sigma.shape)
            norm_param = (1 / nb_param) * np.sqrt(
                ((self.p - prev_p) ** 2).sum() + ((self.t - prev_t) ** 2).sum() + ((self.mu - prev_mu) ** 2).sum() + (
                        (self.sigma - prev_sigma) ** 2).sum())
            print({'iter': q + 1, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma,
                   'diff_norm_param': norm_param})
            if norm_param < early_stopping:
                break


class HESMC_ctod:
    __slots__ = ('p', 't', 'lx', 'mu', 'sigma', 'nbc_x', 'nbc_u1', 'nbc_u2')

    def __init__(self, nbc_x, nbc_u, p=None, t=None, mu=None, sigma=None):
        self.p = p
        self.t = t
        self.mu = mu
        self.sigma = sigma
        self.nbc_x = nbc_x
        self.nbc_u1 = nbc_u
        self.nbc_u2 = self.nbc_x * self.nbc_u1 + 1

    def init_data_prior(self, data, scale=1, perturbation_param=0.5):
        self.lx = np.vstack((np.eye(self.nbc_x * self.nbc_u1), np.ones((self.nbc_x * self.nbc_u1,)))).T
        card = 1 / np.sum(self.lx, axis=0)

        u1 = np.ones((self.nbc_x, self.nbc_x, self.nbc_u1)) * (1 / self.nbc_u1)
        a = np.full((self.nbc_x, self.nbc_x), 1 / (2 * (self.nbc_x - 1)))
        a = a - np.diag(np.diag(a))
        x = np.array([1 / self.nbc_x] * self.nbc_x)
        a = np.diag(np.array([1 / 2] * self.nbc_x)) + a
        p1 = (np.sum((a.T * x).T * u1.T, axis=1)).T.flatten()
        b = np.repeat(np.eye(self.nbc_x, self.nbc_x, k=0), self.nbc_u1, axis=0)
        idx = [i for i in range(b.shape[0]) if ((i + 1) % self.nbc_u1 == 0)]
        for i, e in enumerate(idx):
            b[e] = a[i]
        a = u1
        ut = [[np.eye(self.nbc_u1, k=1) for n1 in range(self.nbc_x)] for n2 in
              range(int((self.nbc_x * self.nbc_u1) / self.nbc_u1))]
        for i, e in enumerate(ut):
            for j, p in enumerate(e):
                p[-1] = a[i, j]
        ut = np.block(ut)
        for i in range(1, self.nbc_x + 1):
            ut[:, (i - 1) * self.nbc_u1:i * self.nbc_u1] = (
                    ut[:, (i - 1) * self.nbc_u1:i * self.nbc_u1].T * b[:, i - 1]).T
        t1 = ut

        c = (t1.T * p1).T

        index1 = self.lx.T.sum(axis=1) == 1
        index2 = self.lx.T.sum(axis=1) != 1
        res = np.copy(self.lx.T)
        res[index1] = res[index1] * (1 - perturbation_param)
        res[index2] = (res[index2].T * perturbation_param).T
        u2 = res @ c @ res.T
        self.p = card * u2.sum(axis=1)
        self.t = ((np.outer(card, card) * u2).T / self.p).T

        M = np.mean(data, axis=0)
        Sig = np.cov(data, rowvar=False).reshape(data.shape[1], data.shape[1])
        self.mu = [None] * self.nbc_x
        self.sigma = [None] * self.nbc_x
        for l in range(self.nbc_x):
            if l % 2 == 0:
                self.mu[l] = M - scale * ((l / (self.nbc_x / 2)) * np.sum(Sig, axis=1))
            else:
                self.mu[l] = M + scale * ((l / (self.nbc_x / 2)) * np.sum(Sig, axis=1))
            self.sigma[l] = Sig

        self.mu = np.array(self.mu)
        self.sigma = np.array(self.sigma)

    def init_kmeans(self, data, perturbation_param=0.5):
        self.lx = np.vstack((np.eye(self.nbc_x * self.nbc_u1), np.ones((self.nbc_x * self.nbc_u1,)))).T
        card = 1 / np.sum(self.lx, axis=0)
        u1 = np.ones((self.nbc_x, self.nbc_x, self.nbc_u1)) * (1 / self.nbc_u1)

        kmeans = KMeans(n_clusters=self.nbc_x).fit(data)
        hidden = kmeans.labels_
        hiddenc = np.stack((hidden[:-1], hidden[1:]), axis=-1)
        aux = np.moveaxis(np.indices((self.nbc_x, self.nbc_x)), 0, -1)
        broadc = (len(aux.shape) - len(hiddenc.shape) + 1)
        self.mu = np.zeros((self.nbc_x, 1 * len(data[0])))
        self.sigma = np.zeros((self.nbc_x, 1 * len(data[0]), 1 * len(data[0])))
        c = (1 / (len(data) - 1)) * (
            np.all(hiddenc.reshape((hiddenc.shape[0],) + (1,) * broadc + hiddenc.shape[1:]) == aux, axis=-1).sum(
                axis=0)).reshape((self.nbc_x, self.nbc_x))
        x = c.sum(axis=1)
        a = (c.T / x).T
        p1 = (np.sum((a.T * x).T * u1.T, axis=1)).T.flatten()
        b = np.repeat(np.eye(self.nbc_x, self.nbc_x, k=0), self.nbc_u1, axis=0)
        idx = [i for i in range(b.shape[0]) if ((i + 1) % self.nbc_u1 == 0)]
        for i, e in enumerate(idx):
            b[e] = a[i]
        a = u1
        ut = [[np.eye(self.nbc_u1, k=1) for n1 in range(self.nbc_x)] for n2 in
              range(int((self.nbc_x * self.nbc_u1) / self.nbc_u1))]
        for i, e in enumerate(ut):
            for j, p in enumerate(e):
                p[-1] = a[i, j]
        ut = np.block(ut)
        for i in range(1, self.nbc_x + 1):
            ut[:, (i - 1) * self.nbc_u1:i * self.nbc_u1] = (
                    ut[:, (i - 1) * self.nbc_u1:i * self.nbc_u1].T * b[:, i - 1]).T
        t1 = ut

        c = (t1.T * p1).T

        index1 = self.lx.T.sum(axis=1) == 1
        index2 = self.lx.T.sum(axis=1) != 1
        res = np.copy(self.lx.T)
        res[index1] = res[index1] * (1 - perturbation_param)
        res[index2] = (res[index2].T * perturbation_param).T
        u2 = res @ c @ res.T
        self.p = card * u2.sum(axis=1)
        self.t = ((np.outer(card, card) * u2).T / self.p).T

        broadc = (len(self.mu.shape) - len(data.shape) + 1)
        self.mu = (((hidden[..., np.newaxis] == np.indices((self.mu.shape[0],))) * data).sum(axis=0) / (
                hidden[..., np.newaxis] == np.indices((self.mu.shape[0],))).sum(axis=0)).reshape(self.mu.shape)
        self.sigma = (((hidden[..., np.newaxis] == np.indices((self.mu.shape[0],))).reshape(
            ((hidden.shape[0],) + self.sigma.shape)) * np.einsum('...i,...j',
                                                                 (data.reshape((data.shape[0],) + (
                                                                     1,) * broadc + data.shape[1:]) - self.mu),
                                                                 (data.reshape(
                                                                     (data.shape[0],) + (
                                                                         1,) * broadc + data.shape[
                                                                                        1:]) - self.mu))).sum(
            axis=0)
                      / ((hidden[..., np.newaxis] == np.indices((self.mu.shape[0],))).sum(
                    axis=0)).reshape(self.sigma.shape))

    def init_from_markov_chain(self, data, iter=100, early_stopping=10 ** -4, perturbation_param=0.1):
        self.lx = np.vstack((np.eye(self.nbc_x * self.nbc_u1), np.ones((self.nbc_x * self.nbc_u1,)))).T
        card = 1 / np.sum(self.lx, axis=0)
        u1 = np.ones((self.nbc_x, self.nbc_x, self.nbc_u1)) * (1 / self.nbc_u1)
        hmc = HMC_ctod(self.nbc_x)
        hmc.init_data_prior(data)
        hmc.get_param_EM(data, iter, early_stopping=early_stopping)

        x = hmc.p
        a = hmc.t
        p1 = (np.sum((a.T * x).T * u1.T, axis=1)).T.flatten()
        b = np.repeat(np.eye(self.nbc_x, self.nbc_x, k=0), self.nbc_u1, axis=0)
        idx = [i for i in range(b.shape[0]) if ((i + 1) % self.nbc_u1 == 0)]
        for i, e in enumerate(idx):
            b[e] = a[i]
        a = u1
        ut = [[np.eye(self.nbc_u1, k=1) for n1 in range(self.nbc_x)] for n2 in
              range(int((self.nbc_x * self.nbc_u1) / self.nbc_u1))]
        for i, e in enumerate(ut):
            for j, p in enumerate(e):
                p[-1] = a[i, j]
        ut = np.block(ut)
        for i in range(1, self.nbc_x + 1):
            ut[:, (i - 1) * self.nbc_u1:i * self.nbc_u1] = (
                    ut[:, (i - 1) * self.nbc_u1:i * self.nbc_u1].T * b[:, i - 1]).T
        t1 = ut

        c = (t1.T * p1).T

        index1 = self.lx.T.sum(axis=1) == 1
        index2 = self.lx.T.sum(axis=1) != 1
        res = np.copy(self.lx.T)
        res[index1] = res[index1] * (1 - perturbation_param)
        res[index2] = (res[index2].T * perturbation_param).T
        u2 = res @ c @ res.T
        self.p = card * u2.sum(axis=1)
        self.t = ((np.outer(card, card) * u2).T / self.p).T

        self.mu = hmc.mu
        self.sigma = hmc.sigma

    def give_param(self, c, u1, perturbation_param, mu, sigma, ):
        self.lx = np.vstack((np.eye(self.nbc_x * self.nbc_u1), np.ones((self.nbc_x * self.nbc_u1,)))).T
        card = 1 / np.sum(self.lx, axis=0)
        p1 = (np.sum(c * u1.T, axis=1)).T.flatten()
        a = (c.T / np.sum(c, axis=1)).T
        b = np.repeat(np.eye(self.nbc_x, self.nbc_x, k=0), self.nbc_u1, axis=0)
        idx = [i for i in range(b.shape[0]) if ((i + 1) % self.nbc_u1 == 0)]
        for i, e in enumerate(idx):
            b[e] = a[i]

        a = u1
        ut = [[np.eye(self.nbc_u1, k=1) for n1 in range(self.nbc_x)] for n2 in
              range(int((self.nbc_x * self.nbc_u1) / self.nbc_u1))]
        for i, e in enumerate(ut):
            for j, p in enumerate(e):
                p[-1] = a[i, j]
        ut = np.block(ut)
        for i in range(1, self.nbc_x + 1):
            ut[:, (i - 1) * self.nbc_u1:i * self.nbc_u1] = (
                    ut[:, (i - 1) * self.nbc_u1:i * self.nbc_u1].T * b[:, i - 1]).T
        t1 = ut
        c = (t1.T * p1).T
        index1 = self.lx.T.sum(axis=1) == 1
        index2 = self.lx.T.sum(axis=1) != 1
        res = np.copy(self.lx.T)
        res[index1] = res[index1] * (1 - perturbation_param)
        res[index2] = (res[index2].T * perturbation_param).T
        u2 = res @ c @ res.T
        self.p = card * u2.sum(axis=1)
        self.t = ((np.outer(card, card) * u2).T / self.p).T

        self.mu = mu
        self.sigma = sigma

    def seg_map(self, data):
        pass

    def seg_mpm(self, data):
        mu = np.repeat(self.mu, self.nbc_u1 * self.nbc_u2, axis=0)
        sigma = np.repeat(self.sigma, self.nbc_u1 * self.nbc_u2, axis=0)
        gaussians = np_multivariate_normal_pdf(data, mu, sigma)
        forward, backward = self.get_forward_backward(gaussians)
        p_apost = forward
        p_apost_x = (p_apost.reshape((p_apost.shape[0], self.nbc_x, self.nbc_u1 * self.nbc_u2))).sum(axis=2)
        return np.argmax(p_apost_x, axis=1)

    def simul_hidden_apost(self, backward, gaussians, x_only=False):
        res = np.zeros(len(backward), dtype=int)
        T = calc_transDS(self.t, self.lx)
        aux = backward[0] / backward[0].sum()
        test = np.random.multinomial(1, aux)
        res[0] = np.argmax(test)
        tapost = (
                (backward[1:, np.newaxis, :] * gaussians[1:, np.newaxis, :]
                 * T[np.newaxis, :, :]) / backward[:-1, :, np.newaxis])
        tapost[np.isnan(tapost)] = 0
        tapost = tapost / tapost.sum(axis=2)[..., np.newaxis]
        tapost[np.isnan(tapost)] = 0
        for i in range(1, len(res)):
            test = np.random.multinomial(1, tapost[i - 1, res[i - 1], :])
            res[i] = np.argmax(test)
        if x_only:
            res = convert_multcls_vectors(res, (self.nbc_u1 * self.nbc_u2, self.nbc_x))[:, 1]
        return res

    def simul_visible(self, hidden):
        res = np.zeros((len(hidden), self.mu.shape[-1]))
        mu = np.repeat(self.mu, self.nbc_u1 * self.nbc_u2, axis=0)
        sigma = np.repeat(self.sigma, self.nbc_u1 * self.nbc_u2, axis=0)
        for i in range(0, len(hidden)):
            res[i] = multivariate_normal.rvs(mu[hidden[i]], sigma[hidden[i]])
        return np.array(res)

    def generate_sample(self, length, x_only=True):
        hidden = np.zeros(length, dtype=int)
        visible = np.zeros((length, self.mu.shape[-1]))
        backward = self.get_backward_apri(length)
        mu = np.repeat(self.mu, self.nbc_u1 * self.nbc_u2, axis=0)
        sigma = np.repeat(self.sigma, self.nbc_u1 * self.nbc_u2, axis=0)
        T = calc_transDS(self.t, self.lx)
        tapri = (
                (backward[1:, np.newaxis, :]
                 * T[np.newaxis, :, :]) / backward[:-1, :, np.newaxis]
        )
        tapri[np.isnan(tapri)] = 0
        tapri = tapri / tapri.sum(axis=2)[..., np.newaxis]
        tapri[np.isnan(tapri)] = 0
        test = np.random.multinomial(1, backward[0] / backward[0].sum())
        hidden[0] = np.argmax(test)
        visible[0] = multivariate_normal.rvs(mu[hidden[0]], sigma[hidden[0]])
        for i in range(1, length):
            test = np.random.multinomial(1, tapri[i - 1, hidden[i - 1], :])
            hidden[i] = np.argmax(test)
            visible[i] = multivariate_normal.rvs(mu[hidden[i]], sigma[hidden[i]])
        if x_only:
            hidden = convert_multcls_vectors(hidden, (self.nbc_u1 * self.nbc_u2, self.nbc_x))[:, 1]
        return hidden, visible

    def get_backward_apri(self, length):
        C = calc_matDS((self.t.T * self.p).T, self.lx)
        T = calc_transDS(self.t, self.lx)
        backward = np.zeros((length, T.shape[0]))
        backward[length - 1] = np.ones(T.shape[0])
        for l in reversed(range(0, length - 1)):
            if l == 0:
                phi = C
            else:
                phi = T
            backward[l] = phi @ (backward[l + 1])
            backward[l] = backward[l] / (backward[l].sum())

        return backward

    def get_gaussians(self, data):
        mu = np.repeat(self.mu, self.nbc_u1 * self.nbc_u2, axis=0)
        sigma = np.repeat(self.sigma, self.nbc_u1 * self.nbc_u2, axis=0)
        return np_multivariate_normal_pdf(data, mu, sigma)

    def get_forward_backward(self, gaussians):
        C = calc_matDS((self.t.T * self.p).T, self.lx)
        T = calc_transDS(self.t, self.lx)
        forward = np.zeros((len(gaussians), T.shape[0]))
        backward = np.zeros((len(gaussians), T.shape[0]))
        backward[len(gaussians) - 1] = np.ones(T.shape[0])
        for l in reversed(range(0, len(gaussians) - 1)):
            if l == 0:
                phi = ((C * gaussians[l + 1]).T * gaussians[l]).T
            else:
                phi = T * gaussians[l + 1]
            backward[l] = phi @ (backward[l + 1])
            backward[l] = backward[l] / (backward[l].sum())

        forward[0] = backward[0] / np.sum(backward[0])
        tapost = (
                (backward[1:, np.newaxis, :] * gaussians[1:, np.newaxis, :]
                 * T[np.newaxis, :, :]) / backward[:-1, :, np.newaxis])
        tapost[np.isnan(tapost)] = 0
        tapost = tapost / tapost.sum(axis=2)[..., np.newaxis]
        tapost[np.isnan(tapost)] = 0
        for k in range(1, len(gaussians)):
            forward[k] = (forward[k - 1] @ tapost[k - 1])

        return forward, backward

    def get_forward_backward_supervised(self, gaussians, aux):
        forward = np.zeros((len(gaussians), self.nbc_u2))
        backward = np.zeros((len(gaussians), self.nbc_u2))
        backward[len(gaussians) - 1] = np.ones(self.nbc_u2)
        C = calc_matDS((self.t.T * self.p).T, self.lx)
        T = calc_transDS(self.t, self.lx)

        for l in reversed(range(0, len(gaussians) - 1)):
            if l == 0:
                phi = (((((C * gaussians[l + 1]).T * gaussians[l]).T).reshape(
                    (self.nbc_x*self.nbc_u1, self.nbc_u2, self.nbc_x*self.nbc_u1, self.nbc_u2)).sum(axis=(0, 2))).T * aux[l]).T * aux[l+1]
            else:
                phi = (T * gaussians[l + 1]).reshape((self.nbc_x*self.nbc_u1, self.nbc_u2, self.nbc_x*self.nbc_u1, self.nbc_u2)).sum(
                    axis=(0, 2)) * aux[l+1]
            backward[l] = phi @ (backward[l + 1])
            backward[l] = backward[l] / (backward[l].sum())

        forward[0] = backward[0] / np.sum(backward[0])
        Tprime = gaussians[1:, np.newaxis, :] * T[np.newaxis, :, :]
        Tprime = Tprime.reshape((Tprime.shape[0], self.nbc_x*self.nbc_u1, self.nbc_u2, self.nbc_x*self.nbc_u1, self.nbc_u2)).sum(axis=(1, 3))
        tapost = (
                (backward[1:, np.newaxis, :] * aux[1:, np.newaxis, :]
                 * Tprime) / backward[:-1, :, np.newaxis])
        tapost[np.isnan(tapost)] = 0
        tapost = tapost / tapost.sum(axis=2)[..., np.newaxis]
        tapost[np.isnan(tapost)] = 0
        for k in range(1, len(gaussians)):
            forward[k] = (forward[k - 1] @ tapost[k - 1])

        return forward, backward

    def get_param_EM(self, data, iter, early_stopping=0):
        print({'iter': 0, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma})
        for q in range(iter):
            prev_p = self.p
            prev_t = self.t
            prev_mu = self.mu
            prev_sigma = self.sigma

            T = calc_transDS(self.t, self.lx)
            T[np.isnan(T)] = 0
            mu = np.repeat(self.mu, self.nbc_u1 * self.nbc_u2, axis=0)
            sigma = np.repeat(self.sigma, self.nbc_u1 * self.nbc_u2, axis=0)
            card = 1 / np.sum(self.lx, axis=0)
            card[card == np.inf] = 0
            gaussians = np_multivariate_normal_pdf(data, mu, sigma)
            forward, backward = self.get_forward_backward(gaussians)
            tapost = (
                    (backward[1:, np.newaxis, :] * gaussians[1:, np.newaxis, :]
                     * T[np.newaxis, :, :]) / backward[:-1, :, np.newaxis])
            tapost[np.isnan(tapost)] = 0
            tapost = tapost / tapost.sum(axis=2)[..., np.newaxis]
            tapost[np.isnan(tapost)] = 0
            gamma = tapost * forward[:-1, :, np.newaxis]
            psi = forward
            self.p = (1 / psi.shape[0]) * card * (
                psi.reshape((psi.shape[0], self.nbc_u1 * self.nbc_x, self.nbc_u2))).sum(axis=(0, 1))
            self.t = np.transpose(np.transpose((np.outer(card, card) *
                                                gamma.reshape((gamma.shape[0], self.nbc_u1 * self.nbc_x, self.nbc_u2,
                                                               self.nbc_u1 * self.nbc_x, self.nbc_u2)).sum(
                                                    axis=(0, 1, 3)))) / (
                                          (card) * (
                                      psi.reshape((psi.shape[0], self.nbc_u1 * self.nbc_x, self.nbc_u2))).sum(
                                      axis=(0, 1))))

            psi = (psi.reshape((psi.shape[0], self.nbc_x, self.nbc_u1 * self.nbc_u2))).sum(axis=2)
            self.mu = (((psi * data).sum(axis=0)) / (psi.sum(axis=0))).reshape(self.mu.shape)
            broadc = (len(self.mu.shape) - len(data.shape) + 1)
            self.sigma = (psi.reshape(((psi.shape[0],) + self.sigma.shape)) * np.einsum('...i,...j',
                                                                                        (data.reshape(
                                                                                            (data.shape[0],) + (
                                                                                                1,) * broadc + data.shape[
                                                                                                               1:]) - self.mu),
                                                                                        (data.reshape(
                                                                                            (data.shape[0],) + (
                                                                                                1,) * broadc + data.shape[
                                                                                                               1:]) - self.mu))).sum(
                axis=0) / ((psi.sum(axis=0)).reshape(self.sigma.shape))

            nb_param = np.prod(self.p.shape) + np.prod(self.t.shape) + np.prod(self.mu.shape) + np.prod(
                self.sigma.shape)

            norm_param = (1 / nb_param) * np.sqrt(
                ((self.p - prev_p) ** 2).sum() + ((self.t - prev_t) ** 2).sum() + ((self.mu - prev_mu) ** 2).sum() + (
                        (self.sigma - prev_sigma) ** 2).sum())

            if np.isnan(norm_param):
                self.p = prev_p
                self.t = prev_t
                self.mu = prev_mu
                self.sigma = prev_sigma

            print({'iter': q + 1, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma,
                   'diff_norm_param': norm_param})

            if norm_param < early_stopping:
                break

    def get_param_ICE(self, data, iter, Nb_simul, early_stopping=0):
        print({'iter': 0, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma})
        for q in range(iter):
            prev_p = self.p
            prev_t = self.t
            prev_mu = self.mu
            prev_sigma = self.sigma

            T = calc_transDS(self.t, self.lx)
            T[np.isnan(T)] = 0
            mu = np.repeat(self.mu, self.nbc_u1 * self.nbc_u2, axis=0)
            sigma = np.repeat(self.sigma, self.nbc_u1 * self.nbc_u2, axis=0)
            card = 1 / np.sum(self.lx, axis=0)
            card[card == np.inf] = 0
            gaussians = np_multivariate_normal_pdf(data, mu, sigma)
            forward, backward = self.get_forward_backward(gaussians)
            hidden = np.stack([self.simul_hidden_apost(backward, gaussians) for n in range(Nb_simul)], axis=0)
            tapost = (
                    (backward[1:, np.newaxis, :] * gaussians[1:, np.newaxis, :]
                     * T[np.newaxis, :, :]) / backward[:-1, :, np.newaxis])
            tapost[np.isnan(tapost)] = 0
            tapost = tapost / tapost.sum(axis=2)[..., np.newaxis]
            tapost[np.isnan(tapost)] = 0
            gamma = tapost * forward[:-1, :, np.newaxis]
            psi = forward
            self.p = (1 / psi.shape[0]) * card * (
                psi.reshape((psi.shape[0], self.nbc_u1 * self.nbc_x, self.nbc_u2))).sum(axis=(0, 1))
            self.t = np.transpose(np.transpose((np.outer(card, card) *
                                                gamma.reshape((gamma.shape[0], self.nbc_u1 * self.nbc_x, self.nbc_u2,
                                                               self.nbc_u1 * self.nbc_x, self.nbc_u2)).sum(
                                                    axis=(0, 1, 3)))) / (
                                          (card) * (
                                      psi.reshape((psi.shape[0], self.nbc_u1 * self.nbc_x, self.nbc_u2))).sum(
                                      axis=(0, 1))))

            broadc = (len(self.mu.shape) - len(data.shape) + 1)
            self.mu = (
                    (((hidden[..., np.newaxis] // (self.nbc_u1 * self.nbc_u2)) == np.indices(
                        (self.mu.shape[0],))) * data).sum(axis=(0, 1)) /
                    ((hidden[..., np.newaxis] // (self.nbc_u1 * self.nbc_u2)) == np.indices((self.mu.shape[0],))).sum(
                        axis=(0, 1))).reshape(self.mu.shape)
            self.sigma = (
                    (((hidden[..., np.newaxis] // (self.nbc_u1 * self.nbc_u2)) == np.indices(
                        (self.mu.shape[0],))).reshape(
                        ((hidden.shape[0], hidden.shape[1]) + self.sigma.shape)) * np.einsum('...i,...j',
                                                                                             (data.reshape(
                                                                                                 (data.shape[0],) + (
                                                                                                     1,) * broadc + data.shape[
                                                                                                                    1:]) - self.mu),
                                                                                             (data.reshape(
                                                                                                 (data.shape[0],) + (
                                                                                                     1,) * broadc + data.shape[
                                                                                                                    1:]) - self.mu))).sum(
                        axis=(0, 1))
                    /
                    (((hidden[..., np.newaxis] // (self.nbc_u1 * self.nbc_u2)) == np.indices((self.mu.shape[0],))).sum(
                        axis=(0, 1))).reshape(self.sigma.shape))

            nb_param = np.prod(self.p.shape) + np.prod(self.t.shape) + np.prod(self.mu.shape) + np.prod(
                self.sigma.shape)
            norm_param = (1 / nb_param) * np.sqrt(
                ((self.p - prev_p) ** 2).sum() + ((self.t - prev_t) ** 2).sum() + ((self.mu - prev_mu) ** 2).sum() + (
                        (self.sigma - prev_sigma) ** 2).sum())
            print({'iter': q + 1, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma,
                   'diff_norm_param': norm_param})
            if norm_param < early_stopping:
                break

    def get_param_SEM(self, data, iter, early_stopping=0):
        print({'iter': 0, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma})
        for q in range(iter):
            prev_p = self.p
            prev_t = self.t
            prev_mu = self.mu
            prev_sigma = self.sigma

            T = calc_transDS((self.t.T / self.p).T, self.lx)
            T[np.isnan(T)] = 0
            mu = np.repeat(self.mu, self.nbc_u1 * self.nbc_u2, axis=0)
            sigma = np.repeat(self.sigma, self.nbc_u1 * self.nbc_u2, axis=0)
            card = 1 / np.sum(self.lx, axis=0)
            gaussians = np_multivariate_normal_pdf(data, mu, sigma)
            forward, backward = self.get_forward_backward(gaussians)
            hidden = self.simul_hidden_apost(backward, gaussians)
            hiddenc = np.stack((hidden[:-1], hidden[1:]), axis=-1)
            aux = np.moveaxis(np.indices(T.shape), 0, -1)
            broadc = (len(aux.shape) - len(hiddenc.shape) + 1)
            self.p = (1 / hidden.shape[0]) * card * (hidden[..., np.newaxis] == np.indices((T.shape[0],))).reshape(
                (hidden.shape[0], self.nbc_u1 * self.nbc_x, self.nbc_u2)).sum(axis=(0, 1))
            c = (1 / hiddenc.shape[0]) * (np.outer(card, card) *
                                          np.all(hiddenc.reshape(
                                              (hiddenc.shape[0],) + (1,) * broadc + hiddenc.shape[1:]) == aux,
                                                 axis=-1).reshape(
                                              (hiddenc.shape[0], self.nbc_u1 * self.nbc_x, self.nbc_u2,
                                               self.nbc_u1 * self.nbc_x, self.nbc_u2)).sum(
                                              axis=(0, 1, 3)))
            self.t = (c.T / self.p).T
            self.t[np.isnan(self.t)] = 0

            broadc = (len(self.mu.shape) - len(data.shape) + 1)
            self.mu = ((((hidden[..., np.newaxis] // (self.nbc_u1 * self.nbc_u2)) == np.indices(
                (self.mu.shape[0],))) * data).sum(
                axis=0) / (
                               (hidden[..., np.newaxis] // (self.nbc_u1 * self.nbc_u2)) == np.indices(
                           (self.mu.shape[0],))).sum(
                axis=0)).reshape(self.mu.shape)

            self.sigma = ((((hidden[..., np.newaxis] // (self.nbc_u1 * self.nbc_u2)) == np.indices(
                (self.mu.shape[0],))).reshape(
                ((hidden.shape[0],) + self.sigma.shape)) * np.einsum('...i,...j',
                                                                     (data.reshape((data.shape[0],) + (
                                                                         1,) * broadc + data.shape[1:]) - self.mu),
                                                                     (data.reshape(
                                                                         (data.shape[0],) + (
                                                                             1,) * broadc + data.shape[
                                                                                            1:]) - self.mu))).sum(
                axis=0)
                          / (((hidden[..., np.newaxis] // (self.nbc_u1 * self.nbc_u2)) == np.indices(
                        (self.mu.shape[0],))).sum(
                        axis=0)).reshape(self.sigma.shape))

            nb_param = np.prod(self.p.shape) + np.prod(self.t.shape) + np.prod(self.mu.shape) + np.prod(
                self.sigma.shape)
            norm_param = (1 / nb_param) * np.sqrt(
                ((self.p - prev_p) ** 2).sum() + ((self.t - prev_t) ** 2).sum() + ((self.mu - prev_mu) ** 2).sum() + (
                        (self.sigma - prev_sigma) ** 2).sum())
            print({'iter': q + 1, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma,
                   'diff_norm_param': norm_param})
            if norm_param < early_stopping:
                break

    def get_param_supervised(self, data, hidden, iter=100, early_stopping=0):
        broadc = (len(self.mu.shape) - len(data.shape) + 1)
        self.mu = (((hidden[..., np.newaxis] == np.indices((self.mu.shape[0],))) * data).sum(axis=0) / (
                hidden[..., np.newaxis] == np.indices((self.mu.shape[0],))).sum(axis=0)).reshape(self.mu.shape)
        self.sigma = (((hidden[..., np.newaxis] == np.indices((self.mu.shape[0],))).reshape(
            ((hidden.shape[0],) + self.sigma.shape)) * np.einsum('...i,...j',
                                                                 (data.reshape((data.shape[0],) + (
                                                                     1,) * broadc + data.shape[1:]) - self.mu),
                                                                 (data.reshape(
                                                                     (data.shape[0],) + (
                                                                         1,) * broadc + data.shape[
                                                                                        1:]) - self.mu))).sum(
            axis=0)
                      / ((hidden[..., np.newaxis] == np.indices((self.mu.shape[0],))).sum(
                    axis=0)).reshape(self.sigma.shape))
        aux = calc_cacheDS(self.lx, hidden, self.nbc_u1)
        print({'iter': 0, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma})
        for q in range(iter):
            prev_p = self.p
            prev_t = self.t
            prev_mu = self.mu
            prev_sigma = self.sigma

            T = calc_transDS(self.t, self.lx)
            T[np.isnan(T)] = 0
            mu = np.repeat(self.mu, self.nbc_u1*self.nbc_u2, axis=0)
            sigma = np.repeat(self.sigma, self.nbc_u1*self.nbc_u2, axis=0)
            card = 1 / np.sum(self.lx, axis=0)
            gaussians = np_multivariate_normal_pdf(data, mu, sigma)
            forward, backward = self.get_forward_backward_supervised(gaussians, aux)
            Tprime = gaussians[1:, np.newaxis, :] * T[np.newaxis, :, :]
            Tprime = Tprime.reshape(
                (Tprime.shape[0], self.nbc_x * self.nbc_u1, self.nbc_u2, self.nbc_x * self.nbc_u1, self.nbc_u2)).sum(
                axis=(1, 3))
            tapost = (
                    (backward[1:, np.newaxis, :] * aux[1:, np.newaxis, :]
                     * Tprime) / backward[:-1, :, np.newaxis])
            tapost[np.isnan(tapost)] = 0
            tapost = tapost / tapost.sum(axis=2)[..., np.newaxis]
            tapost[np.isnan(tapost)] = 0
            gamma = tapost * forward[:-1, :, np.newaxis]
            psi = forward
            self.p = (1 / psi.shape[0]) * card * psi.sum(axis=0)
            self.t = np.transpose(np.transpose((np.outer(card, card) *
                                                gamma.sum(axis=0))) / (
                                          (card) * psi.sum(axis=0)))
            self.t[np.isnan(self.t)] = 0

            nb_param = np.prod(self.p.shape) + np.prod(self.t.shape) + np.prod(self.mu.shape) + np.prod(
                self.sigma.shape)
            norm_param = (1 / nb_param) * np.sqrt(
                ((self.p - prev_p) ** 2).sum() + ((self.t - prev_t) ** 2).sum() + ((self.mu - prev_mu) ** 2).sum() + (
                        (self.sigma - prev_sigma) ** 2).sum())
            print({'iter': q + 1, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma,
                   'diff_norm_param': norm_param})
            if norm_param < early_stopping:
                break


class HEMC2_ctod:
    __slots__ = ('p', 't', 'lx', 'mu', 'sigma', 'nbc_x', 'nbc_u')

    def __init__(self, nbc_x, p=None, t=None, mu=None, sigma=None):
        self.p = p
        self.t = t
        self.mu = mu
        self.sigma = sigma
        self.nbc_x = nbc_x
        self.nbc_u = (2 ** nbc_x) - 1

    def init_data_prior(self, data, scale=1):
        self.lx = np.stack([convertcls_vect(i + 1, (2,) * self.nbc_x) for i in range(self.nbc_u)], axis=0).T
        card = np.sum(self.lx, axis=0)
        a = np.full((self.nbc_u, self.nbc_u), 1 / (2 * (self.nbc_u - 1)))
        a = a - np.diag(np.diag(a))
        p = np.array([1 / self.nbc_u] * self.nbc_u)
        t = np.diag(np.array([1 / 2] * self.nbc_u)) + a
        u = (t.T * p).T
        self.p = (card * u).sum(axis=1)
        self.t = (u.T / self.p).T
        # a = np.full((self.nbc_x, self.nbc_x), 1 / (2 * (self.nbc_x - 1)))
        # a = a - np.diag(np.diag(a))
        # t = np.diag(np.array([1 / 2] * self.nbc_x)) + a
        # p = np.array([1 / self.nbc_x] * self.nbc_x)
        # c = (t.T * p).T
        # u = self.lx.T @ c @ self.lx
        # u = u / u.sum()
        # self.p = (card * u).sum(axis=1)
        # self.t = (u.T / self.p).T
        M = np.mean(data, axis=0)
        Sig = np.cov(data, rowvar=False).reshape(data.shape[1], data.shape[1])
        self.mu = [None] * self.nbc_x
        self.sigma = [None] * self.nbc_x
        for l in range(self.nbc_x):
            if l % 2 == 0:
                self.mu[l] = M - scale * ((l / (self.nbc_x / 2)) * np.sum(Sig, axis=1))
            else:
                self.mu[l] = M + scale * ((l / (self.nbc_x / 2)) * np.sum(Sig, axis=1))
            self.sigma[l] = Sig

        self.mu = np.array(self.mu)
        self.sigma = np.array(self.sigma)

    def give_param(self, u, mu, sigma):
        self.lx = np.stack([convertcls_vect(i + 1, (2,) * self.nbc_x) for i in range(self.nbc_u)], axis=0).T
        card = np.sum(self.lx, axis=0)
        self.p = (card * u).sum(axis=1)
        self.t = (u.T / self.p).T
        self.mu = mu
        self.sigma = sigma

    def seg_map(self, data):
        pass

    def seg_mpm(self, data):
        mu = np.repeat(self.mu, self.nbc_u, axis=0)
        sigma = np.repeat(self.sigma, self.nbc_u, axis=0)
        gaussians = np_multivariate_normal_pdf(data, mu, sigma)
        forward, backward = self.get_forward_backward(gaussians)
        p_apost = forward
        p_apost_x = (p_apost.reshape((p_apost.shape[0], self.nbc_x, self.nbc_u))).sum(axis=2)
        return np.argmax(p_apost_x, axis=1)

    def seg_mpm_u(self, data):
        mu = np.repeat(self.mu, self.nbc_u, axis=0)
        sigma = np.repeat(self.sigma, self.nbc_u, axis=0)
        gaussians = np_multivariate_normal_pdf(data, mu, sigma)
        forward, backward = self.get_forward_backward(gaussians)
        p_apost = forward
        p_apost_u = (p_apost.reshape((p_apost.shape[0], self.nbc_x, self.nbc_u))).sum(axis=1)
        return np.argmax(p_apost_u, axis=1)

    def simul_hidden_apost(self, backward, gaussians, x_only=False):
        res = np.zeros(len(backward), dtype=int)
        T = calc_transDS(self.t, self.lx)
        T[np.isnan(T)] = 0
        aux = backward[0] / backward[0].sum()
        test = np.random.multinomial(1, aux)
        res[0] = np.argmax(test)
        tapost = (
                (backward[1:, np.newaxis, :] * gaussians[1:, np.newaxis, :]
                 * T[np.newaxis, :, :]) / backward[:-1, :, np.newaxis])
        tapost[np.isnan(tapost)] = 0
        tapost = tapost / tapost.sum(axis=2)[..., np.newaxis]
        tapost[np.isnan(tapost)] = 0
        for i in range(1, len(res)):
            test = np.random.multinomial(1, tapost[i - 1, res[i - 1], :])
            res[i] = np.argmax(test)
        if x_only:
            res = convert_multcls_vectors(res, (self.nbc_u, self.nbc_x))[:, 1]
        return res

    def simul_visible(self, hidden):
        res = np.zeros((len(hidden), self.mu.shape[-1]))
        mu = np.repeat(self.mu, self.nbc_u, axis=0)
        sigma = np.repeat(self.sigma, self.nbc_u, axis=0)
        for i in range(0, len(hidden)):
            res[i] = multivariate_normal.rvs(mu[hidden[i]], sigma[hidden[i]])
        return np.array(res)

    def generate_sample(self, length, x_only=True):
        hidden = np.zeros(length, dtype=int)
        visible = np.zeros((length, self.mu.shape[-1]))
        backward = self.get_backward_apri(length)
        mu = np.repeat(self.mu, self.nbc_u, axis=0)
        sigma = np.repeat(self.sigma, self.nbc_u, axis=0)
        T = calc_transDS(self.t, self.lx)
        tapri = (
                (backward[1:, np.newaxis, :]
                 * T[np.newaxis, :, :]) / backward[:-1, :, np.newaxis]
        )
        tapri[np.isnan(tapri)] = 0
        tapri = tapri / tapri.sum(axis=2)[..., np.newaxis]
        tapri[np.isnan(tapri)] = 0
        test = np.random.multinomial(1, backward[0] / backward[0].sum())
        hidden[0] = np.argmax(test)
        visible[0] = multivariate_normal.rvs(mu[hidden[0]], sigma[hidden[0]])
        for i in range(1, length):
            test = np.random.multinomial(1, tapri[i - 1, hidden[i - 1], :])
            hidden[i] = np.argmax(test)
            visible[i] = multivariate_normal.rvs(mu[hidden[i]], sigma[hidden[i]])
        if x_only:
            hidden = convert_multcls_vectors(hidden, (self.nbc_u, self.nbc_x))[:, 1]
        return hidden, visible

    def get_backward_apri(self, length):
        C = calc_matDS((self.t.T * self.p).T, self.lx)
        T = calc_transDS(self.t, self.lx)
        backward = np.zeros((length, T.shape[0]))
        backward[length - 1] = np.ones(T.shape[0])
        for l in reversed(range(0, length - 1)):
            if l == 0:
                phi = C
            else:
                phi = T
            backward[l] = phi @ (backward[l + 1])
            backward[l] = backward[l] / (backward[l].sum())

        return backward

    def get_gaussians(self, data):
        mu = np.repeat(self.mu, self.nbc_u, axis=0)
        sigma = np.repeat(self.sigma, self.nbc_u, axis=0)
        return np_multivariate_normal_pdf(data, mu, sigma)

    def get_forward_backward(self, gaussians):
        C = calc_matDS((self.t.T * self.p).T, self.lx)
        T = calc_transDS(self.t, self.lx)
        forward = np.zeros((len(gaussians), T.shape[0]))
        backward = np.zeros((len(gaussians), T.shape[0]))
        backward[len(gaussians) - 1] = np.ones(T.shape[0])
        for l in reversed(range(0, len(gaussians) - 1)):
            if l == 0:
                phi = ((C * gaussians[l + 1]).T * gaussians[l]).T
            else:
                phi = T * gaussians[l + 1]
            backward[l] = phi @ (backward[l + 1])
            backward[l] = backward[l] / (backward[l].sum())

        forward[0] = backward[0] / np.sum(backward[0])
        tapost = (
                (backward[1:, np.newaxis, :] * gaussians[1:, np.newaxis, :]
                 * T[np.newaxis, :, :]) / backward[:-1, :, np.newaxis])
        tapost[np.isnan(tapost)] = 0
        tapost = tapost / tapost.sum(axis=2)[..., np.newaxis]
        tapost[np.isnan(tapost)] = 0
        for k in range(1, len(gaussians)):
            forward[k] = (forward[k - 1] @ tapost[k - 1])

        return forward, backward

    def get_param_EM(self, data, iter):
        print({'iter': 0, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma})
        for q in range(iter):
            T = calc_transDS(self.t, self.lx)
            mu = np.repeat(self.mu, self.nbc_u, axis=0)
            sigma = np.repeat(self.sigma, self.nbc_u, axis=0)
            card = np.sum(self.lx, axis=0)
            gaussians = np_multivariate_normal_pdf(data, mu, sigma)
            forward, backward = self.get_forward_backward(gaussians)
            tapost = (
                    (backward[1:, np.newaxis, :] * gaussians[1:, np.newaxis, :]
                     * T[np.newaxis, :, :]) / backward[:-1, :, np.newaxis])
            tapost[np.isnan(tapost)] = 0
            tapost = tapost / tapost.sum(axis=2)[..., np.newaxis]
            tapost[np.isnan(tapost)] = 0
            gamma = tapost * forward[:-1, :, np.newaxis]
            psi = forward
            # u = (1/(gamma.shape[0])) * (np.outer((1/card),(1/card)) *
            #     gamma.reshape((gamma.shape[0], self.nbc_x, self.nbc_u, self.nbc_x, self.nbc_u)).sum(
            #         axis=(0, 1, 3)))
            # self.p = (card * u).sum(axis=1)
            # self.t = (u.T / self.p).T
            # self.t[np.isnan(self.t)] = 0
            self.t = (np.outer((1 / card), (1 / card)) * gamma.reshape(
                (gamma.shape[0], self.nbc_x, self.nbc_u, self.nbc_x, self.nbc_u)).sum(axis=(0, 1, 3))) \
                     / ((1 / card) * psi[:-1].reshape((psi[:-1].shape[0], self.nbc_x, self.nbc_u)).sum(axis=(0, 1)))
            self.p = (1 / psi.shape[0]) * (1 / card) * psi.reshape((psi.shape[0], self.nbc_x, self.nbc_u)).sum(
                axis=(0, 1))

            psi = (psi.reshape((psi.shape[0], self.nbc_x, self.nbc_u))).sum(axis=2)
            self.mu = (((psi * data).sum(axis=0)) / (psi.sum(axis=0))).reshape(self.mu.shape)
            broadc = (len(self.mu.shape) - len(data.shape) + 1)
            self.sigma = (psi.reshape(((psi.shape[0],) + self.sigma.shape)) * np.einsum('...i,...j',
                                                                                        (data.reshape(
                                                                                            (data.shape[0],) + (
                                                                                                1,) * broadc + data.shape[
                                                                                                               1:]) - self.mu),
                                                                                        (data.reshape(
                                                                                            (data.shape[0],) + (
                                                                                                1,) * broadc + data.shape[
                                                                                                               1:]) - self.mu))).sum(
                axis=0) / ((psi.sum(axis=0)).reshape(self.sigma.shape))
            print({'iter': q + 1, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma})

    def get_param_ICE(self, data, iter, Nb_simul):
        print({'iter': 0, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma})
        for q in range(iter):
            T = calc_transDS(self.t, self.lx)
            mu = np.repeat(self.mu, self.nbc_u, axis=0)
            sigma = np.repeat(self.sigma, self.nbc_u, axis=0)
            card = np.sum(self.lx, axis=0)
            gaussians = np_multivariate_normal_pdf(data, mu, sigma)
            forward, backward = self.get_forward_backward(gaussians)
            hidden = np.stack([self.simul_hidden_apost(backward, gaussians) for n in range(Nb_simul)], axis=0)
            tapost = (
                    (backward[1:, np.newaxis, :] * gaussians[1:, np.newaxis, :]
                     * T[np.newaxis, :, :]) / backward[:-1, :, np.newaxis])
            tapost[np.isnan(tapost)] = 0
            tapost = tapost / tapost.sum(axis=2)[..., np.newaxis]
            tapost[np.isnan(tapost)] = 0
            gamma = tapost * forward[:-1, :, np.newaxis]
            psi = forward
            u = (1 / (gamma.shape[0])) * (np.outer((1 / card), (1 / card)) *
                                          gamma.reshape(
                                              (gamma.shape[0], self.nbc_x, self.nbc_u, self.nbc_x, self.nbc_u)).sum(
                                              axis=(0, 1, 3)))
            self.p = (card * u).sum(axis=1)
            self.t = (u.T / self.p).T
            self.t[np.isnan(self.t)] = 0

            broadc = (len(self.mu.shape) - len(data.shape) + 1)
            self.mu = (
                    (((hidden[..., np.newaxis] // self.nbc_u) == np.indices((self.mu.shape[0],))) * data).sum(
                        axis=(0, 1)) /
                    ((hidden[..., np.newaxis] // self.nbc_u) == np.indices((self.mu.shape[0],))).sum(
                        axis=(0, 1))).reshape(self.mu.shape)
            self.sigma = (
                    (((hidden[..., np.newaxis] // self.nbc_u) == np.indices((self.mu.shape[0],))).reshape(
                        ((hidden.shape[0], hidden.shape[1]) + self.sigma.shape)) * np.einsum('...i,...j',
                                                                                             (data.reshape(
                                                                                                 (data.shape[0],) + (
                                                                                                     1,) * broadc + data.shape[
                                                                                                                    1:]) - self.mu),
                                                                                             (data.reshape(
                                                                                                 (data.shape[0],) + (
                                                                                                     1,) * broadc + data.shape[
                                                                                                                    1:]) - self.mu))).sum(
                        axis=(0, 1))
                    /
                    (((hidden[..., np.newaxis] // self.nbc_u) == np.indices((self.mu.shape[0],))).sum(
                        axis=(0, 1))).reshape(self.sigma.shape))
            print({'iter': q + 1, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma})

    def get_param_SEM(self, data, iter):
        print({'iter': 0, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma})
        for q in range(iter):
            T = calc_transDS(self.t, self.lx)
            mu = np.repeat(self.mu, self.nbc_u, axis=0)
            sigma = np.repeat(self.sigma, self.nbc_u, axis=0)
            card = np.sum(self.lx, axis=0)
            gaussians = np_multivariate_normal_pdf(data, mu, sigma)
            forward, backward = self.get_forward_backward(gaussians)
            hidden = self.simul_hidden_apost(backward, gaussians)
            hiddenc = np.stack((hidden[:-1], hidden[1:]), axis=-1)
            aux = np.moveaxis(np.indices(T.shape), 0, -1)
            broadc = (len(aux.shape) - len(hiddenc.shape) + 1)
            C = (1 / (len(data) - 1)) * (
                np.all(hiddenc.reshape((hiddenc.shape[0],) + (1,) * broadc + hiddenc.shape[1:]) == aux, axis=-1).sum(
                    axis=0)).reshape(T.shape)
            u = np.outer((1 / card), (1 / card)) * C.reshape(self.nbc_x, self.nbc_u, self.nbc_x, self.nbc_u).sum(
                axis=(0, 2))
            self.p = (card * u).sum(axis=1)
            self.t = (u.T / self.p).T
            self.t[np.isnan(self.t)] = 0

            broadc = (len(self.mu.shape) - len(data.shape) + 1)
            self.mu = ((((hidden[..., np.newaxis] // self.nbc_u) == np.indices((self.mu.shape[0],))) * data).sum(
                axis=0) / (
                               (hidden[..., np.newaxis] // self.nbc_u) == np.indices((self.mu.shape[0],))).sum(
                axis=0)).reshape(self.mu.shape)

            self.sigma = ((((hidden[..., np.newaxis] // self.nbc_u) == np.indices((self.mu.shape[0],))).reshape(
                ((hidden.shape[0],) + self.sigma.shape)) * np.einsum('...i,...j',
                                                                     (data.reshape((data.shape[0],) + (
                                                                         1,) * broadc + data.shape[1:]) - self.mu),
                                                                     (data.reshape(
                                                                         (data.shape[0],) + (
                                                                             1,) * broadc + data.shape[
                                                                                            1:]) - self.mu))).sum(
                axis=0)
                          / (((hidden[..., np.newaxis] // self.nbc_u) == np.indices((self.mu.shape[0],))).sum(
                        axis=0)).reshape(self.sigma.shape))

            print({'iter': q + 1, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma})


class HSMC_class_ctod:
    __slots__ = ('p', 't', 'mu', 'sigma', 'nbc_x', 'nbc_u')

    def __init__(self, nbc_x, p=None, t=None, mu=None, sigma=None):
        self.p = p
        self.t = t
        self.mu = mu
        self.sigma = sigma
        self.nbc_x = nbc_x
        self.nbc_u = 0

    def init_data_prior(self, data, scale=1):
        self.nbc_u = len(data)
        nb_class = self.nbc_x * self.nbc_u
        u = np.ones((self.nbc_x, self.nbc_x, self.nbc_u)) * (1 / self.nbc_u)
        x = np.array([1 / self.nbc_x] * self.nbc_x)
        a = np.full((self.nbc_x, self.nbc_x), 1 / (2 * (self.nbc_x - 1)))
        a = a - np.diag(np.diag(a))
        self.p = (np.sum((a.T * x).T * u.T, axis=1)).T.flatten()
        b = np.repeat(np.eye(self.nbc_x, self.nbc_x, k=0), self.nbc_u, axis=0)
        idx = [i for i in range(b.shape[0]) if ((i + 1) % self.nbc_u == 0)]
        for i, e in enumerate(idx):
            b[e] = a[i]

        a = u
        ut = [[np.eye(self.nbc_u, k=1) for n1 in range(self.nbc_x)] for n2 in range(int(nb_class / self.nbc_u))]
        for i, e in enumerate(ut):
            for j, p in enumerate(e):
                p[-1] = a[i, j]
        ut = np.block(ut)
        for i in range(1, self.nbc_x + 1):
            ut[:, (i - 1) * self.nbc_u:i * self.nbc_u] = (
                    ut[:, (i - 1) * self.nbc_u:i * self.nbc_u].T * b[:, i - 1]).T
        self.t = ut
        M = np.mean(data, axis=0)
        Sig = np.cov(data, rowvar=False).reshape(data.shape[1], data.shape[1])
        self.mu = [None] * self.nbc_x
        self.sigma = [None] * self.nbc_x
        for l in range(self.nbc_x):
            if l % 2 == 0:
                self.mu[l] = M - scale * ((l / (self.nbc_x / 2)) * np.sum(Sig, axis=1))
            else:
                self.mu[l] = M + scale * ((l / (self.nbc_x / 2)) * np.sum(Sig, axis=1))
            self.sigma[l] = Sig

        self.mu = np.array(self.mu)
        self.sigma = np.array(self.sigma)

    def init_kmeans(self, data, labels):
        self.nbc_u = len(data)
        hidden = labels
        c = (1 / (len(data) - 1)) * sum([np.array(
            [[(hidden[i - 1] == k and hidden[i] == l) for l in range(self.t.shape[1])] for k in
             range(self.t.shape[0])]) for i in range(1, len(data))])

        nb_class = self.nbc_x * self.nbc_u
        pu = np.array([1 / self.nbc_u] * self.nbc_u)
        self.p = np.outer(np.sum(c, axis=1), pu).flatten()
        self.p = self.p / np.sum(self.p)
        a = (c.T / np.sum(c, axis=1)).T
        a = a - np.diag(np.diag(a))
        b = np.repeat(np.eye(self.nbc_x, self.nbc_x, k=0), self.nbc_u, axis=0)
        idx = [i for i in range(b.shape[0]) if ((i + 1) % self.nbc_u == 0)]
        for i, e in enumerate(idx):
            b[e] = a[i]

        a = np.full((nb_class * self.nbc_x, self.nbc_u), 1 / (2 * (self.nbc_u - 1)))
        a = a - np.block([[np.eye(self.nbc_u)] * int(nb_class * self.nbc_x / self.nbc_u)]).T * 1 / (
                2 * (self.nbc_u - 1))
        a = a + np.block([[np.eye(self.nbc_u)] * int(nb_class * self.nbc_x / self.nbc_u)]).T * 1 / 2
        ut = [[np.eye(self.nbc_u, k=1) for n1 in range(self.nbc_x)] for n2 in range(int(nb_class / self.nbc_u))]
        ind = 0
        for e in ut:
            for p in e:
                p[-1] = a[ind]
                ind = ind + 1
        ut = np.block(ut)
        for i in range(1, self.nbc_x + 1):
            print(b[:, i - 1])
            ut[:, (i - 1) * self.nbc_u:i * self.nbc_u] = (
                    ut[:, (i - 1) * self.nbc_u:i * self.nbc_u].T * b[:, i - 1]).T
        self.t = ut

        self.mu = sum([np.array(
            [((hidden[i]) == l) * data[i] for l in range(self.mu.shape[0])]) for i in
            range(1, len(data))]) / sum(
            [np.array(
                [np.full((len(data[0])), ((hidden[i]) == l)) for l in range(self.mu.shape[0])]) for i in
                range(1, len(data))])

        self.sigma = sum([np.array(
            [((hidden[i]) == l) * np.outer(data[i] - self.mu[l],
                                           data[i] - self.mu[l])
             for l in
             range(self.sigma.shape[0])]) for i in range(1, len(data))]) / sum(
            [np.array([np.full((len(data[0]), len(data[0])), ((hidden[i]) == l)) for l in
                       range(self.sigma.shape[0])])
             for i
             in
             range(1, len(data))])

    def give_param(self, c, u, mu, sigma):
        nb_class = self.nbc_x * self.nbc_u
        self.p = (np.sum(c * u.T, axis=1)).T.flatten()
        a = (c.T / np.sum(c, axis=1)).T
        a = a - np.diag(np.diag(a))
        b = np.repeat(np.eye(self.nbc_x, self.nbc_x, k=0), self.nbc_u, axis=0)
        idx = [i for i in range(b.shape[0]) if ((i + 1) % self.nbc_u == 0)]
        for i, e in enumerate(idx):
            b[e] = a[i]

        a = u
        ut = [[np.eye(self.nbc_u, k=1) for n1 in range(self.nbc_x)] for n2 in range(int(nb_class / self.nbc_u))]
        for i, e in enumerate(ut):
            for j, p in enumerate(e):
                p[-1] = a[i, j]
        ut = np.block(ut)
        for i in range(1, self.nbc_x + 1):
            ut[:, (i - 1) * self.nbc_u:i * self.nbc_u] = (
                    ut[:, (i - 1) * self.nbc_u:i * self.nbc_u].T * b[:, i - 1]).T
        self.t = ut
        self.mu = mu
        self.sigma = sigma

    def seg_map(self, data):
        pass

    def seg_mpm(self, data):
        mu = np.repeat(self.mu, self.nbc_u, axis=0)
        sigma = np.repeat(self.sigma, self.nbc_u, axis=0)
        gaussians = np_multivariate_normal_pdf(data, mu, sigma)
        forward, backward = self.get_forward_backward(gaussians)
        p_apost = forward * backward
        p_apost = p_apost / (p_apost.sum(axis=1)[..., np.newaxis])
        p_apost_x = (p_apost.reshape((p_apost.shape[0], self.nbc_x, self.nbc_u))).sum(axis=2)
        return np.argmax(p_apost_x, axis=1)

    def simul_hidden_apost(self, backward, gaussians):
        res = np.zeros(len(backward), dtype=int)
        T = self.t
        aux = (gaussians[0] * self.p) * backward[0]
        test = np.random.multinomial(1, aux / np.sum(aux))
        res[0] = np.argmax(test)
        tapost = (
                (gaussians[1:, np.newaxis, :]
                 * backward[1:, np.newaxis, :]
                 * T[np.newaxis, :, :]) /
                (backward[:-1, :, np.newaxis])
        )
        tapost = tapost / tapost.sum(axis=2)[..., np.newaxis]
        for i in range(1, len(res)):
            test = np.random.multinomial(1, tapost[i - 1, res[i - 1], :])
            res[i] = np.argmax(test)
        return res

    def simul_visible(self, hidden):
        res = np.zeros((len(hidden), self.mu.shape[-1]))
        mu = np.repeat(self.mu, self.nbc_u, axis=0)
        sigma = np.repeat(self.sigma, self.nbc_u, axis=0)
        for i in range(0, len(hidden)):
            res[i] = multivariate_normal.rvs(mu[hidden[i]], sigma[hidden[i]])
        return np.array(res)

    def generate_sample(self, length):
        hidden = np.zeros(length, dtype=int)
        visible = np.zeros((length, self.mu.shape[-1]))
        T = self.t
        mu = np.repeat(self.mu, self.nbc_u, axis=0)
        sigma = np.repeat(self.sigma, self.nbc_u, axis=0)
        test = np.random.multinomial(1, self.p)
        hidden[0] = np.argmax(test)
        visible[0] = multivariate_normal.rvs(mu[hidden[0]], sigma[hidden[0]])
        for i in range(1, length):
            test = np.random.multinomial(1, T[hidden[i - 1], :])
            hidden[i] = np.argmax(test)
            visible[i] = multivariate_normal.rvs(mu[hidden[i]], sigma[hidden[i]])

        return hidden, visible

    def get_forward_backward(self, gaussians):
        forward = np.zeros((len(gaussians), self.t.shape[0]))
        backward = np.zeros((len(gaussians), self.t.shape[0]))
        backward[len(gaussians) - 1] = np.ones(self.t.shape[0])
        forward[0] = self.p * gaussians[0]
        forward[0] = forward[0] / (forward[0].sum())
        T = self.t
        for l in range(1, len(gaussians)):
            k = len(gaussians) - 1 - l
            forward[l] = gaussians[l] * (forward[l - 1] @ T)
            forward[l] = forward[l] / forward[l].sum()
            backward[k] = (gaussians[k + 1] * backward[k + 1]) @ T.T
            backward[k] = backward[k] / (backward[k].sum())

        return forward, backward

    def get_param_EM(self, data, iter):
        print({'iter': 0, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma})
        for q in range(iter):
            T = self.t
            mu = np.repeat(self.mu, self.nbc_u, axis=0)
            sigma = np.repeat(self.sigma, self.nbc_u, axis=0)
            gaussians = np_multivariate_normal_pdf(data, mu, sigma)
            forward, backward = self.get_forward_backward(gaussians)
            gamma = (
                    forward[:-1, :, np.newaxis]
                    * (gaussians[1:, np.newaxis, :]
                       * backward[1:, np.newaxis, :]
                       * T[np.newaxis, :, :])
            )
            gamma = gamma / (gamma.sum(axis=(1, 2))[..., np.newaxis, np.newaxis])
            psi = forward * backward
            psi = psi / (psi.sum(axis=1)[..., np.newaxis])
            self.t = np.transpose(np.transpose((gamma.sum(axis=0))) / (psi[:-1:].sum(axis=0)))
            self.p = (psi.sum(axis=0)) / psi.shape[0]
            psi = (psi.reshape((psi.shape[0], self.nbc_x, self.nbc_u))).sum(axis=2)
            self.mu = (((psi * data).sum(axis=0)) / (psi.sum(axis=0))).reshape(self.mu.shape)
            broadc = (len(self.mu.shape) - len(data.shape) + 1)
            self.sigma = (psi.reshape(((psi.shape[0],) + self.sigma.shape)) * np.einsum('...i,...j',
                                                                                        (data.reshape(
                                                                                            (data.shape[0],) + (
                                                                                                1,) * broadc + data.shape[
                                                                                                               1:]) - self.mu),
                                                                                        (data.reshape(
                                                                                            (data.shape[0],) + (
                                                                                                1,) * broadc + data.shape[
                                                                                                               1:]) - self.mu))).sum(
                axis=0) / ((psi.sum(axis=0)).reshape(self.sigma.shape))
            print({'iter': q + 1, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma})

    def get_param_ICE(self, data, iter, Nb_simul):
        print({'iter': 0, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma})
        for q in range(iter):
            T = self.t
            mu = np.repeat(self.mu, self.nbc_u, axis=0)
            sigma = np.repeat(self.sigma, self.nbc_u, axis=0)
            gaussians = np_multivariate_normal_pdf(data, mu, sigma)
            forward, backward = self.get_forward_backward(gaussians)
            hidden = np.stack([self.simul_hidden_apost(backward, gaussians) for n in range(Nb_simul)], axis=0)
            gamma = (
                    forward[:-1, :, np.newaxis]
                    * (gaussians[1:, np.newaxis, :]
                       * backward[1:, np.newaxis, :]
                       * T[np.newaxis, :, :])
            )
            gamma = gamma / (gamma.sum(axis=(1, 2))[..., np.newaxis, np.newaxis])
            psi = forward * backward
            psi = psi / (psi.sum(axis=1)[..., np.newaxis])
            self.t = np.transpose(np.transpose((gamma.sum(axis=0))) / (psi[:-1:].sum(axis=0)))
            self.p = (psi.sum(axis=0)) / psi.shape[0]

            broadc = (len(self.mu.shape) - len(data.shape) + 1)
            self.mu = (
                    (((hidden[..., np.newaxis] // self.nbc_u) == np.indices((self.mu.shape[0],))) * data).sum(
                        axis=(0, 1))
                    /
                    ((hidden[..., np.newaxis] // self.nbc_u) == np.indices((self.mu.shape[0],))).sum(
                        axis=(0, 1))).reshape(self.mu.shape)
            self.sigma = (
                    (((hidden[..., np.newaxis] // self.nbc_u) == np.indices((self.mu.shape[0],))).reshape(
                        ((hidden.shape[0], hidden.shape[1]) + self.sigma.shape)) * np.einsum('...i,...j',
                                                                                             (data.reshape(
                                                                                                 (data.shape[0],) + (
                                                                                                     1,) * broadc + data.shape[
                                                                                                                    1:]) - self.mu),
                                                                                             (data.reshape(
                                                                                                 (data.shape[0],) + (
                                                                                                     1,) * broadc + data.shape[
                                                                                                                    1:]) - self.mu))).sum(
                        axis=(0, 1)) /
                    (((hidden[..., np.newaxis] // self.nbc_u) == np.indices((self.mu.shape[0],))).sum(
                        axis=(0, 1))).reshape(self.sigma.shape))

            print({'iter': q + 1, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma})

    def get_param_SEM(self, data, iter):
        print({'iter': 0, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma})
        for q in range(iter):
            mu = np.repeat(self.mu, self.nbc_u, axis=0)
            sigma = np.repeat(self.sigma, self.nbc_u, axis=0)
            gaussians = np_multivariate_normal_pdf(data, mu, sigma)
            forward, backward = self.get_forward_backward(gaussians)
            hidden = self.simul_hidden_apost(backward, gaussians)
            hiddenc = np.stack((hidden[:-1], hidden[1:]), axis=-1)
            aux = np.moveaxis(np.indices(self.t.shape), 0, -1)
            broadc = (len(aux.shape) - len(hiddenc.shape) + 1)
            c = (1 / (len(data) - 1)) * (
                np.all(hiddenc.reshape((hiddenc.shape[0],) + (1,) * broadc + hiddenc.shape[1:]) == aux, axis=-1).sum(
                    axis=0)).reshape(self.t.shape)

            self.p = (1 / (len(data))) * (hidden[..., np.newaxis] == np.indices((self.p.shape[0],))).sum(axis=0)
            self.t = (c.T / self.p).T
            self.t[np.isnan(self.t)] = 0

            broadc = (len(self.mu.shape) - len(data.shape) + 1)
            self.mu = ((((hidden[..., np.newaxis] // self.nbc_u) == np.indices((self.mu.shape[0],))) * data).sum(
                axis=0) / (
                               (hidden[..., np.newaxis] // self.nbc_u) == np.indices((self.mu.shape[0],))).sum(
                axis=0)).reshape(self.mu.shape)
            self.sigma = ((((hidden[..., np.newaxis] // self.nbc_u) == np.indices((self.mu.shape[0],))).reshape(
                ((hidden.shape[0],) + self.sigma.shape)) * np.einsum('...i,...j',
                                                                     (data.reshape((data.shape[0],) + (
                                                                         1,) * broadc + data.shape[1:]) - self.mu),
                                                                     (data.reshape(
                                                                         (data.shape[0],) + (
                                                                             1,) * broadc + data.shape[
                                                                                            1:]) - self.mu))).sum(
                axis=0)
                          / (((hidden[..., np.newaxis] // self.nbc_u) == np.indices((self.mu.shape[0],))).sum(
                        axis=0)).reshape(self.sigma.shape))

            print({'iter': q + 1, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma})


class HMT_ctod:
    __slots__ = ('p', 't', 'mu', 'sigma', 'struct')

    def __init__(self, struct, p=None, t=None, mu=None, sigma=None):
        self.p = p
        self.t = t
        self.mu = mu
        self.sigma = sigma
        self.struct = struct

    def init_kmeans(self, data, labels):
        nb_class = len(np.unique(labels))

        aux_c = (1 / (len(data) - 1)) * sum([np.array(
            [[(labels[i - 1] == k and labels[i] == l) for l in range(nb_class)] for k in
             range(nb_class)]) for i in range(1, len(data))])

        aux_p = (1 / (len(data) - 1)) * sum([np.array(
            [labels[i] == l for l in range(nb_class)]) for i in range(0, len(data) - 1)])

        self.p = (1 / (len(data) - 1)) * sum([np.array(
            [labels[i] == l for l in range(nb_class)]) for i in range(0, len(data))])

        self.t = (aux_c.T / aux_p).T

        self.mu = sum([np.array(
            [(labels[i] == l) * data[i] for l in range(nb_class)]) for i in range(1, len(data))]) / sum(
            [np.array(
                [np.full((len(data[0])), (labels[i] == l)) for l in range(nb_class)]) for i in range(1, len(data))])

        self.sigma = sum([np.array(
            [(labels[i] == l) * np.outer(data[i] - self.mu[l], data[i] - self.mu[l])
             for l in
             range(nb_class)]) for i in range(1, len(data))]) / sum(
            [np.array([np.full((len(data[0]), len(data[0])), (labels[i] == l)) for l in range(nb_class)]) for i in
             range(1, len(data))])

    def init_data_prior(self, data, nb_class, scale=1):
        self.p = np.array([1 / nb_class] * nb_class)
        a = np.full((nb_class, nb_class), 1 / (2 * (nb_class - 1)))
        a = a - np.diag(np.diag(a))
        self.t = np.diag(np.array([1 / 2] * nb_class)) + a
        data = np.concatenate(data, axis=0)
        M = np.mean(data, axis=0)
        Sig = np.cov(data, rowvar=False).reshape(data.shape[1], data.shape[1])
        self.mu = [None] * nb_class
        self.sigma = [None] * nb_class
        for l in range(nb_class):
            if l % 2 == 0:
                self.mu[l] = M - scale * ((l / (nb_class / 2)) * np.sum(Sig, axis=1))
            else:
                self.mu[l] = M + scale * ((l / (nb_class / 2)) * np.sum(Sig, axis=1))
            self.sigma[l] = Sig

        self.mu = np.array(self.mu)
        self.sigma = np.array(self.sigma)

    def seg_map(self, data):
        pass

    def seg_mpm(self, data):
        n = np.log(len(data[0])) / np.log(self.struct)
        assert n.is_integer(), 'Bad data length'
        n = int(n)
        backward = self.get_backward(data)
        forward = self.get_forward(backward, data)
        res = [np.zeros((int(len(backward[0]) / (self.struct ** k)))) for k in
               range(n + 1)]
        aux = backward[n][0] * forward[n][0]
        test = np.random.multinomial(1, aux / np.sum(aux))
        res[n][0] = test.tolist().index(1)
        for k in reversed(range(len(res) - 1)):
            for l in range(0, int(len(backward[0]) / (self.struct ** k))):
                aux = backward[k][l] * forward[k][l]
                aux = aux / np.sum(aux)
                res[k][l] = aux.tolist().index(max(aux))

        return [np.array(r) for r in res]

    def simul_hidden_indep(self, data, forward, backward):
        n = np.log(len(data[0])) / np.log(self.struct)
        assert n.is_integer(), 'Bad data length'
        n = int(n)
        res = [np.zeros((int(len(backward[0]) / (self.struct ** k))), dtype=int) for k in
               range(n + 1)]
        aux = backward[n][0] * forward[n][0]
        test = np.random.multinomial(1, aux / np.sum(aux))
        res[n][0] = test.tolist().index(1)
        for k in reversed(range(len(res) - 1)):
            for l in range(0, int(len(backward[0]) / (self.struct ** k))):
                aux = backward[k][l] * forward[k][l]
                test = np.random.multinomial(1, aux / np.sum(aux))
                res[k][l] = test.tolist().index(1)

        return [np.array(r) for r in res]

    def simul_hidden_apost(self, data, backward):
        n = np.log(len(data[0])) / np.log(self.struct)
        assert n.is_integer(), 'Bad data length'
        n = int(n)
        res = [np.zeros((int(len(backward[0]) / (self.struct ** k))), dtype=int) for k in
               range(n + 1)]
        aux = backward[n][0] * self.p
        test = np.random.multinomial(1, aux / np.sum(aux))
        res[n][0] = test.tolist().index(1)
        for k in reversed(range(len(res) - 1)):
            for l in range(0, int(len(backward[0]) / (self.struct ** k))):
                aux = (backward[k][l] * self.t) / np.sum(backward[k][l] * self.t)
                aux = aux[res[k - 1][(l // self.struct)]]
                test = np.random.multinomial(1, aux / np.sum(aux))
                res[k][l] = test.tolist().index(1)

        return [np.array(r) for r in res]

    def simul_visible(self, hidden, p=1):
        res = [np.zeros((int(len(hidden[0]) / (self.struct ** k)))) for k in
               range(p)]
        for k in range(len(res)):
            for l in range(0, int(len(res[0]) / (self.struct ** k))):
                res[k][l] = multivariate_normal.rvs(self.mu[hidden[k][l]], self.sigma[hidden[k][l]])
        return [np.array(r) for r in res]

    def generate_sample(self, length, p=1):
        n = np.log(length) / np.log(self.struct)
        assert n.is_integer(), 'Bad data length'
        n = int(n)
        assert p <= n, 'Bad tree length'
        hidden = [np.zeros((int(length / (self.struct ** k))), dtype=int) for k in
                  range(n + 1)]
        visible = [np.zeros((int(len(hidden[0]) / (self.struct ** k)))) for k in
                   range(p + 1)]
        test = np.random.multinomial(1, self.p)

        hidden[n][0] = test.tolist().index(1)
        if p == n:
            visible[p][0] = multivariate_normal.rvs(self.mu[hidden[n][0]], self.sigma[hidden[n][0]])
        for k in reversed(range(len(hidden) - 1)):
            for l in range(0, int(len(hidden[0]) / (self.struct ** k))):
                test = np.random.multinomial(1, self.t[hidden[k + 1][(l // self.struct)], :])
                hidden[k][l] = test.tolist().index(1)
                if k < len(visible):
                    visible[k][l] = multivariate_normal.rvs(self.mu[hidden[k][l]], self.sigma[hidden[k][l]])

        return [np.array(h) for h in hidden], [np.array(v) for v in visible]

    def get_backward(self, data):
        n = np.log(len(data[0])) / np.log(self.struct)
        assert n.is_integer(), 'Bad data length'
        n = int(n)
        backward = [[np.ones(self.t.shape[0]) for n in range(int(len(data[0]) / (self.struct ** k)))] for k in
                    range(n + 1)]
        for k in range(1, n + 1):
            for l in range(0, int(len(data[0]) / (self.struct ** k))):
                if (k - 1) < len(data):
                    backward[k][l] = np.prod(
                        [np.dot(self.t,
                                backward[k - 1][i] * np_multivariate_normal_pdf(data[k - 1][i], self.mu, self.sigma))
                         for i in range((l - 1) * self.struct, l * self.struct)], axis=0)
                else:
                    backward[k][l] = np.prod(
                        [np.dot(self.t,
                                backward[k - 1][i] * 1)
                         for i in range((l - 1) * self.struct, l * self.struct)], axis=0)

                backward[k][l] = backward[k][l] / np.sum(backward[k][l])

        return backward

    def get_forward(self, backward, data):
        n = len(backward) - 1
        forward = [[np.ones(self.t.shape[0]) for n in range(int(len(backward[0]) / (self.struct ** k)))] for k in
                   range(n + 1)]
        forward[n][0] = backward[n][0] * self.p
        forward[n][0] = forward[n][0] / np.sum(forward[n][0])
        for k in reversed(range(len(forward) - 1)):
            for l in range(0, int(len(backward[0]) / (self.struct ** k))):
                if (k) < len(data):
                    aux = backward[k][l] * self.t * np_multivariate_normal_pdf(data[k][l], self.mu, self.sigma)
                    aux = (aux.T / np.sum(aux, axis=1)).T
                else:
                    aux = backward[k][l] * self.t
                    aux = (aux.T / np.sum(aux, axis=1)).T
                forward[k][l] = np.dot(aux.T, forward[k + 1][l // self.struct])

        return forward

    def get_param_EM(self, data, iter):
        n = np.log(len(data[0])) / np.log(self.struct)
        assert n.is_integer(), 'Bad data length'
        n = int(n)
        print({'iter': 0, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma})
        for q in range(iter):
            backward = self.get_backward(data)
            forward = self.get_forward(backward, data)

            aux1 = [
                [np.zeros((self.t.shape[0], self.t.shape[1])) for n in range(int(len(data[0]) / (self.struct ** k)))]
                for k in
                range(n)]
            mu_aux = [
                [np.zeros((self.mu.shape[0])) for n in range(int(len(data[0]) / (self.struct ** k)))]
                for k in
                range(len(data))]
            mu_aux_norm = [
                [np.zeros((self.mu.shape[0])) for n in range(int(len(data[0]) / (self.struct ** k)))]
                for k in
                range(len(data))]

            sigma_aux = [
                [np.zeros((self.sigma.shape[0])) for n in range(int(len(data[0]) / (self.struct ** k)))]
                for k in
                range(len(data))]

            sigma_aux_norm = [
                [np.zeros((self.sigma.shape[0])) for n in range(int(len(data[0]) / (self.struct ** k)))]
                for k in
                range(len(data))]

            for k in range(n + 1):
                for l in range(0, int(len(data[0]) / (self.struct ** k))):
                    if k < len(data):
                        if k < n:
                            aux1[k][l] = (backward[k][l] * self.t * np_multivariate_normal_pdf(data[k][l], self.mu,
                                                                                               self.sigma))
                            aux1[k][l] = (aux1[k][l].T / np.sum(aux1[k][l], axis=1)).T
                            aux1[k][l] = (aux1[k][l].T * forward[k + 1][l // self.struct]).T

                        mu_aux[k][l] = np.array([(forward[k][l][i]) * data[k][l] for i in range(self.mu.shape[0])])
                        mu_aux_norm[k][l] = np.array(
                            [np.full((len(data[k][l])), (forward[k][l][i])) for i in range(self.mu.shape[0])])
                        sigma_aux[k][l] = np.array(
                            [(forward[k][l][i]) * np.outer(data[k][l] - self.mu[i],
                                                           data[k][l] - self.mu[i])
                             for i in
                             range(self.sigma.shape[0])])
                        sigma_aux_norm[k][l] = np.array(
                            [np.full((len(data[k][l]), len(data[k][l])), (forward[k][l][i])) for i in
                             range(self.sigma.shape[0])])
                    else:
                        if k < n:
                            aux1[k][l] = (backward[k][l] * self.t)
                            aux1[k][l] = (aux1[k][l].T / np.sum(aux1[k][l], axis=1)).T
                            aux1[k][l] = (aux1[k][l].T * forward[k + 1][l // self.struct]).T

            gamma = [forward[k + 1][(l // self.struct)] for k in range(len(aux1)) for l in range(len(aux1[k]))]
            aux1 = list(itertools.chain(*aux1))
            aux2 = list(itertools.chain(*forward))
            mu_aux = list(itertools.chain(*mu_aux))
            mu_aux_norm = list(itertools.chain(*mu_aux_norm))
            sigma_aux = list(itertools.chain(*sigma_aux))
            sigma_aux_norm = list(itertools.chain(*sigma_aux_norm))

            self.p = (1 / len(aux2)) * sum(aux2)

            self.t = (sum(aux1).T / sum(gamma)).T

            self.mu = sum(mu_aux) / sum(mu_aux_norm)

            self.sigma = sum(sigma_aux) / sum(sigma_aux_norm)

            print({'iter': q + 1, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma})

    def get_param_ICE(self, data, iter, Nb_simul, modified=False):
        n = np.log(len(data[0])) / np.log(self.struct)
        assert n.is_integer(), 'Bad data length'
        n = int(n)
        print({'iter': 0, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma})
        for q in range(iter):
            backward = self.get_backward(data)
            forward = self.get_forward(backward)
            aux1 = [
                [np.zeros((self.t.shape[0], self.t.shape[1])) for n in range(int(len(data[0]) / (self.struct ** k)))]
                for k in
                range(n + 1)]
            aux2 = [
                [np.zeros((self.t.shape[0])) for n in range(int(len(data[0]) / (self.struct ** k)))]
                for k in
                range(n + 1)]

            for k in range(1, n + 1):
                for l in range(0, int(len(data[0]) / (self.struct ** k))):
                    aux1[k][l] = (backward[k][l] * self.t) / np.sum(backward[k][l] * self.t)
                    aux2[k][l] = (forward[k][l] * backward[k][l]) / np.sum(forward[k][l] * backward[k][l])

            backward = self.get_backward(data)
            forward = self.get_forward(backward)
            if modified:
                hidden_list = [self.simul_hidden_indep(data, forward, backward) for n in range(Nb_simul)]
            else:
                hidden_list = [self.simul_hidden_apost(data, backward) for n in range(Nb_simul)]

            self.mu = sum([sum([np.array(
                [(hidden[k][i] == l) * data[k][i] for l in range(self.mu.shape[0])]) for k in range(len(data)) for i in
                range(len(data[k]))]) for hidden
                in
                hidden_list]) / sum(
                [sum([np.array(
                    [np.full((len(data[0][0])), (hidden[k][i] == l)) for l in
                     range(self.mu.shape[0])]) for k in range(len(data)) for i in range(len(data[k]))]) for hidden in
                    hidden_list])

            self.sigma = sum([sum([np.array(
                [(hidden[k][i] == l) * np.outer(data[k][i] - self.mu[l],
                                                data[k][i] - self.mu[l]) for l
                 in range(self.sigma.shape[0])]) for k in range(len(data)) for i in range(len(data[k]))]) for hidden in
                hidden_list]) / sum(
                [sum([np.array(
                    [np.full((len(data[0][0]), len(data[0][0])), (hidden[k][i] == l)) for
                     l in range(self.sigma.shape[0])]) for k in range(len(data)) for i in range(len(data[k]))]) for
                    hidden in hidden_list])

            print({'iter': q + 1, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma})

    def get_param_SEM(self, data, iter, modified=False):
        print({'iter': 0, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma})
        for q in range(iter):
            forward, backward = self.get_forward_backward(data)
            if modified:
                hidden = self.simul_hidden_indep_couple(data, forward, backward)
            else:
                hidden = self.simul_hidden_apost(data, backward)

            c = (1 / (len(data) - 1)) * sum([np.array(
                [[(hidden[i - 1] == k and hidden[i] == l) for l in range(self.t.shape[1])] for k in
                 range(self.t.shape[0])]) for i in range(1, len(data))])

            self.p = (1 / (len(data))) * sum([np.array(
                [(hidden[i] == l) for l in range(len(self.p))]) for i in range(len(data))])
            self.t = (c.T / self.p).T

            self.mu = sum([np.array(
                [(hidden[i] == l) * data[i] for l in range(self.mu.shape[0])]) for i in
                range(1, len(data))]) / sum(
                [np.array(
                    [np.full((len(data[0])), (hidden[i] == l)) for l in range(self.mu.shape[0])]) for i in
                    range(1, len(data))])

            self.sigma = sum([np.array(
                [(hidden[i] == l) * np.outer(data[i] - self.mu[l],
                                             data[i] - self.mu[l])
                 for l in
                 range(self.sigma.shape[0])]) for i in range(1, len(data))]) / sum(
                [np.array([np.full((len(data[0]), len(data[0])), (hidden[i] == l)) for l in range(self.sigma.shape[0])])
                 for i
                 in
                 range(1, len(data))])

            print({'iter': q + 1, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma})

    def get_param_supervised(self, data, hidden):

        c = (1 / (len(data) - 1)) * sum([np.array(
            [[(hidden[i - 1] == k and hidden[i] == l) for l in range(self.t.shape[1])] for k in
             range(self.t.shape[0])]) for i in range(1, len(data))])

        self.p = (1 / (len(data) - 1)) * sum([np.array(
            [(hidden[i] == l) for l in range(len(self.p))]) for i in range(1, len(data))])
        self.t = (c.T / self.p).T

        self.mu = sum([np.array(
            [(hidden[i] == l) * data[i] for l in range(self.mu.shape[0])]) for i in
            range(1, len(data))]) / sum(
            [np.array(
                [np.full((len(data[0])), (hidden[i] == l)) for l in range(self.mu.shape[0])]) for i in
                range(1, len(data))])

        self.sigma = sum([np.array(
            [(hidden[i] == l) * np.outer(data[i] - self.mu[l],
                                         data[i] - self.mu[l])
             for l in
             range(self.sigma.shape[0])]) for i in range(1, len(data))]) / sum(
            [np.array([np.full((len(data[0]), len(data[0])), (hidden[i] == l)) for l in range(self.sigma.shape[0])])
             for i
             in
             range(1, len(data))])


class HMC_multiR_ctod:
    __slots__ = ('c', 'mu', 'sigma', 'resoffset', 'resnbc', 'indep')

    def __init__(self, resoffset, resnbc, indep=False, c=None, mu=None, sigma=None):
        self.c = c
        self.mu = mu
        self.sigma = sigma
        self.resoffset = resoffset
        self.resnbc = resnbc
        self.indep = indep

    def init_rand_prior(self, dimY, mean_bound, var_bound, nb_obs=1):
        nb_class = np.prod(self.resnbc)
        self.c = np.random.dirichlet(np.ones((nb_class ** 2))).reshape(nb_class, nb_class)
        if self.indep:
            aux_mu = [[np.random.rand(1, dimY) * (mean_bound[1] - mean_bound[0]) + mean_bound[0] for i in range(x)] for
                      x in self.resnbc]

            self.mu = np.random.rand(nb_class, dimY * nb_obs) * (mean_bound[1] - mean_bound[0]) + mean_bound[0]
            self.sigma = np.array(
                [block_diag(*[generate_semipos_sym_mat((dimY, dimY), var_bound) for p in range(nb_obs)]) for n in
                 range(nb_class)])
        else:
            self.mu = np.random.rand(nb_class, dimY * nb_obs) * (mean_bound[1] - mean_bound[0]) + mean_bound[0]
            self.sigma = np.array(
                [generate_semipos_sym_mat((dimY * nb_obs, dimY * nb_obs), var_bound) for n in range(nb_class)])

    def init_data_prior(self, data, scale=1):
        nb_class = np.prod(self.resnbc)
        pi = np.array([1 / nb_class] * nb_class)
        a = np.full((nb_class, nb_class), 1 / (2 * (nb_class - 1)))
        a = a - np.diag(np.diag(a))
        a = np.diag(np.array([1 / 2] * nb_class)) + a
        self.c = (a.T * pi).T
        M = np.mean(data, axis=0)
        Sig = np.cov(data, rowvar=False).reshape(data.shape[1], data.shape[1])
        self.mu = [None] * nb_class
        self.sigma = [None] * nb_class
        for l in range(nb_class):
            if l % 2 == 0:
                self.mu[l] = M - scale * ((l / (nb_class / 2)) * np.sum(Sig, axis=1))
            else:
                self.mu[l] = M + scale * ((l / (nb_class / 2)) * np.sum(Sig, axis=1))
            self.sigma[l] = Sig

        self.mu = np.array(self.mu)
        self.sigma = np.array(self.sigma)

    def init_kmeans(self, data, labels):
        nb_class = len(np.unique(labels))

        self.c = (1 / (len(data) - 1)) * sum([np.array(
            [[(labels[i - 1] == k and labels[i] == l) for l in range(nb_class)] for k in
             range(nb_class)]) for i in range(1, len(data))])

        self.mu = sum([np.array(
            [(labels[i] == l) * data[i] for l in range(nb_class)]) for i in range(1, len(data))]) / sum(
            [np.array(
                [np.full((len(data[0])), (labels[i] == l)) for l in range(nb_class)]) for i in range(1, len(data))])

        self.sigma = sum([np.array(
            [(labels[i] == l) * np.outer(data[i] - self.mu[l], data[i] - self.mu[l])
             for l in
             range(nb_class)]) for i in range(1, len(data))]) / sum(
            [np.array([np.full((len(data[0]), len(data[0])), (labels[i] == l)) for l in range(nb_class)]) for i in
             range(1, len(data))])

    def seg_map(self, data):
        pass

    def seg_mpm(self, data, res_lvl):
        forward, backward = self.get_forward_backward(data)
        seg_res = self.resnbc[res_lvl]
        marginal_dim = np.prod([x for i, x in enumerate(self.resnbc) if i != res_lvl])
        res = [None] * len(data)
        for i in range(0, len(data)):
            aux = (forward[i] * backward[i])
            aux = aux / np.sum(aux)
            aux = np.sum(aux.reshape((marginal_dim, seg_res)), axis=0)
            res[i] = aux.tolist().index(max(aux))
        return np.array(res)

    def simul_hidden_indep(self, data, forward, backward):
        res = [None] * len(data)
        aux = forward[0] * backward[0]
        test = np.random.multinomial(1, aux / np.sum(aux))
        res[0] = test.tolist().index(1)
        for i in range(1, len(data)):
            check = False
            for l in reversed(range(len(self.resoffset))):
                if (i % np.prod(self.resoffset[:l + 1])) == 0 and not check:
                    sep = np.prod(self.resnbc[:l + 2])
                    cls = np.array(range(len(forward[i]))).reshape(-1, sep).tolist()
                    cls = [c for c in cls if res[i - 1] in c][0]
                    aux = (forward[i] * backward[i])
                    aux = aux / np.sum(aux)
                    aux = np.sum(aux.reshape(-1, sep), axis=0)
                    test = np.random.multinomial(1, aux)
                    res[i] = cls[test.tolist().index(1)]
                    check = True
                    break
            if not check:
                sep = self.resnbc[0]
                cls = np.array(range(len(forward[i]))).reshape(-1, sep).tolist()
                cls = [c for c in cls if res[i - 1] in c][0]
                aux = (forward[i] * backward[i])
                aux = aux / np.sum(aux)
                aux = np.sum(aux.reshape(-1, sep), axis=0)
                test = np.random.multinomial(1, aux)
                res[i] = cls[test.tolist().index(1)]

        return np.array(res)

    def simul_hidden_indep_couple(self, data, forward, backward):
        res = [None] * len(data)
        T = (self.c.T / np.sum(self.c, axis=1)).T
        if (len(data) % 2 == 0):
            for i in range(1, len(data), 2):
                check = False
                for l in reversed(range(len(self.resoffset))):
                    if (i % np.prod(self.resoffset[:l + 1])) == 0 and not check:
                        aux = (forward[i - 1] * (
                                T * np_multivariate_normal_pdf(data[i], self.mu, self.sigma) * backward[i]).T).T
                        aux = aux / np.sum(aux)
                        cls = [a.flatten() for a in np.indices(aux.shape)]
                        test = np.random.multinomial(1, aux[cls[0], cls[1]])
                        res[i - 1] = cls[0][test.tolist().index(1)]
                        res[i] = cls[1][test.tolist().index(1)]
                        check = True
                        break
                if not check:
                    aux = (forward[i - 1] * (
                            T * np_multivariate_normal_pdf(data[i], self.mu, self.sigma) * backward[i]).T).T
                    aux = aux / np.sum(aux)
                    cls = [a.flatten() for a in np.indices(aux.shape)]
                    test = np.random.multinomial(1, aux[cls[0], cls[1]])
                    res[i - 1] = cls[0][test.tolist().index(1)]
                    res[i] = cls[1][test.tolist().index(1)]

        else:
            aux = forward[0] * backward[0]
            test = np.random.multinomial(1, aux / np.sum(aux))
            res[0] = test.tolist().index(1)
            for i in range(2, len(data), 2):
                check = False
                for l in reversed(range(len(self.resoffset))):
                    if (i % np.prod(self.resoffset[:l + 1])) == 0 and not check:
                        aux = (forward[i - 1] * (
                                T * np_multivariate_normal_pdf(data[i], self.mu, self.sigma) * backward[i]).T).T
                        aux = aux / np.sum(aux)
                        cls = [a.flatten() for a in np.indices(aux.shape)]
                        test = np.random.multinomial(1, aux[cls[0], cls[1]])
                        res[i - 1] = cls[0][test.tolist().index(1)]
                        res[i] = cls[1][test.tolist().index(1)]
                        check = True
                        break
                if not check:
                    aux = (forward[i - 1] * (
                            T * np_multivariate_normal_pdf(data[i], self.mu, self.sigma) * backward[i]).T).T
                    aux = aux / np.sum(aux)
                    cls = [a.flatten() for a in np.indices(aux.shape)]
                    test = np.random.multinomial(1, aux[cls[0], cls[1]])
                    res[i - 1] = cls[0][test.tolist().index(1)]
                    res[i] = cls[1][test.tolist().index(1)]

        return np.array(res)

    def simul_hidden_apost(self, data, backward):
        res = [None] * len(data)
        T = (self.c.T / np.sum(self.c, axis=1)).T
        aux = (np_multivariate_normal_pdf(data[0], self.mu, self.sigma) * np.sum(self.c, axis=1))
        test = np.random.multinomial(1, aux / np.sum(aux))
        res[0] = test.tolist().index(1)
        for i in range(1, len(data)):
            check = False
            for l in reversed(range(len(self.resoffset))):
                if (i % np.prod(self.resoffset[:l + 1])) == 0 and not check:
                    sep = np.prod(self.resnbc[:l + 2])
                    aux = T * np_multivariate_normal_pdf(data[i], self.mu, self.sigma) * backward[i]
                    aux = (aux.T / (backward[i - 1] * np.sum(aux))).T[res[i - 1]]
                    cls = np.array(range(len(aux))).reshape(-1, sep).tolist()
                    cls = [c for c in cls if res[i - 1] in c][0]
                    aux = np.sum(aux.reshape(-1, sep), axis=0)
                    test = np.random.multinomial(1, aux)
                    res[i] = cls[test.tolist().index(1)]
                    check = True
                    break
            if not check:
                sep = self.resnbc[0]
                aux = T * np_multivariate_normal_pdf(data[i], self.mu, self.sigma) * backward[i]
                aux = (aux.T / (backward[i - 1] * np.sum(aux))).T[res[i - 1]]
                cls = np.array(range(len(aux))).reshape(-1, sep).tolist()
                cls = [c for c in cls if res[i - 1] in c][0]
                aux = np.sum(aux.reshape(-1, sep), axis=0)
                test = np.random.multinomial(1, aux)
                res[i] = cls[test.tolist().index(1)]

        return np.array(res)

    def simul_visible(self, hidden):
        res = [None] * len(hidden)
        for i in range(0, len(hidden)):
            res[i] = multivariate_normal.rvs(self.mu[hidden[i]], self.sigma[hidden[i]])
        return np.array(res)

    def generate_sample(self, length):
        hidden = [None] * length
        visible = [None] * length
        test = np.random.multinomial(1, np.sum(self.c, axis=1))
        hidden[0] = test.tolist().index(1)
        visible[0] = multivariate_normal.rvs(self.mu[hidden[0]], self.sigma[hidden[0]])
        T = (self.c.T / np.sum(self.c, axis=1)).T
        for i in range(1, length):
            test = np.random.multinomial(1, T[hidden[i - 1], :])
            hidden[i] = test.tolist().index(1)
            visible[i] = multivariate_normal.rvs(self.mu[hidden[i]], self.sigma[hidden[i]])

        return np.array(hidden), np.array(visible)

    def get_forward_backward(self, data):
        forward = [None] * len(data)
        backward = [None] * len(data)
        forward[0] = (np_multivariate_normal_pdf(data[0], self.mu, self.sigma) * np.sum(self.c, axis=1))
        forward[0] = forward[0] / np.sum(forward[0])
        backward[len(data) - 1] = np.array([1] * self.c.shape[0])
        T = (self.c.T / np.sum(self.c, axis=1)).T
        for l in range(1, len(data)):
            k = len(data) - 1 - l

            forward[l] = np.dot(T.T, forward[l - 1]) * np_multivariate_normal_pdf(data[l], self.mu, self.sigma)
            forward[l] = forward[l] / np.sum(forward[l])

            backward[k] = np.dot(T, backward[k + 1] * np_multivariate_normal_pdf(data[k + 1], self.mu, self.sigma))
            backward[k] = backward[k] / np.sum(backward[k])

        return np.array(forward), np.array(backward)

    def get_param_EM(self, data, iter):
        print({'iter': 0, 'c': self.c, 'mu': self.mu, 'sigma': self.sigma})
        for q in range(iter):
            forward, backward = self.get_forward_backward(data)
            T = (self.c.T / np.sum(self.c, axis=1)).T

            aux1 = [(forward[i - 1] * (T * np_multivariate_normal_pdf(data[i], self.mu, self.sigma) * backward[i]).T).T
                    for i in range(1, len(data))]
            epsilon = [a / np.sum(a) for a in aux1]

            aux2 = [(forward[i] * backward[i]) for i in range(len(data))]
            gamma = [a / np.sum(a) for a in aux2]

            self.c = (1 / (len(epsilon))) * sum(epsilon)

            self.mu = sum([np.array(
                [(gamma[i][l]) * data[i] for l in range(self.mu.shape[0])]) for i in
                range(1, len(data))]) / sum(
                [np.array(
                    [np.full((len(data[0])), (gamma[i][l])) for l in range(self.mu.shape[0])]) for i in
                    range(1, len(data))])

            self.sigma = sum([np.array(
                [(gamma[i][l]) * np.outer(data[i] - self.mu[l],
                                          data[i] - self.mu[l])
                 for l in
                 range(self.sigma.shape[0])]) for i in range(1, len(data))]) / sum(
                [np.array([np.full((len(data[0]), len(data[0])), (gamma[i][l])) for l in range(self.sigma.shape[0])])
                 for i
                 in
                 range(1, len(data))])

            print({'iter': q + 1, 'c': self.c, 'mu': self.mu, 'sigma': self.sigma})

    def get_param_ICE(self, data, iter, Nb_simul, modified=False):
        print({'iter': 0, 'c': self.c, 'mu': self.mu, 'sigma': self.sigma})
        for q in range(iter):
            forward, backward = self.get_forward_backward(data)
            T = (self.c.T / np.sum(self.c, axis=1)).T
            aux1 = [(forward[i - 1] * (T * np_multivariate_normal_pdf(data[i], self.mu, self.sigma) * backward[i]).T).T
                    for i in range(1, len(data))]
            epsilon = [a / np.sum(a) for a in aux1]

            self.c = (1 / (len(epsilon))) * sum(epsilon)

            forward, backward = self.get_forward_backward(data)
            if modified:
                hidden_list = [self.simul_hidden_indep(data, forward, backward) for n in range(Nb_simul)]
            else:
                hidden_list = [self.simul_hidden_apost(data, backward) for n in range(Nb_simul)]

            self.mu = sum([sum([np.array(
                [(hidden[i] == l) * data[i] for l in range(self.mu.shape[0])]) for i in range(1, len(data))]) for hidden
                in
                hidden_list]) / sum(
                [sum([np.array(
                    [np.full((len(data[0])), (hidden[i] == l)) for l in
                     range(self.mu.shape[0])]) for i in range(1, len(data))]) for hidden in hidden_list])

            self.sigma = sum([sum([np.array(
                [(hidden[i] == l) * np.outer(data[i] - self.mu[l],
                                             data[i] - self.mu[l]) for l
                 in range(self.sigma.shape[0])]) for i in range(1, len(data))]) for hidden in hidden_list]) / sum(
                [sum([np.array(
                    [np.full((len(data[0]), len(data[0])), (hidden[i] == l)) for
                     l in range(self.sigma.shape[0])]) for i in range(1, len(data))]) for hidden in hidden_list])

            print({'iter': q + 1, 'c': self.c, 'mu': self.mu, 'sigma': self.sigma})

    def get_param_SEM(self, data, iter, modified=False):

        print({'iter': 0, 'c': self.c, 'mu': self.mu, 'sigma': self.sigma})
        for q in range(iter):
            forward, backward = self.get_forward_backward(data)
            if modified:
                hidden = self.simul_hidden_indep_couple(data, forward, backward)
            else:
                hidden = self.simul_hidden_apost(data, backward)

            self.c = (1 / (len(data) - 1)) * sum([np.array(
                [[(hidden[i - 1] == k and hidden[i] == l) for l in range(self.c.shape[1])] for k in
                 range(self.c.shape[0])]) for i in range(1, len(data))])

            self.mu = sum([np.array(
                [(hidden[i] == l) * data[i] for l in range(self.mu.shape[0])]) for i in
                range(1, len(data))]) / sum(
                [np.array(
                    [np.full((len(data[0])), (hidden[i] == l)) for l in range(self.mu.shape[0])]) for i in
                    range(1, len(data))])

            self.sigma = sum([np.array(
                [(hidden[i] == l) * np.outer(data[i] - self.mu[l],
                                             data[i] - self.mu[l])
                 for l in
                 range(self.sigma.shape[0])]) for i in range(1, len(data))]) / sum(
                [np.array([np.full((len(data[0]), len(data[0])), (hidden[i] == l)) for l in range(self.sigma.shape[0])])
                 for i
                 in
                 range(1, len(data))])

            print({'iter': q + 1, 'c': self.c, 'mu': self.mu, 'sigma': self.sigma})

    def get_param_supervised(self, data, hidden):

        self.c = (1 / (len(data) - 1)) * sum([np.array(
            [[(hidden[i - 1] == k and hidden[i] == l) for l in range(self.c.shape[1])] for k in
             range(self.c.shape[0])]) for i in range(1, len(data))])

        self.mu = sum([np.array(
            [(hidden[i] == l) * data[i] for l in range(self.mu.shape[0])]) for i in
            range(1, len(data))]) / sum(
            [np.array(
                [np.full((len(data[0])), (hidden[i] == l)) for l in range(self.mu.shape[0])]) for i in
                range(1, len(data))])

        self.sigma = sum([np.array(
            [(hidden[i] == l) * np.outer(data[i] - self.mu[l],
                                         data[i] - self.mu[l])
             for l in
             range(self.sigma.shape[0])]) for i in range(1, len(data))]) / sum(
            [np.array([np.full((len(data[0]), len(data[0])), (hidden[i] == l)) for l in range(self.sigma.shape[0])])
             for i
             in
             range(1, len(data))])


class HMC_multiR_ctod2:
    __slots__ = ('p', 't', 'mu', 'sigma', 'resoffset', 'resnbc', 'indep')

    def __init__(self, resoffset, resnbc, indep=False, p=None, t=None, mu=None, sigma=None):
        self.p = p
        self.t = t
        self.mu = mu
        self.sigma = sigma
        self.resoffset = resoffset
        self.resnbc = resnbc
        self.indep = indep

    def init_data_prior(self, data, scale=1):
        nb_class = np.prod(self.resnbc)
        self.p = np.array([1 / nb_class] * nb_class)
        a = [np.full((int(np.prod(self.resnbc[i + 1:])) * nb_class, nbc), 1 / (2 * (nbc - 1))) for i, nbc in
             enumerate(self.resnbc)]
        b = [np.block([[np.eye(nbc)] * int((np.prod(self.resnbc[i + 1:]) * nb_class) / nbc)]).T * 1 / (2 * (nbc - 1))
             for
             i, nbc in enumerate(self.resnbc)]
        c = [np.block([[np.eye(nbc)] * int((np.prod(self.resnbc[i + 1:]) * nb_class) / nbc)]).T * 1 / 2 for i, nbc in
             enumerate(self.resnbc)]
        self.t = [a[i] - b[i] + c[i] for i in range(len(a))]
        M = np.mean(data, axis=0)
        Sig = np.cov(data, rowvar=False).reshape(data.shape[1], data.shape[1])
        self.mu = [None] * nb_class
        self.sigma = [None] * nb_class
        for l in range(nb_class):
            if l % 2 == 0:
                self.mu[l] = M - scale * ((l / (nb_class / 2)) * np.sum(Sig, axis=1))
            else:
                self.mu[l] = M + scale * ((l / (nb_class / 2)) * np.sum(Sig, axis=1))
            self.sigma[l] = Sig

        self.mu = np.array(self.mu)
        self.sigma = np.array(self.sigma)

    def get_transition_matrix(self, i):
        T = self.t
        for l in reversed(range(len(self.resoffset))):
            if (i % np.prod(self.resoffset[:l + 1])) == 0:
                T[l + 1] = np.block(
                    [[np.eye(self.t[l + 1].shape[1])] * int(self.t[l + 1].shape[0] / self.t[l + 1].shape[1])]).T
        return calc_product(T)

    def init_kmeans(self, data, labels):
        nb_class = len(np.unique(labels))

        self.c = (1 / (len(data) - 1)) * sum([np.array(
            [[(labels[i - 1] == k and labels[i] == l) for l in range(nb_class)] for k in
             range(nb_class)]) for i in range(1, len(data))])

        self.mu = sum([np.array(
            [(labels[i] == l) * data[i] for l in range(nb_class)]) for i in range(1, len(data))]) / sum(
            [np.array(
                [np.full((len(data[0])), (labels[i] == l)) for l in range(nb_class)]) for i in range(1, len(data))])

        self.sigma = sum([np.array(
            [(labels[i] == l) * np.outer(data[i] - self.mu[l], data[i] - self.mu[l])
             for l in
             range(nb_class)]) for i in range(1, len(data))]) / sum(
            [np.array([np.full((len(data[0]), len(data[0])), (labels[i] == l)) for l in range(nb_class)]) for i in
             range(1, len(data))])

    def seg_map(self, data):
        pass

    def seg_mpm(self, data, res_lvl):
        forward, backward = self.get_forward_backward(data)
        seg_res = self.resnbc[res_lvl]
        marginal_dim = np.prod([x for i, x in enumerate(self.resnbc) if i != res_lvl])
        res = [None] * len(data)
        for i in range(0, len(data)):
            aux = (forward[i] * backward[i])
            aux = aux / np.sum(aux)
            aux = np.sum(aux.reshape((marginal_dim, seg_res)), axis=0)
            res[i] = aux.tolist().index(max(aux))
        return np.array(res)

    def simul_hidden_indep(self, data, forward, backward):
        res = [None] * len(data)
        aux = forward[0] * backward[0]
        test = np.random.multinomial(1, aux / np.sum(aux))
        res[0] = test.tolist().index(1)
        for i in range(1, len(data)):
            check = False
            for l in reversed(range(len(self.resoffset))):
                if (i % np.prod(self.resoffset[:l + 1])) == 0 and not check:
                    sep = np.prod(self.resnbc[:l + 2])
                    cls = np.array(range(len(forward[i]))).reshape(-1, sep).tolist()
                    cls = [c for c in cls if res[i - 1] in c][0]
                    aux = (forward[i] * backward[i])
                    aux = aux / np.sum(aux)
                    aux = np.sum(aux.reshape(-1, sep), axis=0)
                    test = np.random.multinomial(1, aux)
                    res[i] = cls[test.tolist().index(1)]
                    check = True
                    break
            if not check:
                sep = self.resnbc[0]
                cls = np.array(range(len(forward[i]))).reshape(-1, sep).tolist()
                cls = [c for c in cls if res[i - 1] in c][0]
                aux = (forward[i] * backward[i])
                aux = aux / np.sum(aux)
                aux = np.sum(aux.reshape(-1, sep), axis=0)
                test = np.random.multinomial(1, aux)
                res[i] = cls[test.tolist().index(1)]

        return np.array(res)

    def simul_hidden_indep_couple(self, data, forward, backward):
        res = [None] * len(data)
        T = (self.c.T / np.sum(self.c, axis=1)).T
        if (len(data) % 2 == 0):
            for i in range(1, len(data), 2):
                check = False
                for l in reversed(range(len(self.resoffset))):
                    if (i % np.prod(self.resoffset[:l + 1])) == 0 and not check:
                        aux = (forward[i - 1] * (
                                T * np_multivariate_normal_pdf(data[i], self.mu, self.sigma) * backward[i]).T).T
                        aux = aux / np.sum(aux)
                        cls = [a.flatten() for a in np.indices(aux.shape)]
                        test = np.random.multinomial(1, aux[cls[0], cls[1]])
                        res[i - 1] = cls[0][test.tolist().index(1)]
                        res[i] = cls[1][test.tolist().index(1)]
                        check = True
                        break
                if not check:
                    aux = (forward[i - 1] * (
                            T * np_multivariate_normal_pdf(data[i], self.mu, self.sigma) * backward[i]).T).T
                    aux = aux / np.sum(aux)
                    cls = [a.flatten() for a in np.indices(aux.shape)]
                    test = np.random.multinomial(1, aux[cls[0], cls[1]])
                    res[i - 1] = cls[0][test.tolist().index(1)]
                    res[i] = cls[1][test.tolist().index(1)]

        else:
            aux = forward[0] * backward[0]
            test = np.random.multinomial(1, aux / np.sum(aux))
            res[0] = test.tolist().index(1)
            for i in range(2, len(data), 2):
                check = False
                for l in reversed(range(len(self.resoffset))):
                    if (i % np.prod(self.resoffset[:l + 1])) == 0 and not check:
                        aux = (forward[i - 1] * (
                                T * np_multivariate_normal_pdf(data[i], self.mu, self.sigma) * backward[i]).T).T
                        aux = aux / np.sum(aux)
                        cls = [a.flatten() for a in np.indices(aux.shape)]
                        test = np.random.multinomial(1, aux[cls[0], cls[1]])
                        res[i - 1] = cls[0][test.tolist().index(1)]
                        res[i] = cls[1][test.tolist().index(1)]
                        check = True
                        break
                if not check:
                    aux = (forward[i - 1] * (
                            T * np_multivariate_normal_pdf(data[i], self.mu, self.sigma) * backward[i]).T).T
                    aux = aux / np.sum(aux)
                    cls = [a.flatten() for a in np.indices(aux.shape)]
                    test = np.random.multinomial(1, aux[cls[0], cls[1]])
                    res[i - 1] = cls[0][test.tolist().index(1)]
                    res[i] = cls[1][test.tolist().index(1)]

        return np.array(res)

    def simul_hidden_apost(self, data, backward):
        res = [None] * len(data)
        T = (self.c.T / np.sum(self.c, axis=1)).T
        aux = (np_multivariate_normal_pdf(data[0], self.mu, self.sigma) * np.sum(self.c, axis=1))
        test = np.random.multinomial(1, aux / np.sum(aux))
        res[0] = test.tolist().index(1)
        for i in range(1, len(data)):
            check = False
            for l in reversed(range(len(self.resoffset))):
                if (i % np.prod(self.resoffset[:l + 1])) == 0 and not check:
                    sep = np.prod(self.resnbc[:l + 2])
                    aux = T * np_multivariate_normal_pdf(data[i], self.mu, self.sigma) * backward[i]
                    aux = (aux.T / (backward[i - 1] * np.sum(aux))).T[res[i - 1]]
                    cls = np.array(range(len(aux))).reshape(-1, sep).tolist()
                    cls = [c for c in cls if res[i - 1] in c][0]
                    aux = np.sum(aux.reshape(-1, sep), axis=0)
                    test = np.random.multinomial(1, aux)
                    res[i] = cls[test.tolist().index(1)]
                    check = True
                    break
            if not check:
                sep = self.resnbc[0]
                aux = T * np_multivariate_normal_pdf(data[i], self.mu, self.sigma) * backward[i]
                aux = (aux.T / (backward[i - 1] * np.sum(aux))).T[res[i - 1]]
                cls = np.array(range(len(aux))).reshape(-1, sep).tolist()
                cls = [c for c in cls if res[i - 1] in c][0]
                aux = np.sum(aux.reshape(-1, sep), axis=0)
                test = np.random.multinomial(1, aux)
                res[i] = cls[test.tolist().index(1)]

        return np.array(res)

    def simul_visible(self, hidden):
        res = [None] * len(hidden)
        for i in range(0, len(hidden)):
            res[i] = multivariate_normal.rvs(self.mu[hidden[i]], self.sigma[hidden[i]])
        return np.array(res)

    def generate_sample(self, length):
        hidden = [None] * length
        visible = [None] * length
        test = np.random.multinomial(1, np.sum(self.c, axis=1))
        hidden[0] = test.tolist().index(1)
        visible[0] = multivariate_normal.rvs(self.mu[hidden[0]], self.sigma[hidden[0]])
        T = (self.c.T / np.sum(self.c, axis=1)).T
        for i in range(1, length):
            test = np.random.multinomial(1, T[hidden[i - 1], :])
            hidden[i] = test.tolist().index(1)
            visible[i] = multivariate_normal.rvs(self.mu[hidden[i]], self.sigma[hidden[i]])

        return np.array(hidden), np.array(visible)

    def get_forward_backward(self, data):
        forward = [None] * len(data)
        backward = [None] * len(data)
        forward[0] = (np_multivariate_normal_pdf(data[0], self.mu, self.sigma) * self.p)
        forward[0] = forward[0] / np.sum(forward[0])
        backward[len(data) - 1] = np.array([1] * self.c.shape[0])
        for l in range(1, len(data)):
            k = len(data) - 1 - l

            Tf = self.get_transition_matrix(l)
            Tb = self.get_transition_matrix(k)

            forward[l] = np.dot(Tf.T, forward[l - 1]) * np_multivariate_normal_pdf(data[l], self.mu, self.sigma)
            forward[l] = forward[l] / np.sum(forward[l])

            backward[k] = np.dot(Tb, backward[k + 1] * np_multivariate_normal_pdf(data[k + 1], self.mu, self.sigma))
            backward[k] = backward[k] / np.sum(backward[k])

        return np.array(forward), np.array(backward)

    def get_param_EM(self, data, iter):
        print({'iter': 0, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma})
        for q in range(iter):
            forward, backward = self.get_forward_backward(data)

            aux1 = [(forward[i - 1] * (
                    self.get_transition_matrix(i) * np_multivariate_normal_pdf(data[i], self.mu, self.sigma) *
                    backward[i]).T).T
                    for i in range(1, len(data))]
            epsilon = [a / np.sum(a) for a in aux1]

            aux2 = [(forward[i] * backward[i]) for i in range(len(data))]
            gamma = [a / np.sum(a) for a in aux2]

            c = (1 / (len(epsilon))) * sum(epsilon)

            self.mu = sum([np.array(
                [(gamma[i][l]) * data[i] for l in range(self.mu.shape[0])]) for i in
                range(1, len(data))]) / sum(
                [np.array(
                    [np.full((len(data[0])), (gamma[i][l])) for l in range(self.mu.shape[0])]) for i in
                    range(1, len(data))])

            self.sigma = sum([np.array(
                [(gamma[i][l]) * np.outer(data[i] - self.mu[l],
                                          data[i] - self.mu[l])
                 for l in
                 range(self.sigma.shape[0])]) for i in range(1, len(data))]) / sum(
                [np.array([np.full((len(data[0]), len(data[0])), (gamma[i][l])) for l in range(self.sigma.shape[0])])
                 for i
                 in
                 range(1, len(data))])

            print({'iter': q + 1, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma})

    def get_param_ICE(self, data, iter, Nb_simul, modified=False):
        print({'iter': 0, 'c': self.c, 'mu': self.mu, 'sigma': self.sigma})
        for q in range(iter):
            forward, backward = self.get_forward_backward(data)
            T = (self.c.T / np.sum(self.c, axis=1)).T
            aux1 = [(forward[i - 1] * (T * np_multivariate_normal_pdf(data[i], self.mu, self.sigma) * backward[i]).T).T
                    for i in range(1, len(data))]
            epsilon = [a / np.sum(a) for a in aux1]

            self.c = (1 / (len(epsilon))) * sum(epsilon)

            forward, backward = self.get_forward_backward(data)
            if modified:
                hidden_list = [self.simul_hidden_indep(data, forward, backward) for n in range(Nb_simul)]
            else:
                hidden_list = [self.simul_hidden_apost(data, backward) for n in range(Nb_simul)]

            self.mu = sum([sum([np.array(
                [(hidden[i] == l) * data[i] for l in range(self.mu.shape[0])]) for i in range(1, len(data))]) for hidden
                in
                hidden_list]) / sum(
                [sum([np.array(
                    [np.full((len(data[0])), (hidden[i] == l)) for l in
                     range(self.mu.shape[0])]) for i in range(1, len(data))]) for hidden in hidden_list])

            self.sigma = sum([sum([np.array(
                [(hidden[i] == l) * np.outer(data[i] - self.mu[l],
                                             data[i] - self.mu[l]) for l
                 in range(self.sigma.shape[0])]) for i in range(1, len(data))]) for hidden in hidden_list]) / sum(
                [sum([np.array(
                    [np.full((len(data[0]), len(data[0])), (hidden[i] == l)) for
                     l in range(self.sigma.shape[0])]) for i in range(1, len(data))]) for hidden in hidden_list])

            print({'iter': q + 1, 'c': self.c, 'mu': self.mu, 'sigma': self.sigma})

    def get_param_SEM(self, data, iter, modified=False):

        print({'iter': 0, 'c': self.c, 'mu': self.mu, 'sigma': self.sigma})
        for q in range(iter):
            forward, backward = self.get_forward_backward(data)
            if modified:
                hidden = self.simul_hidden_indep_couple(data, forward, backward)
            else:
                hidden = self.simul_hidden_apost(data, backward)

            self.c = (1 / (len(data) - 1)) * sum([np.array(
                [[(hidden[i - 1] == k and hidden[i] == l) for l in range(self.c.shape[1])] for k in
                 range(self.c.shape[0])]) for i in range(1, len(data))])

            self.mu = sum([np.array(
                [(hidden[i] == l) * data[i] for l in range(self.mu.shape[0])]) for i in
                range(1, len(data))]) / sum(
                [np.array(
                    [np.full((len(data[0])), (hidden[i] == l)) for l in range(self.mu.shape[0])]) for i in
                    range(1, len(data))])

            self.sigma = sum([np.array(
                [(hidden[i] == l) * np.outer(data[i] - self.mu[l],
                                             data[i] - self.mu[l])
                 for l in
                 range(self.sigma.shape[0])]) for i in range(1, len(data))]) / sum(
                [np.array([np.full((len(data[0]), len(data[0])), (hidden[i] == l)) for l in range(self.sigma.shape[0])])
                 for i
                 in
                 range(1, len(data))])

            print({'iter': q + 1, 'c': self.c, 'mu': self.mu, 'sigma': self.sigma})

    def get_param_supervised(self, data, hidden):

        self.c = (1 / (len(data) - 1)) * sum([np.array(
            [[(hidden[i - 1] == k and hidden[i] == l) for l in range(self.c.shape[1])] for k in
             range(self.c.shape[0])]) for i in range(1, len(data))])

        self.mu = sum([np.array(
            [(hidden[i] == l) * data[i] for l in range(self.mu.shape[0])]) for i in
            range(1, len(data))]) / sum(
            [np.array(
                [np.full((len(data[0])), (hidden[i] == l)) for l in range(self.mu.shape[0])]) for i in
                range(1, len(data))])

        self.sigma = sum([np.array(
            [(hidden[i] == l) * np.outer(data[i] - self.mu[l],
                                         data[i] - self.mu[l])
             for l in
             range(self.sigma.shape[0])]) for i in range(1, len(data))]) / sum(
            [np.array([np.full((len(data[0]), len(data[0])), (hidden[i] == l)) for l in range(self.sigma.shape[0])])
             for i
             in
             range(1, len(data))])
