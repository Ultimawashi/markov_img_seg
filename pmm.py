import numpy as np
from utils import generate_semipos_sym_mat, np_multivariate_normal_pdf, np_multivariate_normal_pdf_marginal
from scipy.stats import multivariate_normal


class PMC_ctod:

    __slots__ = ('p', 't', 'mu', 'sigma')

    def __init__(self, p=None, t=None, mu=None, sigma=None):
        self.p = p
        self.t = t
        self.mu = mu
        self.sigma = sigma

    def init_kmeans(self, data, labels):
        nb_class = len(np.unique(labels))

        c = (1 / (len(data) - 1)) * sum([np.array(
            [[(labels[i - 1] == k and labels[i] == l) for l in range(nb_class)] for k in
             range(nb_class)]) for i in range(1, len(data))])
        self.p = (1 / (len(data))) * sum([np.array(
            [(labels[i - 1] == k) for k in
             range(nb_class)]) for i in range(len(data))])
        self.t = (c.T / self.p).T

        self.mu = sum([np.array(
            [[(labels[i - 1] == k and labels[i] == l) * data[i - 1:i + 1].flatten() for l in range(nb_class)] for k in
             range(nb_class)]) for i in range(1, len(data))]) / sum([np.array(
            [[np.full((2 * len(data[0])), (labels[i - 1] == k and labels[i] == l)) for l in range(nb_class)] for k in
             range(nb_class)]) for i in range(1, len(data))])

        self.sigma = sum([np.array(
            [[(labels[i - 1] == k and labels[i] == l) * np.outer(data[i - 1:i + 1].flatten() - self.mu[k, l],
                                                                 data[i - 1:i + 1].flatten() - self.mu[k, l]) for l in
              range(nb_class)] for k in
             range(nb_class)]) for i in range(1, len(data))]) / sum([np.array(
            [[np.full((2 * len(data[0]), 2 * len(data[0])), (labels[i - 1] == k and labels[i] == l)) for
              l in range(nb_class)] for k in
             range(nb_class)]) for i in range(1, len(data))])

    def init_data_prior(self, data, nb_class, scale=1):
        self.p = np.array([1 / nb_class] * nb_class)
        a = np.full((nb_class, nb_class), 1 / (2 * (nb_class - 1)))
        a = a - np.diag(np.diag(a))
        self.t = np.diag(np.array([1 / 2] * nb_class)) + a
        data_aux = np.concatenate([data[:-1], data[1:]], axis=1)
        M = np.mean(data_aux, axis=0)
        Sig = np.cov(data_aux, rowvar=False).reshape(data_aux.shape[1], data_aux.shape[1])
        self.mu = [[0 for n in range(nb_class)] for p in range(nb_class)]
        self.sigma = [[0 for n in range(nb_class)] for p in range(nb_class)]
        count = 0
        for l in range(nb_class):
            for k in range(nb_class):
                if count % 2 == 0:
                    self.mu[l][k] = M - scale * ((count / (nb_class ** 2 / 2)) * np.sum(Sig / 2, axis=1))
                else:
                    self.mu[l][k] = M + scale * ((count / (nb_class ** 2 / 2)) * np.sum(Sig / 2, axis=1))
                self.sigma[l][k] = Sig
                count = count + 1

        self.mu = np.array(self.mu)
        self.sigma = np.array(self.sigma)

    def seg_map(self, data):
        pass

    def seg_mpm(self, data):
        forward, backward, _ = self.get_forward_backward(data)
        res = [None] * len(data)
        for i in range(0, len(data)):
            aux = forward[i] * backward[i]
            aux = aux / np.sum(aux)
            res[i] = aux.tolist().index(max(aux))
        return np.array(res)

    def get_estimated_mean_error(self, data):
        forward, backward, _ = self.get_forward_backward(data)
        res = [None] * len(data)
        for i in range(0, len(data)):
            aux = forward[i] * backward[i]
            aux = aux / np.sum(aux)
            res[i] = min(aux)
        return np.sum(np.array(res))/len(data)

    def simul_hidden_indep(self, data, forward, backward):
        res = [None] * len(data)
        for i in range(0, len(data)):
            aux = forward[i] * backward[i]
            test = np.random.multinomial(1, aux / np.sum(aux))
            res[i] = test.tolist().index(1)
        return np.array(res)

    def simul_hidden_indep_couple(self, data, forward, backward, pcond):
        res = [None] * len(data)
        T = self.t
        if (len(data) % 2 == 0):
            for i in range(1, len(data), 2):
                aux = (forward[i - 1] * (pcond[i - 1] * backward[i]).T).T
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
                aux = (forward[i - 1] * (pcond[i - 1] * backward[i]).T).T
                aux = aux / np.sum(aux)
                cls = [a.flatten() for a in np.indices(aux.shape)]
                test = np.random.multinomial(1, aux[cls[0], cls[1]])
                res[i - 1] = cls[0][test.tolist().index(1)]
                res[i] = cls[1][test.tolist().index(1)]

        return np.array(res)

    def simul_hidden_apost(self, data, backward, pcond):
        res = [None] * len(data)
        aux = np.sum(
            np.multiply((self.t.T * self.p).T,
                        np_multivariate_normal_pdf_marginal(data[0], self.mu, self.sigma, (len(data[0]) - 1))),
            axis=1) * backward[0]
        test = np.random.multinomial(1, aux / np.sum(aux))
        res[0] = test.tolist().index(1)
        for i in range(1, len(data)):
            aux = pcond[i - 1] * backward[i]
            aux = (aux.T / (backward[i - 1] * np.sum(aux))).T
            test = np.random.multinomial(1, aux[res[i - 1]] / np.sum(aux[res[i - 1]]))
            res[i] = test.tolist().index(1)
        return np.array(res)

    def simul_visible(self, hidden, dimY):
        res = [None] * len(hidden)
        v = multivariate_normal.rvs(self.mu[hidden[0], hidden[1]], self.sigma[hidden[0], hidden[1]]).reshape(-1, dimY)
        res[0] = v[0]
        res[1] = v[1]
        for i in range(2, len(hidden)):
            mean = self.mu[hidden[i - 1], hidden[i]][dimY:] + self.sigma[hidden[i - 1], hidden[i]][:dimY, dimY:] @ (
                    self.sigma[hidden[i - 1], hidden[i]][dimY:, dimY:] @ np.linalg.inv(
                self.sigma[hidden[i - 1], hidden[i]][:dimY, :dimY])) @ (
                           res[i - 1] - self.mu[hidden[i - 1], hidden[i]][:dimY])
            cov = self.sigma[hidden[i - 1], hidden[i]][dimY:, dimY:] @ np.sqrt(1 - (
                    self.sigma[hidden[i - 1], hidden[i]][:dimY, dimY:] @ self.sigma[hidden[i - 1], hidden[i]][:dimY,
                                                                         dimY:]))
            res[i] = multivariate_normal.rvs(mean, cov)
        return np.array(res)

    def generate_sample(self, length, dimY):
        hidden = [None] * length
        visible = [None] * length
        cls = [a.flatten() for a in np.indices(self.t.shape)]
        test = np.random.multinomial(1, (self.t.T * self.p).T[cls[0], cls[1]])
        hidden[0] = cls[0][test.tolist().index(1)]
        hidden[1] = cls[1][test.tolist().index(1)]
        v = multivariate_normal.rvs(self.mu[hidden[0], hidden[1]], self.sigma[hidden[0], hidden[1]]).reshape(-1, dimY)
        visible[0] = v[0].tolist()
        visible[1] = v[1].tolist()
        for i in range(2, length):
            auxpb = np.multiply((self.t.T * self.p).T,
                                np_multivariate_normal_pdf_marginal(visible[i - 1], self.mu, self.sigma,
                                                                    (dimY - 1)))
            pcond = (auxpb.T / np.sum(auxpb, axis=1)).T
            test = np.random.multinomial(1, pcond[hidden[i - 1], :])
            hidden[i] = test.tolist().index(1)
            mean = self.mu[hidden[i - 1], hidden[i]][dimY:] + self.sigma[hidden[i - 1], hidden[i]][:dimY, dimY:] @ (
                    self.sigma[hidden[i - 1], hidden[i]][dimY:, dimY:] @ np.linalg.inv(
                self.sigma[hidden[i - 1], hidden[i]][:dimY, :dimY])) @ (
                           visible[i - 1] - self.mu[hidden[i - 1], hidden[i]][:dimY])
            cov = self.sigma[hidden[i - 1], hidden[i]][dimY:, dimY:] @ np.sqrt(1 - (
                    self.sigma[hidden[i - 1], hidden[i]][:dimY, dimY:] @ self.sigma[hidden[i - 1], hidden[i]][:dimY,
                                                                         dimY:]))
            visible[i] = multivariate_normal.rvs(mean, cov).reshape(dimY).tolist()

        return np.array(hidden), np.array(visible)

    def get_forward_backward(self, data):
        pcond = [None] * (len(data) - 1)
        forward = [None] * len(data)
        backward = [None] * len(data)
        forward[0] = np.sum(
            np.multiply((self.t.T * self.p).T,
                        np_multivariate_normal_pdf_marginal(data[0], self.mu, self.sigma, (len(data[0]) - 1))),
            axis=1)
        forward[0] = forward[0] / np.sum(forward[0])
        backward[len(data) - 1] = np.array([1] * self.t.shape[0])
        for l in range(1, len(data)):
            k = len(data) - 1 - l
            if pcond[l - 1] is None:
                auxpb = np.multiply((self.t.T * self.p).T,
                                    np_multivariate_normal_pdf(data[l - 1:l + 1].flatten(), self.mu, self.sigma))
                auxpb2 = np.sum(np.multiply((self.t.T * self.p),
                                            np_multivariate_normal_pdf_marginal(data[l - 1], self.mu, self.sigma,
                                                                                (len(data[l - 1]) - 1))), axis=1)
                pcond[l - 1] = (auxpb.T / auxpb2).T

            if pcond[k] is None:
                auxpb = np.multiply((self.t.T * self.p).T,
                                    np_multivariate_normal_pdf(data[k:k + 2].flatten(), self.mu, self.sigma))
                auxpb2 = np.sum(
                    np.multiply((self.t.T * self.p).T, np_multivariate_normal_pdf_marginal(data[k], self.mu, self.sigma,
                                                                                           (len(data[k]) - 1))), axis=1)
                pcond[k] = (auxpb.T / auxpb2).T

            forward[l] = np.dot(pcond[l - 1].T, forward[l - 1])
            forward[l] = forward[l] / np.sum(forward[l])

            backward[k] = np.dot(pcond[k], backward[k + 1])
            backward[k] = backward[k] / np.sum(backward[k])

        return forward, backward, pcond

    def get_param_ICE(self, data, iter, Nb_simul, modified=False):
        print({'iter': 0, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma})
        for q in range(iter):
            forward, backward, pcond = self.get_forward_backward(data)

            aux1 = [(forward[i - 1] * (pcond[i - 1] *
                                       backward[i]).T).T for i in range(1, len(data))]

            epsilon = [a / np.sum(a) for a in aux1]

            aux2 = [(forward[i] * backward[i]) for i in range(len(data))]
            gamma = [a / np.sum(a) for a in aux2]

            self.p = (1 / (len(gamma))) * sum(gamma)

            self.t = (sum(epsilon).T / sum([gamma[i] for i in range(len(epsilon))]).T).T

            if modified:
                hidden_list = [self.simul_hidden_indep(data, forward, backward) for n in range(Nb_simul)]
            else:
                hidden_list = [self.simul_hidden_apost(data, backward, pcond) for n in range(Nb_simul)]

            aux_mu = sum([sum([np.array(
                [[(hidden[i - 1] == k and hidden[i] == l) * data[i - 1:i + 1].flatten() for l in
                  range(self.mu.shape[1])] for k in
                 range(self.mu.shape[0])]) for i in range(1, len(data))]) for hidden in hidden_list]) / sum(
                [sum([np.array(
                    [[np.full((2 * len(data[0])), (hidden[i - 1] == k and hidden[i] == l)) for l in
                      range(self.mu.shape[1])] for k in
                     range(self.mu.shape[0])]) for i in range(1, len(data))]) for hidden in hidden_list])

            aux_sigma = sum([sum([np.array(
                [[(hidden[i - 1] == k and hidden[i] == l) * np.outer(data[i - 1:i + 1].flatten() - aux_mu[k, l],
                                                                     data[i - 1:i + 1].flatten() - aux_mu[k, l]) for l
                  in
                  range(self.sigma.shape[1])] for k in
                 range(self.sigma.shape[0])]) for i in range(1, len(data))]) for hidden in hidden_list]) / sum(
                [sum([np.array(
                    [[np.full((2 * len(data[0]), 2 * len(data[0])), (hidden[i - 1] == k and hidden[i] == l)) for
                      l in range(self.sigma.shape[1])] for k in
                     range(self.sigma.shape[0])]) for i in range(1, len(data))]) for hidden in hidden_list])

            self.mu[np.all(np.isfinite(aux_mu), axis=(-1))] = aux_mu[np.all(np.isfinite(aux_mu), axis=(-1))]
            self.sigma[np.logical_and(np.logical_and(np.all(np.isfinite(aux_sigma), axis=(-2, -1)),
                                                     np.invert(np.all(aux_sigma == 0, axis=(-2, -1)))),
                                      (np.linalg.det(aux_sigma) > 0))] = aux_sigma[
                np.logical_and(np.logical_and(np.all(np.isfinite(aux_sigma), axis=(-2, -1)),
                                              np.invert(np.all(aux_sigma == 0, axis=(-2, -1)))),
                               (np.linalg.det(aux_sigma) > 0))]

            print({'iter': q + 1, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma})

    def get_param_SEM(self, data, iter, modified=False):
        print({'iter': 0, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma})
        for q in range(iter):
            forward, backward, pcond = self.get_forward_backward(data)

            if modified:
                hidden = self.simul_hidden_indep_couple(data, forward, backward, pcond)
            else:
                hidden = self.simul_hidden_apost(data, backward, pcond)

            c = (1 / (len(data) - 1)) * sum([np.array(
                [[(hidden[i - 1] == k and hidden[i] == l) for l in range(self.t.shape[1])] for k in
                 range(self.t.shape[0])]) for i in range(1, len(data))])
            self.p = (1 / (len(data))) * sum([np.array(
                [(hidden[i] == l) for l in range(len(self.p))]) for i in range(len(data))])
            self.t = (c.T / self.p).T

            aux_mu = sum([np.array(
                [[(hidden[i - 1] == k and hidden[i] == l) * data[i - 1:i + 1].flatten() for l in
                  range(self.mu.shape[1])] for k in
                 range(self.mu.shape[0])]) for i in range(1, len(data))]) / sum([np.array(
                [[np.full((2 * len(data[0])), (hidden[i - 1] == k and hidden[i] == l)) for l in range(self.mu.shape[1])]
                 for k in
                 range(self.mu.shape[0])]) for i in range(1, len(data))])

            aux_sigma = sum([np.array(
                [[(hidden[i - 1] == k and hidden[i] == l) * np.outer(data[i - 1:i + 1].flatten() - aux_mu[k, l],
                                                                     data[i - 1:i + 1].flatten() - aux_mu[k, l]) for l
                  in
                  range(self.sigma.shape[1])] for k in
                 range(self.sigma.shape[0])]) for i in range(1, len(data))]) / sum([np.array(
                [[np.full((2 * len(data[0]), 2 * len(data[0])), (hidden[i - 1] == k and hidden[i] == l)) for
                  l in range(self.sigma.shape[1])] for k in
                 range(self.sigma.shape[0])]) for i in range(1, len(data))])

            self.mu[np.all(np.isfinite(aux_mu), axis=(-1))] = aux_mu[np.all(np.isfinite(aux_mu), axis=(-1))]
            self.sigma[np.logical_and(np.logical_and(np.all(np.isfinite(aux_sigma), axis=(-2, -1)),
                                                     np.invert(np.all(aux_sigma == 0, axis=(-2, -1)))),
                                      (np.linalg.det(aux_sigma) > 0))] = aux_sigma[
                np.logical_and(np.logical_and(np.all(np.isfinite(aux_sigma), axis=(-2, -1)),
                                              np.invert(np.all(aux_sigma == 0, axis=(-2, -1)))),
                               (np.linalg.det(aux_sigma) > 0))]

            print({'iter': q + 1, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma})

    def get_param_supervised(self, data, hidden):

        c = (1 / (len(data) - 1)) * sum([np.array(
            [[(hidden[i - 1] == k and hidden[i] == l) for l in range(self.t.shape[1])] for k in
             range(self.t.shape[0])]) for i in range(1, len(data))])
        self.p = (1 / (len(data))) * sum([np.array(
            [(hidden[i] == l) for l in range(len(self.p))]) for i in range(len(data))])
        self.t = (c.T / self.p).T

        self.mu = sum([np.array(
            [[(hidden[i - 1] == k and hidden[i] == l) * data[i - 1:i + 1].flatten() for l in
              range(self.mu.shape[1])] for k in
             range(self.mu.shape[0])]) for i in range(1, len(data))]) / sum([np.array(
            [[np.full((2 * len(data[0])), (hidden[i - 1] == k and hidden[i] == l)) for l in range(self.mu.shape[1])]
             for k in
             range(self.mu.shape[0])]) for i in range(1, len(data))])

        self.sigma = sum([np.array(
            [[(hidden[i - 1] == k and hidden[i] == l) * np.outer(data[i - 1:i + 1].flatten() - self.mu[k, l],
                                                                 data[i - 1:i + 1].flatten() - self.mu[k, l]) for l
              in
              range(self.sigma.shape[1])] for k in
             range(self.sigma.shape[0])]) for i in range(1, len(data))]) / sum([np.array(
            [[np.full((2 * len(data[0]), 2 * len(data[0])), (hidden[i - 1] == k and hidden[i] == l)) for
              l in range(self.sigma.shape[1])] for k in
             range(self.sigma.shape[0])]) for i in range(1, len(data))])


class PSMC_ctod:
    __slots__ = ('p', 't', 'mu', 'sigma', 'nbc_x', 'nbc_u', 'tps_min')

    def __init__(self, nbc_x, nbc_u, p=None, t=None, mu=None, sigma=None, tps_min=False):
        self.p = p
        self.t = t
        self.mu = mu
        self.sigma = sigma
        self.nbc_x = nbc_x
        self.nbc_u = nbc_u
        self.tps_min = tps_min

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
        if self.tps_min:
            ut = [[np.eye(self.nbc_u, k=1) for n1 in range(self.nbc_x)] for n2 in range(int(nb_class / self.nbc_u))]
            for e in ut:
                for a in e:
                    aux = np.zeros((len(a[-1])))
                    aux[0] = 1
                    a[-1] = aux
            ut = np.block(ut)
            for i in range(1, self.nbc_x + 1):
                ut[:, (i - 1) * self.nbc_u:i * self.nbc_u] = (
                        ut[:, (i - 1) * self.nbc_u:i * self.nbc_u].T * b[:, i - 1]).T
        else:
            a = u
            ut = [[np.eye(self.nbc_u, k=1) for n1 in range(self.nbc_x)] for n2 in range(int(nb_class / self.nbc_u))]
            for i,e in enumerate(ut):
                for j,p in enumerate(e):
                    p[-1] = a[i,j]
            ut = np.block(ut)
            for i in range(1, self.nbc_x + 1):
                ut[:, (i - 1) * self.nbc_u:i * self.nbc_u] = (
                        ut[:, (i - 1) * self.nbc_u:i * self.nbc_u].T * b[:, i - 1]).T
        self.t = ut
        data_aux = np.concatenate([data[:-1], data[1:]], axis=1)
        M = np.mean(data_aux, axis=0)
        Sig = np.cov(data_aux, rowvar=False).reshape(data_aux.shape[1], data_aux.shape[1])
        self.mu = [[0 for n in range(self.nbc_x)] for p in range(self.nbc_x)]
        self.sigma = [[0 for n in range(self.nbc_x)] for p in range(self.nbc_x)]
        count = 0
        for l in range(self.nbc_x):
            for k in range(self.nbc_x):
                if count % 2 == 0:
                    self.mu[l][k] = M - scale * ((count / (self.nbc_x ** 2 / 2)) * np.sum(Sig / 2, axis=1))
                else:
                    self.mu[l][k] = M + scale * ((count / (self.nbc_x ** 2 / 2)) * np.sum(Sig / 2, axis=1))
                self.sigma[l][k] = Sig
                count = count + 1

        self.mu = np.array(self.mu)
        self.sigma = np.array(self.sigma)

    def init_kmeans(self, data, labels):
        hidden = labels
        c = (1 / (len(data) - 1)) * sum([np.array(
            [[(hidden[i - 1] == k and hidden[i] == l) for l in range(self.t.shape[1])] for k in
             range(self.t.shape[0])]) for i in range(1, len(data))])

        nb_class = self.nbc_x * self.nbc_u
        pu = np.array([1 / self.nbc_u] * self.nbc_u)
        self.p = np.outer(np.sum(c, axis=1), pu).flatten()
        self.p = self.p / np.sum(self.p)
        a = (c.T / np.sum(c, axis=1)).T
        b = np.repeat(np.eye(self.nbc_x, self.nbc_x, k=0), self.nbc_u, axis=0)
        idx = [i for i in range(b.shape[0]) if ((i + 1) % self.nbc_u == 0)]
        for i, e in enumerate(idx):
            b[e] = a[i]
        if self.tps_min:
            ut = [[np.eye(self.nbc_u, k=1) for n1 in range(self.nbc_x)] for n2 in range(int(nb_class / self.nbc_u))]
            for e in ut:
                for a in e:
                    aux = np.zeros((len(a[-1])))
                    aux[0] = 1
                    a[-1] = aux
            ut = np.block(ut)
            for i in range(1, self.nbc_x + 1):
                ut[:, (i - 1) * self.nbc_u:i * self.nbc_u] = (
                        ut[:, (i - 1) * self.nbc_u:i * self.nbc_u].T * b[:, i - 1]).T
        else:
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
            [[(hidden[i - 1] == k and hidden[i] == l) * data[i - 1:i + 1].flatten() for l in
              range(self.mu.shape[1])] for k in
             range(self.mu.shape[0])]) for i in range(1, len(data))]) / sum([np.array(
            [[np.full((2 * len(data[0])), (hidden[i - 1] == k and hidden[i] == l)) for l in range(self.mu.shape[1])]
             for k in
             range(self.mu.shape[0])]) for i in range(1, len(data))])

        self.sigma = sum([np.array(
            [[((hidden[i - 1]) == k and (hidden[i]) == l) * np.outer(
                data[i - 1:i + 1].flatten() - self.mu[k, l],
                data[i - 1:i + 1].flatten() - self.mu[k, l]) for l
              in
              range(self.sigma.shape[1])] for k in
             range(self.sigma.shape[0])]) for i in range(1, len(data))]) / sum([np.array(
            [[np.full((2 * len(data[0]), 2 * len(data[0])),
                      ((hidden[i - 1]) == k and (hidden[i]) == l)) for
              l in range(self.sigma.shape[1])] for k in
             range(self.sigma.shape[0])]) for i in range(1, len(data))])

    def give_param(self, c, u, mu, sigma):
        nb_class = self.nbc_x * self.nbc_u
        self.p = (np.sum(c * u.T, axis=1)).T.flatten()
        a = (c.T / np.sum(c, axis=1)).T
        b = np.repeat(np.eye(self.nbc_x, self.nbc_x, k=0), self.nbc_u, axis=0)
        idx = [i for i in range(b.shape[0]) if ((i + 1) % self.nbc_u == 0)]
        for i, e in enumerate(idx):
            b[e] = a[i]
        if self.tps_min:
            ut = [[np.eye(self.nbc_u, k=1) for n1 in range(self.nbc_x)] for n2 in range(int(nb_class / self.nbc_u))]

            for e in ut:
                for a in e:
                    aux = np.zeros((len(a[-1])))
                    aux[0] = 1
                    a[-1] = aux
            ut = np.block(ut)

            for i in range(1, self.nbc_x + 1):
                ut[:, (i - 1) * self.nbc_u:i * self.nbc_u] = (
                        ut[:, (i - 1) * self.nbc_u:i * self.nbc_u].T * b[:, i - 1]).T
        else:
            a = u
            ut = [[np.eye(self.nbc_u, k=1) for n1 in range(self.nbc_x)] for n2 in range(int(nb_class / self.nbc_u))]
            for i,e in enumerate(ut):
                for j,p in enumerate(e):
                    p[-1] = a[i,j]
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
        forward, backward, _ = self.get_forward_backward(data)
        res = [None] * len(data)
        for i in range(0, len(data)):
            aux = (forward[i] * backward[i])
            aux = aux / np.sum(aux)
            aux = np.sum(aux.reshape((self.nbc_x, self.nbc_u)), axis=1)
            res[i] = aux.tolist().index(max(aux))
        return np.array(res)

    def simul_hidden_indep(self, data, forward, backward):
        res = [None] * len(data)
        for i in range(0, len(data)):
            aux = forward[i] * backward[i]
            test = np.random.multinomial(1, aux / np.sum(aux))
            res[i] = test.tolist().index(1)
        return np.array(res)

    def simul_hidden_indep_couple(self, data, forward, backward, pcond):
        res = [None] * len(data)
        T = self.t
        if (len(data) % 2 == 0):
            for i in range(1, len(data), 2):
                aux = (forward[i - 1] * (pcond[i - 1] * backward[i]).T).T
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
                aux = (forward[i - 1] * (pcond[i - 1] * backward[i]).T).T
                aux = aux / np.sum(aux)
                cls = [a.flatten() for a in np.indices(aux.shape)]
                test = np.random.multinomial(1, aux[cls[0], cls[1]])
                res[i - 1] = cls[0][test.tolist().index(1)]
                res[i] = cls[1][test.tolist().index(1)]

        return np.array(res)

    def simul_hidden_apost(self, data, backward, pcond):
        res = [None] * len(data)
        mu = np.repeat(np.repeat(self.mu, self.nbc_u, axis=0), self.nbc_u, axis=1)
        sigma = np.repeat(np.repeat(self.sigma, self.nbc_u, axis=0), self.nbc_u, axis=1)
        aux = np.sum(
            np.multiply((self.t.T * self.p).T,
                        np_multivariate_normal_pdf_marginal(data[0], mu, sigma, (len(data[0]) - 1))),
            axis=1) * backward[0]
        test = np.random.multinomial(1, aux / np.sum(aux))
        res[0] = test.tolist().index(1)
        for i in range(1, len(data)):
            aux = pcond[i - 1] * backward[i]
            aux = (aux.T / (backward[i - 1] * np.sum(aux))).T
            test = np.random.multinomial(1, aux[res[i - 1]] / np.sum(aux[res[i - 1]]))
            res[i] = test.tolist().index(1)
        return np.array(res)

    def simul_visible(self, hidden, dimY):
        res = [None] * len(hidden)
        mu = np.repeat(np.repeat(self.mu, self.nbc_u, axis=0), self.nbc_u, axis=1)
        sigma = np.repeat(np.repeat(self.sigma, self.nbc_u, axis=0), self.nbc_u, axis=1)
        v = multivariate_normal.rvs(mu[hidden[0], hidden[1]], sigma[hidden[0], hidden[1]]).reshape(-1, dimY)
        res[0] = v[0]
        res[1] = v[1]
        for i in range(2, len(hidden)):
            mean = mu[hidden[i - 1], hidden[i]][dimY:] + sigma[hidden[i - 1], hidden[i]][:dimY, dimY:] @ (
                    sigma[hidden[i - 1], hidden[i]][dimY:, dimY:] @ np.linalg.inv(
                sigma[hidden[i - 1], hidden[i]][:dimY, :dimY])) @ (
                           res[i - 1] - mu[hidden[i - 1], hidden[i]][:dimY])
            cov = sigma[hidden[i - 1], hidden[i]][dimY:, dimY:] @ np.sqrt(1 - (
                    sigma[hidden[i - 1], hidden[i]][:dimY, dimY:] @ sigma[hidden[i - 1], hidden[i]][:dimY,
                                                                    dimY:]))
            res[i] = multivariate_normal.rvs(mean, cov)
        return np.array(res)

    def generate_sample(self, length, dimY):
        hidden = [None] * length
        visible = [None] * length
        mu = np.repeat(np.repeat(self.mu, self.nbc_u, axis=0), self.nbc_u, axis=1)
        sigma = np.repeat(np.repeat(self.sigma, self.nbc_u, axis=0), self.nbc_u, axis=1)
        cls = [a.flatten() for a in np.indices(self.t.shape)]
        test = np.random.multinomial(1, (self.t.T * self.p).T[cls[0], cls[1]])
        hidden[0] = cls[0][test.tolist().index(1)]
        hidden[1] = cls[1][test.tolist().index(1)]
        v = multivariate_normal.rvs(mu[hidden[0], hidden[1]], sigma[hidden[0], hidden[1]]).reshape(-1, dimY)
        visible[0] = v[0].tolist()
        visible[1] = v[1].tolist()
        for i in range(2, length):
            auxpb = np.multiply((self.t.T * self.p).T,
                                np_multivariate_normal_pdf_marginal(visible[i - 1], mu, sigma,
                                                                    (dimY - 1)))
            pcond = (auxpb.T / np.sum(auxpb, axis=1)).T
            test = np.random.multinomial(1, pcond[hidden[i - 1], :])
            hidden[i] = test.tolist().index(1)
            mean = mu[hidden[i - 1], hidden[i]][dimY:] + sigma[hidden[i - 1], hidden[i]][:dimY, dimY:] @ (
                    sigma[hidden[i - 1], hidden[i]][dimY:, dimY:] @ np.linalg.inv(
                sigma[hidden[i - 1], hidden[i]][:dimY, :dimY])) @ (
                           visible[i - 1] - mu[hidden[i - 1], hidden[i]][:dimY])
            cov = sigma[hidden[i - 1], hidden[i]][dimY:, dimY:] @ np.sqrt(1 - (
                    sigma[hidden[i - 1], hidden[i]][:dimY, dimY:] @ sigma[hidden[i - 1], hidden[i]][:dimY,
                                                                    dimY:]))
            visible[i] = multivariate_normal.rvs(mean, cov).reshape(dimY).tolist()

        return np.array(hidden), np.array(visible)

    def get_forward_backward(self, data):
        pcond = [None] * (len(data) - 1)
        forward = [None] * len(data)
        backward = [None] * len(data)
        mu = np.repeat(np.repeat(self.mu, self.nbc_u, axis=0), self.nbc_u, axis=1)
        sigma = np.repeat(np.repeat(self.sigma, self.nbc_u, axis=0), self.nbc_u, axis=1)
        forward[0] = np.sum(
            np.multiply((self.t.T * self.p).T,
                        np_multivariate_normal_pdf_marginal(data[0], mu, sigma, (len(data[0]) - 1))),
            axis=1)
        forward[0] = forward[0] / np.sum(forward[0])
        backward[len(data) - 1] = np.array([1] * self.t.shape[0])
        for l in range(1, len(data)):
            k = len(data) - 1 - l
            if pcond[l - 1] is None:
                auxpb = np.multiply((self.t.T * self.p).T,
                                    np_multivariate_normal_pdf(data[l - 1:l + 1].flatten(), mu, sigma))
                auxpb2 = np.sum(np.multiply((self.t.T * self.p),
                                            np_multivariate_normal_pdf_marginal(data[l - 1], mu, sigma,
                                                                                (len(data[l - 1]) - 1))), axis=1)
                pcond[l - 1] = (auxpb.T / auxpb2).T

            if pcond[k] is None:
                auxpb = np.multiply((self.t.T * self.p).T,
                                    np_multivariate_normal_pdf(data[k:k + 2].flatten(), mu, sigma))
                auxpb2 = np.sum(
                    np.multiply((self.t.T * self.p).T, np_multivariate_normal_pdf_marginal(data[k], mu, sigma,
                                                                                           (len(data[k]) - 1))), axis=1)
                pcond[k] = (auxpb.T / auxpb2).T

            forward[l] = np.dot(pcond[l - 1].T, forward[l - 1])
            forward[l] = forward[l] / np.sum(forward[l])

            backward[k] = np.dot(pcond[k], backward[k + 1])
            backward[k] = backward[k] / np.sum(backward[k])

        return forward, backward, pcond

    def get_param_ICE(self, data, iter, Nb_simul, modified=False):
        print({'iter': 0, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma})
        for q in range(iter):
            forward, backward, pcond = self.get_forward_backward(data)

            aux1 = [(forward[i - 1] * (pcond[i - 1] *
                                       backward[i]).T).T for i in range(1, len(data))]

            epsilon = [a / np.sum(a) for a in aux1]

            aux2 = [(forward[i] * backward[i]) for i in range(len(data))]
            gamma = [a / np.sum(a) for a in aux2]

            self.p = (1 / (len(gamma))) * sum(gamma)

            self.t = (sum(epsilon).T / sum([gamma[i] for i in range(len(epsilon))]).T).T

            if modified:
                hidden_list = [self.simul_hidden_indep(data, forward, backward) for n in range(Nb_simul)]
            else:
                hidden_list = [self.simul_hidden_apost(data, backward, pcond) for n in range(Nb_simul)]

            aux_mu = sum([sum([np.array(
                [[((hidden[i - 1] // self.nbc_u) == k and (hidden[i] // self.nbc_u) == l) * data[i - 1:i + 1].flatten()
                  for l in
                  range(self.mu.shape[1])] for k in
                 range(self.mu.shape[0])]) for i in range(1, len(data))]) for hidden in hidden_list]) / sum(
                [sum([np.array(
                    [[np.full((2 * len(data[0])),
                              ((hidden[i - 1] // self.nbc_u) == k and (hidden[i] // self.nbc_u) == l)) for l in
                      range(self.mu.shape[1])] for k in
                     range(self.mu.shape[0])]) for i in range(1, len(data))]) for hidden in hidden_list])

            aux_sigma = sum([sum([np.array(
                [[((hidden[i - 1] // self.nbc_u) == k and (hidden[i] // self.nbc_u) == l) * np.outer(
                    data[i - 1:i + 1].flatten() - aux_mu[k, l],
                    data[i - 1:i + 1].flatten() - aux_mu[k, l]) for l
                  in
                  range(self.sigma.shape[1])] for k in
                 range(self.sigma.shape[0])]) for i in range(1, len(data))]) for hidden in hidden_list]) / sum(
                [sum([np.array(
                    [[np.full((2 * len(data[0]), 2 * len(data[0])),
                              ((hidden[i - 1] // self.nbc_u) == k and (hidden[i] // self.nbc_u) == l)) for
                      l in range(self.sigma.shape[1])] for k in
                     range(self.sigma.shape[0])]) for i in range(1, len(data))]) for hidden in hidden_list])

            self.mu[np.all(np.isfinite(aux_mu), axis=(-1))] = aux_mu[np.all(np.isfinite(aux_mu), axis=(-1))]
            self.sigma[np.logical_and(np.logical_and(np.all(np.isfinite(aux_sigma), axis=(-2, -1)),
                                                     np.invert(np.all(aux_sigma == 0, axis=(-2, -1)))),
                                      (np.linalg.det(aux_sigma) > 0))] = aux_sigma[
                np.logical_and(np.logical_and(np.all(np.isfinite(aux_sigma), axis=(-2, -1)),
                                              np.invert(np.all(aux_sigma == 0, axis=(-2, -1)))),
                               (np.linalg.det(aux_sigma) > 0))]

            print({'iter': q + 1, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma})

    def get_param_SEM(self, data, iter, modified=False):
        print({'iter': 0, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma})
        for q in range(iter):
            forward, backward, pcond = self.get_forward_backward(data)

            if modified:
                hidden = self.simul_hidden_indep_couple(data, forward, backward, pcond)
            else:
                hidden = self.simul_hidden_apost(data, backward, pcond)

            c = (1 / (len(data) - 1)) * sum([np.array(
                [[(hidden[i - 1] == k and hidden[i] == l) for l in range(self.t.shape[1])] for k in
                 range(self.t.shape[0])]) for i in range(1, len(data))])
            self.p = (1 / (len(data))) * sum([np.array(
                [(hidden[i] == l) for l in range(len(self.p))]) for i in range(len(data))])
            self.t = (c.T / self.p).T

            aux_mu = sum([np.array(
                [[((hidden[i - 1] // self.nbc_u) == k and (hidden[i] // self.nbc_u) == l) * data[i - 1:i + 1].flatten()
                  for l in
                  range(self.mu.shape[1])] for k in
                 range(self.mu.shape[0])]) for i in range(1, len(data))]) / sum([np.array(
                [[np.full((2 * len(data[0])), ((hidden[i - 1] // self.nbc_u) == k and (hidden[i] // self.nbc_u) == l))
                  for l in range(self.mu.shape[1])]
                 for k in
                 range(self.mu.shape[0])]) for i in range(1, len(data))])

            aux_sigma = sum([np.array(
                [[((hidden[i - 1] // self.nbc_u) == k and (hidden[i] // self.nbc_u) == l) * np.outer(
                    data[i - 1:i + 1].flatten() - aux_mu[k, l],
                    data[i - 1:i + 1].flatten() - aux_mu[k, l]) for l
                  in
                  range(self.sigma.shape[1])] for k in
                 range(self.sigma.shape[0])]) for i in range(1, len(data))]) / sum([np.array(
                [[np.full((2 * len(data[0]), 2 * len(data[0])),
                          ((hidden[i - 1] // self.nbc_u) == k and (hidden[i] // self.nbc_u) == l)) for
                  l in range(self.sigma.shape[1])] for k in
                 range(self.sigma.shape[0])]) for i in range(1, len(data))])

            self.mu[np.all(np.isfinite(aux_mu), axis=(-1))] = aux_mu[np.all(np.isfinite(aux_mu), axis=(-1))]
            self.sigma[np.logical_and(np.logical_and(np.all(np.isfinite(aux_sigma), axis=(-2, -1)),
                                                     np.invert(np.all(aux_sigma == 0, axis=(-2, -1)))),
                                      (np.linalg.det(aux_sigma) > 0))] = aux_sigma[
                np.logical_and(np.logical_and(np.all(np.isfinite(aux_sigma), axis=(-2, -1)),
                                              np.invert(np.all(aux_sigma == 0, axis=(-2, -1)))),
                               (np.linalg.det(aux_sigma) > 0))]

            print({'iter': q + 1, 'p': self.p, 't': self.t, 'mu': self.mu, 'sigma': self.sigma})
