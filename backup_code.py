import numpy as np
from scipy.stats import norm


def ln_sum(a,b):
    return np.maximum(a,b) + np.log(1 + np.exp(np.minimum(b-a,a-b)))

def ln_sum_np(iterable):
    res = np.NINF
    for i in range(len(iterable)):
        res = ln_sum(res,iterable[i])
    return res

def gen_hmc_dtod(l, p1, a, b):
    test = np.random.multinomial(1, p1)
    aux = [None] * l
    aux[0] = test.tolist().index(1)
    for i in range(1, l):
        test = np.random.multinomial(1, a[aux[i - 1], :])
        aux[i] = test.tolist().index(1)
    return np.array(aux), np.array([np.random.multinomial(1, b[a, :]).tolist().index(1) for a in aux])

def init_hmc_rand_prior(nb_class, dimY, dirhyperparams, mean_bound, var_bound):
    p1 = np.random.dirichlet(dirhyperparams, size=1)[0]
    a = np.random.dirichlet(dirhyperparams, size=nb_class)
    b = np.full((nb_class), None)
    for i in range(len(b)):
        b[i] = {'mu': np.random.rand(dimY) * (mean_bound[1] - mean_bound[0]) + mean_bound[0],
                'sigma': generate_semipos_sym_mat((dimY, dimY), var_bound)}

    return p1, a, b


def init_hmc_kmeans(data, labels):
    nb_class = len(np.unique(labels))
    p1 = np.zeros((nb_class))
    a = np.zeros((nb_class, nb_class))
    b = np.full((nb_class), None)
    for i in range(len(b)):
        b[i] = {'mu': np.zeros((len(data[0]))), 'sigma': np.zeros((len(data[0]), len(data[0])))}

    c = (1 / (len(data) - 1)) * sum([np.array(
        [[(labels[i - 1] == k and labels[i] == l) for l in range(a.shape[1])] for k in
         range(a.shape[0])]) for i in range(1, len(data))])
    gamma = (1 / (len(data) - 1)) * sum(
        [np.array([labels[i] == l for l in range(a.shape[0])]) for i in range(1, len(data))])
    p1 = np.array([labels[0] == l for l in range(a.shape[0])]).astype(int)
    a = (c.T / gamma).T

    mu_aux = sum([np.array(
        [(labels[i] == l) * data[i] for l in range(len(b))]) for i in range(1, len(data))]) / sum(
        [np.array(
            [np.full((len(data[0])), (labels[i] == l)) for l in range(len(b))]) for i in range(1, len(data))])

    sigma_aux = sum([np.array(
        [(labels[i] == l) * np.outer(data[i] - mu_aux[l], data[i] - mu_aux[l])
         for l in
         range(len(b))]) for i in range(1, len(data))]) / sum(
        [np.array([np.full((len(data[0]), len(data[0])), (labels[i] == l)) for l in range(len(b))]) for i in
         range(1, len(data))])

    for i in range(len(b)):
        b[i]['mu'] = mu_aux[i]
        b[i]['sigma'] = sigma_aux[i]
    return p1, a, b


class HMC_ctod2:
    __slots__ = ('p1', 'a', 'b')

    def __init__(self, p1=None, a=None, b=None):
        self.p1 = p1
        self.a = a
        self.b = b

    def init_kmeans(self, data, labels):
        nb_class = len(np.unique(labels))
        b = np.full((nb_class), None)
        c = np.zeros((nb_class, nb_class))
        for i in range(len(b)):
            b[i] = {'mu': np.zeros((len(data[0]))), 'sigma': np.zeros((len(data[0]), len(data[0])))}

        c = (1 / (len(data) - 1)) * sum([np.array(
            [[(labels[i - 1] == k and labels[i] == l) for l in range(c.shape[1])] for k in
             range(c.shape[0])]) for i in range(1, len(data))])
        gamma = (1 / (len(data) - 1)) * sum(
            [np.array([labels[i] == l for l in range(c.shape[0])]) for i in range(1, len(data))])


        mu_aux = sum([np.array(
            [(labels[i] == l) * data[i] for l in range(len(b))]) for i in range(1, len(data))]) / sum(
            [np.array(
                [np.full((len(data[0])), (labels[i] == l)) for l in range(len(b))]) for i in range(1, len(data))])

        sigma_aux = sum([np.array(
            [(labels[i] == l) * np.outer(data[i] - mu_aux[l], data[i] - mu_aux[l])
             for l in
             range(len(b))]) for i in range(1, len(data))]) / sum(
            [np.array([np.full((len(data[0]), len(data[0])), (labels[i] == l)) for l in range(len(b))]) for i in
             range(1, len(data))])

        for i in range(len(b)):
            b[i]['mu'] = mu_aux[i]
            b[i]['sigma'] = sigma_aux[i]

        self.p1=gamma
        self.a = (c.T / gamma).T
        self.b = b

    def seg_map(self, data):
        pass

    def seg_mpm(self, data):
        forward, backward = self.get_forward_backward(data)
        res = [None] * len(data)
        for i in range(0, len(data)):
            aux = forward[i] * backward[i]
            aux = aux / np.sum(aux)
            res[i] = aux.tolist().index(max(aux))
        return np.array(res)

    def simul_hidden(self, data, forward, backward):
        res = [None] * len(data)
        for i in range(0, len(data)):
            aux = forward[i] * backward[i]
            test = np.random.multinomial(1, aux / np.sum(aux))
            res[i] = test.tolist().index(1)
        return np.array(res)

    def get_forward_backward(self, data):
        forward = [None] * len(data)
        backward = [None] * len(data)
        forward[0] = (multi_norm_pdf_v(data=data[0], x=self.b) * self.p1)
        forward[0] = forward[0] / np.sum(forward[0])
        backward[len(data) - 1] = np.array([1] * self.a.shape[0])

        for l in range(1, len(data)):
            k = len(data) - 1 - l
            forward[l] = (np.sum((forward[l - 1] * self.a.T).T, axis=0) * multi_norm_pdf_v(data=data[l], x=self.b))
            forward[l] = forward[l] / np.sum(forward[l])
            backward[k] = np.sum(
                backward[k + 1] * self.a * multi_norm_pdf_v(data=data[k + 1], x=self.b),
                axis=1)
            backward[k] = backward[k] / np.sum(backward[k])

        return forward, backward

    def get_param_EM(self, data, iter):
        print({'iter': 0, 'p1': self.p1, 'a': self.a, 'b': self.b})
        for q in range(iter):
            forward, backward = self.get_forward_backward(data)
            aux1 = [(forward[i - 1] * (self.a * multi_norm_pdf_v(data=data[i], x=self.b) * backward[i]).T).T for i in
                    range(1, len(data))]
            epsilon = [a / np.sum(a) for a in aux1]

            aux2 = [(forward[i] * backward[i]) for i in range(len(data))]
            gamma = [a / np.sum(a) for a in aux2]

            self.p1 = sum(gamma)/len(data)
            self.a = (sum([epsilon[i] for i in range(len(data) - 1)]).T /
                      sum([gamma[i] for i in range(len(data) - 1)])).T

            mu_aux = sum([gamma[i] * data[i] for i in range(len(data))]) / sum(gamma)

            sigma_aux = sum(
                [gamma[i] * np.outer((data[i] - mu_aux), (data[i] - mu_aux)) for i in range(len(data))]) / sum(gamma)

            for i in range(len(self.b)):
                self.b[i]['mu'] = mu_aux[i]
                self.b[i]['sigma'] = sigma_aux[i]

            print({'iter': q + 1, 'p1': self.p1, 'a': self.a, 'b': self.b})

    def get_param_ICE(self, data, iter, Nb_simul):
        print({'iter': 0, 'p1': self.p1, 'a': self.a, 'b': self.b})
        for q in range(iter):
            forward, backward = self.get_forward_backward(data)
            aux1 = [
                (forward[i - 1] * (self.a * multi_norm_pdf_v(data=data[i], x=self.b) * backward[i]).T).T
                for i in
                range(1, len(data))]
            epsilon = [a / np.sum(a) for a in aux1]

            aux2 = [(forward[i] * backward[i]) for i in range(len(data))]
            gamma = [a / np.sum(a) for a in aux2]

            self.p1 = sum(gamma)/len(data)
            self.a = (sum([epsilon[i] for i in range(len(data) - 1)]).T /
                      sum([gamma[i] for i in range(len(data) - 1)])).T

            forward, backward = self.get_forward_backward(data)
            hidden_list = [self.simul_hidden(data, forward, backward) for n in range(Nb_simul)]

            mu_aux = sum([sum([np.array(
                [(hidden[i] == l) * data[i] for l in range(len(self.b))]) for i in range(1, len(data))]) for hidden in hidden_list]) / sum(
                [sum([np.array(
                    [np.full((len(data[0])), (hidden[i] == l)) for l in
                      range(len(self.b))]) for i in range(1, len(data))]) for hidden in hidden_list])

            sigma_aux = sum([sum([np.array(
                [(hidden[i] == l) * np.outer(data[i] - mu_aux[l],
                                                                     data[i] - mu_aux[l]) for l
                in range(len(self.b))]) for i in range(1, len(data))]) for hidden in hidden_list]) / sum(
                [sum([np.array(
                    [np.full((len(data[0]), len(data[0])), (hidden[i] == l)) for
                      l in range(len(self.b))]) for i in range(1, len(data))]) for hidden in hidden_list])

            for i in range(len(self.b)):
                self.b[i]['mu'] = mu_aux[i]
                self.b[i]['sigma'] = sigma_aux[i]

            print({'iter': q + 1, 'p1': self.p1, 'a': self.a, 'b': self.b})

    def get_param_SEM(self, data, iter):
        print({'iter': 0, 'p1': self.p1, 'a': self.a, 'b': self.b})
        for q in range(iter):
            forward, backward = self.get_forward_backward(data)
            hidden = self.simul_hidden(data, forward, backward)
            c = (1 / (len(data) - 1)) * sum([np.array(
                [[(hidden[i - 1] == k and hidden[i] == l) for l in range(self.a.shape[1])] for k in
                 range(self.a.shape[0])]) for i in range(1, len(data))])
            gamma = (1 / (len(data) - 1)) * sum(
                [np.array([hidden[i] == l for l in range(self.a.shape[0])]) for i in range(1, len(data))])
            self.p1 = sum([np.array([hidden[i] == l for l in range(self.a.shape[0])]).astype(int) for i in range(len(data))])/len(data)
            self.a = (c.T / gamma).T

            mu_aux = sum([np.array(
                [(hidden[i] == l) * data[i] for l in range(len(self.b))]) for i in
                range(1, len(data))]) / sum(
                [np.array(
                    [np.full((len(data[0])), (hidden[i] == l)) for l in range(len(self.b))]) for i in range(1, len(data))])

            sigma_aux = sum([np.array(
                [(hidden[i] == l) * np.outer(data[i] - mu_aux[l],
                                             data[i] - mu_aux[l])
                 for l in
                 range(len(self.b))]) for i in range(1, len(data))]) / sum(
                [np.array([np.full((len(data[0]), len(data[0])), (hidden[i] == l)) for l in range(len(self.b))]) for i in
                 range(1, len(data))])

            for i in range(len(self.b)):
                self.b[i]['mu'] = mu_aux[i]
                self.b[i]['sigma'] = sigma_aux[i]

            print({'iter': q + 1, 'p1': self.p1, 'a': self.a, 'b': self.b})



class HMC_ctod_log:
    __slots__ = ('p1', 'a', 'b')

    def __init__(self, p1, a, b):
        self.p1 = p1
        self.a = a
        self.b = b

    def seg_map(self, data):
        pass

    def seg_mpm(self, data):
        forward = self.get_forward(data)
        backward = self.get_backward(data)

        aux = forward[0] + backward[0]
        res = [None] * len(data)
        res[0] = aux.tolist().index(max(aux))
        for i in range(1, len(data)):
            aux = (self.a[res[i - 1], :] + np.log(norm.pdf(data[i], self.b['mu'], self.b['sigma'])) + backward[i]) - \
                  backward[i - 1]
            res[i] = aux.tolist().index(max(aux))
        return np.array(res)

    def simul_hidden(self, data):
        forward = self.get_forward(data)
        backward = self.get_backward(data)

        aux = forward[0] + backward[0]
        test = np.random.multinomial(1, np.exp(aux - ln_sum_np(aux)))
        res = [None] * len(data)
        res[0] = test.tolist().index(1)
        for i in range(1, len(data)):
            aux = (self.a[res[i - 1], :] + np.log(norm.pdf(data[i], self.b['mu'], self.b['sigma'])) + backward[i]) - \
                  backward[
                      i - 1]
            test = np.random.multinomial(1, np.exp(aux - ln_sum_np(aux)))
            res[i] = test.tolist().index(1)
        return np.array(res)

    def get_backward(self, data):
        res = [np.array([0] * self.a.shape[0])] * len(data)
        for l in reversed(range(0, len(data) - 1)):
            res[l] = np.apply_along_axis(ln_sum_np, 1, np.log(self.a) + np.log(norm.pdf(data[l + 1], self.b['mu'], self.b['sigma'])) + res[l + 1])
        return res

    def get_forward(self, data):
        res = [np.log(norm.pdf(data[0], self.b['mu'], self.b['sigma']) * self.p1)] * len(data)
        for l in range(1, len(data)):
            res[l] = np.apply_along_axis(ln_sum_np, 0, (np.log(self.a.T) + res[l - 1]).T + np.log(
                norm.pdf(data[l], self.b['mu'], self.b['sigma'])))
        return res

    def get_param_EM(self, data, iter):
        print({'iter': 0, 'p1': self.p1, 'a': self.a, 'b': self.b})
        for q in range(iter):
            forward = self.get_forward(data)
            backward = self.get_backward(data)
            aux1 = [(forward[i - 1] + (np.log(self.a) + np.log(norm.pdf(data[i], self.b['mu'], self.b['sigma'])) + backward[i]).T).T for i in
                    range(1, len(data))]
            epsilon = [a - ln_sum_np(np.apply_along_axis(ln_sum_np, 0, a)) for a in aux1]

            aux2 = [(forward[i] + backward[i]) for i in range(len(data))]
            gamma = [a - ln_sum_np(a) for a in aux2]

            self.p1 = np.exp(gamma[0])
            self.a = np.exp(((ln_sum_np([epsilon[i] for i in range(len(data) - 1)]).T) - ln_sum_np(
                [gamma[i] for i in range(len(data) - 1)])).T)

            self.b['mu'] = sum([np.exp(gamma[i]) * data[i] for i in range(len(data))]) / np.exp(ln_sum_np(gamma))

            self.b['sigma'] = np.sqrt(
                sum([np.exp(gamma[i]) * (data[i] - self.b['mu']) ** 2 for i in range(len(data))]) / np.exp(ln_sum_np(gamma)))

            print({'iter': q + 1, 'p1': self.p1, 'a': self.a, 'b': self.b})

    def get_param_ICE(self, data, iter, Nb_simul):
        print({'iter': 0, 'p1': self.p1, 'a': self.a, 'b': self.b})
        for q in range(iter):
            forward = self.get_forward(data)
            backward = self.get_backward(data)
            aux1 = [(forward[i - 1] + (
                        np.log(self.a) + np.log(norm.pdf(data[i], self.b['mu'], self.b['sigma'])) + backward[i]).T).T
                    for i in
                    range(1, len(data))]
            epsilon = [a - ln_sum_np(np.apply_along_axis(ln_sum_np, 0, a)) for a in aux1]

            aux2 = [(forward[i] + backward[i]) for i in range(len(data))]
            gamma = [a - ln_sum_np(a) for a in aux2]

            self.p1 = np.exp(gamma[0])
            self.a = np.exp(((ln_sum_np([epsilon[i] for i in range(len(data) - 1)]).T) - ln_sum_np(
                [gamma[i] for i in range(len(data) - 1)])).T)

            hidden_list = [self.simul_hidden(data) for n in range(Nb_simul)]
            mu_aux = [np.array([np.sum(data[np.where(hidden == i)]) for i in range(len(self.b['mu']))]) / np.array(
                [np.sum(hidden == i) for i in range(len(self.b['mu']))]) for hidden in hidden_list]
            sigma_aux = [np.array([np.sum((data[np.where(hidden == i)] - self.b['mu'][i]) ** 2) for i in
                                   range(len(self.b['mu']))]) / np.array(
                [np.sum(hidden == i) for i in range(len(self.b['mu']))]) for hidden in hidden_list]
            self.b['mu'] = (1 / len(mu_aux)) * sum(mu_aux)
            self.b['sigma'] = np.sqrt((1 / len(sigma_aux)) * sum(sigma_aux))
            print({'iter': q + 1, 'p1': self.p1, 'a': self.a, 'b': self.b})

    def get_param_SEM(self, data, iter):
        print({'iter': 0, 'p1': self.p1, 'a': self.a, 'b': self.b})
        for q in range(iter):
            hidden = self.simul_hidden(data)
            c = (1 / (len(data) - 1)) * sum([np.array(
                [[(hidden[i - 1] == k and hidden[i] == l) for l in range(self.a.shape[1])] for k in
                 range(self.a.shape[0])]) for i in range(1, len(data))])
            gamma = (1 / (len(data) - 1)) * sum(
                [np.array([hidden[i] == l for l in range(self.a.shape[0])]) for i in range(1, len(data))])
            self.p1 = np.array([hidden[0] == l for l in range(self.a.shape[0])]).astype(int)
            self.a = (c.T / gamma).T
            self.b['mu'] = np.array([np.sum(data[np.where(hidden == i)]) for i in range(len(self.b['mu']))]) / np.array(
                [np.sum(hidden == i) for i in range(len(self.b['mu']))])
            self.b['sigma'] = np.array([np.sum((data[np.where(hidden == i)] - self.b['mu'][i]) ** 2) for i in
                                        range(len(self.b['mu']))]) / np.array(
                [np.sum(hidden == i) for i in range(len(self.b['mu']))])
            print({'iter': q + 1, 'p1': self.p1, 'a': self.a, 'b': self.b})


class HMC_dtod:

    def __init__(self, p1, a, b):
        self.p1 = p1
        self.a = a
        self.b = b

    def seg_map(self, data):
        pass

    def seg_mpm(self, data):
        aux = self.get_forward(data, 0) * self.p1 * self.get_backward(data, 0)
        aux = aux / np.sum(aux)
        res = [None] * len(data)
        res[0] = aux.tolist().index(max(aux))
        for i in range(1, len(data)):
            aux = (self.a[res[i - 1], :] * norm.pdf(data[i], self.b['mu'], self.b['sigma']) * self.get_backward(data,
                                                                                                                i)) / self.get_backward(
                data, i - 1)
            res[i] = aux.tolist().index(max(aux))
        return np.array(res)

    def simul_hidden(self, data):
        aux = self.get_forward(data, 0) * self.get_backward(data, 0)
        aux = aux / np.sum(aux)
        test = np.random.multinomial(1, aux / np.sum(aux))
        res = [None] * len(data)
        res[0] = test.tolist().index(1)
        for i in range(1, len(data)):
            aux = (self.a[res[i - 1], :] * norm.pdf(data[i], self.b['mu'], self.b['sigma']) * self.get_backward(data,
                                                                                                                i)) / self.get_backward(
                data, i - 1)
            test = np.random.multinomial(1, aux / np.sum(aux))
            res[i] = test.tolist().index(1)
        return np.array(res)

    def get_backward(self, data, i):
        res = np.array([1] * self.a.shape[0])
        for l in reversed(range(i, len(data) - 1)):
            res = np.sum(
                res * self.a * self.b[:, data[i + 1]],
                axis=1)
        return res

    def get_forward(self, data, i):
        res = self.b[:, data[0]] * self.p1
        for l in range(1, i + 1):
            res = np.sum((res * self.a.T).T, axis=0) * self.b[:, data[i]]
        return res

    def get_param_EM(self, data, iter):
        print({'iter': 0, 'p1': self.p1, 'a': self.a, 'b': self.b})
        for q in range(iter):
            aux1 = [(self.get_forward(data, i - 1) * (self.a * self.b[:, data[i]] *
                                                      self.get_backward(data, i)).T).T for i in range(1, len(data))]
            epsilon = [a / np.sum(a) for a in aux1]
            aux2 = [(self.get_forward(data, i) * self.get_backward(data, i)) for i in range(len(data))]
            gamma = [a / np.sum(a) for a in aux2]
            # self.p1 = (1/len(data))*sum(gamma)
            self.p1 = gamma[0]
            # self.a = (1 / len(data)) * (sum([(epsilon[i].T / gamma[i]).T for i in range(len(data) - 1)]))
            self.a = (sum([epsilon[i] for i in range(len(data) - 1)]).T / sum(
                [gamma[i] for i in range(len(data) - 1)])).T
            self.b = (sum(
                [np.outer(gamma[i], (data[i] == np.array(range(len(gamma[i]))))) for i in range(len(data))]).T / sum(
                gamma)).T
            print({'iter': q + 1, 'p1': self.p1, 'a': self.a, 'b': self.b})

    def get_param_ICE(self, data, iter, Nb_simul):
        print({'iter': 0, 'p1': self.p1, 'a': self.a, 'b': self.b})
        for q in range(iter):
            aux1 = [(self.get_forward(data, i - 1) * (self.a * self.b[:, data[i]] *
                                                      self.get_backward(data, i)).T).T for i in range(1, len(data))]
            epsilon = [a / np.sum(a) for a in aux1]
            aux2 = [(self.get_forward(data, i) * self.get_backward(data, i)) for i in range(len(data))]
            gamma = [a / np.sum(a) for a in aux2]
            # self.p1 = (1/len(data))*sum(gamma)
            self.p1 = gamma[0]
            # self.a = (1 / len(data)) * (sum([(epsilon[i].T / gamma[i]).T for i in range(len(data) - 1)]))
            self.a = (sum([epsilon[i] for i in range(len(data) - 1)]).T / sum(
                [gamma[i] for i in range(len(data) - 1)])).T

            hidden_list = [self.simul_hidden(data) for n in range(Nb_simul)]
            c_aux = [(1 / (len(data) - 1)) * sum([np.array(
                [[(hidden[i] == k and data[i] == l) for l in range(self.b.shape[1])] for k in
                 range(self.b.shape[0])]) for i in range(len(data))]) for hidden in hidden_list]
            gamma_aux = [(1 / (len(data) - 1)) * sum(
                [np.array([hidden[i] == l for l in range(self.b.shape[0])]) for i in range(len(data))]) for hidden in
                         hidden_list]
            c = (1 / len(c_aux)) * sum(c_aux)
            gamma = (1 / len(gamma_aux)) * sum(gamma_aux)
            self.b = (c.T / gamma).T
            print({'iter': q + 1, 'p1': self.p1, 'a': self.a, 'b': self.b})

    def get_param_SEM(self, data, iter):
        print({'iter': 0, 'p1': self.p1, 'a': self.a, 'b': self.b})
        for q in range(iter):
            hidden = self.simul_hidden(data)
            c = (1 / (len(data) - 1)) * sum([np.array(
                [[(hidden[i - 1] == k and hidden[i] == l) for l in range(self.a.shape[1])] for k in
                 range(self.a.shape[0])]) for i in range(1, len(data))])
            gamma = (1 / (len(data) - 1)) * sum(
                [np.array([hidden[i] == l for l in range(self.a.shape[0])]) for i in range(1, len(data))])
            self.p1 = np.array([hidden[0] == l for l in range(self.a.shape[0])]).astype(int)
            self.a = (c.T / gamma).T
            c2 = (1 / (len(data) - 1)) * sum([np.array(
                [[(hidden[i] == k and data[i] == l) for l in range(self.b.shape[1])] for k in
                 range(self.b.shape[0])]) for i in range(len(data))])
            gamma2 = (1 / (len(data) - 1)) * sum(
                [np.array([hidden[i] == l for l in range(self.b.shape[0])]) for i in range(len(data))])
            self.b = (c2.T / gamma2).T
            print({'iter': q + 1, 'p1': self.p1, 'a': self.a, 'b': self.b})


class HMC_ctoc:
    __slots__ = ('p1', 'a', 'b')

    def __init__(self, p1, a, b):
        self.p1 = p1
        self.a = a
        self.b = b