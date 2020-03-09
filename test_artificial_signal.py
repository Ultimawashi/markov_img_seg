import numpy as np
import os
import cv2 as cv
import json
from utils import get_peano_index, convert_multcls_vectors, moving_average, calc_err, sigmoid_np
from hmm import HMC_ctod, HSMC_ctod, HEMC_ctod, HESMC_ctod, HSEMC_ctod
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal


resolution = (64, 64)
signal_length = np.prod(resolution)
test = get_peano_index(resolution[0])  # Parcours de peano
# test = [a.flatten() for a in np.indices(resolution)] #Parcours ligne par ligne
max_val = 255

def generate_signal21(length, mu, sigma):
    hidden = np.zeros(length, dtype=int)
    visible = np.zeros((length, mu.shape[-1]))
    delta0 = 1 / length
    p0 = np.array([delta0, 1 - delta0])
    test = np.random.multinomial(1, p0)
    hidden[0] = np.argmax(test)
    visible[0] = multivariate_normal.rvs(mu[hidden[0]], sigma[hidden[0]])
    for i in range(1, length):
        deltai = (i + 1) / length
        T = np.array([[deltai, 1 - deltai], [1 - deltai, deltai]])
        test = np.random.multinomial(1, T[hidden[i - 1], :])
        hidden[i] = np.argmax(test)
        visible[i] = multivariate_normal.rvs(mu[hidden[i]], sigma[hidden[i]])
    return hidden, visible


def generate_signal22(length, mu, sigma):
    hidden = np.zeros(length, dtype=int)
    visible = np.zeros((length, mu.shape[-1]))
    delta0 = (3 / 4) + (1 / 4) * np.sin(1 / 5)
    p0 = np.array([delta0, 1 - delta0])
    test = np.random.multinomial(1, p0)
    hidden[0] = np.argmax(test)
    visible[0] = multivariate_normal.rvs(mu[hidden[0]], sigma[hidden[0]])
    for i in range(1, length):
        deltai = (3 / 4) + (1 / 4) * np.sin(1 / 5)
        T = np.array([[deltai, 1 - deltai], [1 - deltai, deltai]])
        test = np.random.multinomial(1, T[hidden[i - 1], :])
        hidden[i] = np.argmax(test)
        visible[i] = multivariate_normal.rvs(mu[hidden[i]], sigma[hidden[i]])
    return hidden, visible


def generate_signal31(length, mu, sigma):
    hidden = np.zeros(length, dtype=int)
    visible = np.zeros((length, mu.shape[-1]))
    delta0 = np.random.uniform(0,1)
    p0 = np.array([delta0, 1 - delta0])
    test = np.random.multinomial(1, p0)
    hidden[0] = np.argmax(test)
    visible[0] = multivariate_normal.rvs(mu[hidden[0]], sigma[hidden[0]])
    for i in range(1, length):
        deltai =  np.random.uniform(0,1)
        T = np.array([[deltai, 1 - deltai], [1 - deltai, deltai]])
        test = np.random.multinomial(1, T[hidden[i - 1], :])
        hidden[i] = np.argmax(test)
        visible[i] = multivariate_normal.rvs(mu[hidden[i]], sigma[hidden[i]])
    return hidden, visible


def generate_signal32(length, mu, sigma, varparm=0.1):
    hidden = np.zeros(length, dtype=int)
    visible = np.zeros((length, mu.shape[-1]))
    delta = np.zeros(length, dtype=float)
    delta[0] = np.random.uniform(0,1)
    p0 = np.array([delta[0], 1 - delta[0]])
    test = np.random.multinomial(1, p0)
    hidden[0] = np.argmax(test)
    visible[0] = multivariate_normal.rvs(mu[hidden[0]], sigma[hidden[0]])
    for i in range(1, length):
        delta[i] = sigmoid_np(multivariate_normal.rvs(delta[i-1], varparm))
        T = np.array([[delta[i], 1 - delta[i]], [1 - delta[i], delta[i]]])
        test = np.random.multinomial(1, T[hidden[i - 1], :])
        hidden[i] = np.argmax(test)
        visible[i] = multivariate_normal.rvs(mu[hidden[i]], sigma[hidden[i]])
    return hidden, visible


def generate_nonstat_noise(length, mu, sigma, law=np.array([[0.19,0.1],[0.1,0.79]]), varmean=0.2):
    hidden = np.zeros(length, dtype=int)
    visible = np.zeros((length, mu.shape[-1]))
    p0 = law.sum(axis=1)
    T = (law.T / p0).T
    test = np.random.multinomial(1, p0)
    hidden[0] = np.argmax(test)
    visible[0] = multivariate_normal.rvs(mu[hidden[0]], sigma[hidden[0]])
    for i in range(1, length):
        mu_aux = np.copy(mu)
        mu_aux[1] = multivariate_normal.rvs(mu[1], varmean)
        test = np.random.multinomial(1, T[hidden[i - 1], :])
        hidden[i] = np.argmax(test)
        visible[i] = multivariate_normal.rvs(mu_aux[hidden[i]], sigma[hidden[i]])
    return hidden, visible


def generate_semi_markov_non_stat1(length, mu, sigma, lawu=np.array([[[0.2,0.2,0.2,0.2,0.2], [0.2,0.2,0.2,0.2,0.2]],[[0.2,0.2,0.2,0.2,0.2], [0.2,0.2,0.2,0.2,0.2]]])):
    nbc_x = 2
    nbc_u = lawu.shape[-1]
    nb_class = nbc_x * nbc_u
    hidden = np.zeros(length, dtype=int)
    visible = np.zeros((length, mu.shape[-1]))
    mu = np.repeat(mu, nbc_u, axis=0)
    sigma = np.repeat(sigma, nbc_u, axis=0)
    p0 = np.random.uniform(0, 1, (nbc_x*nbc_u,))
    p0 = p0/p0.sum()
    test = np.random.multinomial(1, p0)
    hidden[0] = np.argmax(test)
    visible[0] = multivariate_normal.rvs(mu[hidden[0]], sigma[hidden[0]])
    for l in range(1, length):
        deltai = (3 / 4) + (1 / 4) * np.sin(l / 5)
        a = np.array([[deltai, 1 - deltai], [1 - deltai, deltai]])
        b = np.repeat(np.eye(nbc_x, nbc_x, k=0), nbc_u, axis=0)
        idx = [i for i in range(b.shape[0]) if ((i + 1) % nbc_u == 0)]
        for i, e in enumerate(idx):
            b[e] = a[i]

        a = lawu
        ut = [[np.eye(nbc_u, k=1) for n1 in range(nbc_x)] for n2 in range(int(nb_class / nbc_u))]
        for i, e in enumerate(ut):
            for j, p in enumerate(e):
                p[-1] = a[i, j]
        ut = np.block(ut)
        for i in range(1, nbc_x + 1):
            ut[:, (i - 1) * nbc_u:i * nbc_u] = (
                    ut[:, (i - 1) * nbc_u:i * nbc_u].T * b[:, i - 1]).T
        T = ut
        test = np.random.multinomial(1, T[hidden[l - 1], :])
        hidden[l] = np.argmax(test)
        visible[l] = multivariate_normal.rvs(mu[hidden[l]], sigma[hidden[l]])
    return convert_multcls_vectors(hidden, (nbc_u, nbc_x))[:, 1], visible


def generate_semi_markov_non_stat2(length, mu, sigma):
    nbc_x = 2
    nbc_u = 5
    nb_class = nbc_x * nbc_u
    hidden = np.zeros(length, dtype=int)
    visible = np.zeros((length, mu.shape[-1]))
    mu = np.repeat(mu, nbc_u, axis=0)
    sigma = np.repeat(sigma, nbc_u, axis=0)
    p0 = np.random.uniform(0, 1, (nbc_x * nbc_u,))
    p0 = p0 / p0.sum()
    test = np.random.multinomial(1, p0)
    hidden[0] = np.argmax(test)
    visible[0] = multivariate_normal.rvs(mu[hidden[0]], sigma[hidden[0]])
    for l in range(1, length):
        lawu = np.random.uniform(0, 1, (nbc_x, nbc_x, nbc_u))
        max_vals = np.sum(lawu, axis=-1)
        lawu = lawu / max_vals[:, :, np.newaxis]
        deltai = (3 / 4) + (1 / 4) * np.sin(l / 5)
        a = np.array([[deltai, 1 - deltai], [1 - deltai, deltai]])
        b = np.repeat(np.eye(nbc_x, nbc_x, k=0), nbc_u, axis=0)
        idx = [i for i in range(b.shape[0]) if ((i + 1) % nbc_u == 0)]
        for i, e in enumerate(idx):
            b[e] = a[i]

        a = lawu
        ut = [[np.eye(nbc_u, k=1) for n1 in range(nbc_x)] for n2 in range(int(nb_class / nbc_u))]
        for i, e in enumerate(ut):
            for j, p in enumerate(e):
                p[-1] = a[i, j]
        ut = np.block(ut)
        for i in range(1, nbc_x + 1):
            ut[:, (i - 1) * nbc_u:i * nbc_u] = (
                    ut[:, (i - 1) * nbc_u:i * nbc_u].T * b[:, i - 1]).T
        T = ut
        test = np.random.multinomial(1, T[hidden[l - 1], :])
        hidden[l] = np.argmax(test)
        visible[l] = multivariate_normal.rvs(mu[hidden[l]], sigma[hidden[l]])
    return convert_multcls_vectors(hidden, (nbc_u, nbc_x))[:, 1], visible


def generate_semi_markov_non_stat3(length, mu, sigma, lawu=np.array([[[0.2,0.2,0.2,0.2,0.2], [0.2,0.2,0.2,0.2,0.2]],[[0.2,0.2,0.2,0.2,0.2], [0.2,0.2,0.2,0.2,0.2]]])):
    nbc_x = 2
    nbc_u = lawu.shape[-1]
    nb_class = nbc_x * nbc_u
    hidden = np.zeros(length, dtype=int)
    visible = np.zeros((length, mu.shape[-1]))
    mu = np.repeat(mu, nbc_u, axis=0)
    sigma = np.repeat(sigma, nbc_u, axis=0)
    p0 = np.random.uniform(0, 1, (nbc_x*nbc_u,))
    p0 = p0/p0.sum()
    test = np.random.multinomial(1, p0)
    hidden[0] = np.argmax(test)
    visible[0] = multivariate_normal.rvs(mu[hidden[0]], sigma[hidden[0]])
    for l in range(1, length):
        deltai = np.random.uniform(0,1)
        a = np.array([[deltai, 1 - deltai], [1 - deltai, deltai]])
        b = np.repeat(np.eye(nbc_x, nbc_x, k=0), nbc_u, axis=0)
        idx = [i for i in range(b.shape[0]) if ((i + 1) % nbc_u == 0)]
        for i, e in enumerate(idx):
            b[e] = a[i]

        a = lawu
        ut = [[np.eye(nbc_u, k=1) for n1 in range(nbc_x)] for n2 in range(int(nb_class / nbc_u))]
        for i, e in enumerate(ut):
            for j, p in enumerate(e):
                p[-1] = a[i, j]
        ut = np.block(ut)
        for i in range(1, nbc_x + 1):
            ut[:, (i - 1) * nbc_u:i * nbc_u] = (
                    ut[:, (i - 1) * nbc_u:i * nbc_u].T * b[:, i - 1]).T
        T = ut
        test = np.random.multinomial(1, T[hidden[l - 1], :])
        hidden[l] = np.argmax(test)
        visible[l] = multivariate_normal.rvs(mu[hidden[l]], sigma[hidden[l]])
    return convert_multcls_vectors(hidden, (nbc_u, nbc_x))[:, 1], visible


def generate_semi_markov_non_stat4(length, mu, sigma):
    nbc_x = 2
    nbc_u = 5
    nb_class = nbc_x * nbc_u
    hidden = np.zeros(length, dtype=int)
    visible = np.zeros((length, mu.shape[-1]))
    mu = np.repeat(mu, nbc_u, axis=0)
    sigma = np.repeat(sigma, nbc_u, axis=0)
    p0 = np.random.uniform(0, 1, (nbc_x * nbc_u,))
    p0 = p0 / p0.sum()
    test = np.random.multinomial(1, p0)
    hidden[0] = np.argmax(test)
    visible[0] = multivariate_normal.rvs(mu[hidden[0]], sigma[hidden[0]])
    for l in range(1, length):
        lawu = np.random.uniform(0, 1, (nbc_x, nbc_x, nbc_u))
        max_vals = np.sum(lawu, axis=-1)
        lawu = lawu / max_vals[:, :, np.newaxis]
        deltai = np.random.uniform(0,1)
        a = np.array([[deltai, 1 - deltai], [1 - deltai, deltai]])
        b = np.repeat(np.eye(nbc_x, nbc_x, k=0), nbc_u, axis=0)
        idx = [i for i in range(b.shape[0]) if ((i + 1) % nbc_u == 0)]
        for i, e in enumerate(idx):
            b[e] = a[i]

        a = lawu
        ut = [[np.eye(nbc_u, k=1) for n1 in range(nbc_x)] for n2 in range(int(nb_class / nbc_u))]
        for i, e in enumerate(ut):
            for j, p in enumerate(e):
                p[-1] = a[i, j]
        ut = np.block(ut)
        for i in range(1, nbc_x + 1):
            ut[:, (i - 1) * nbc_u:i * nbc_u] = (
                    ut[:, (i - 1) * nbc_u:i * nbc_u].T * b[:, i - 1]).T
        T = ut
        test = np.random.multinomial(1, T[hidden[l - 1], :])
        hidden[l] = np.argmax(test)
        visible[l] = multivariate_normal.rvs(mu[hidden[l]], sigma[hidden[l]])
    return convert_multcls_vectors(hidden, (nbc_u, nbc_x))[:, 1], visible


def generate_semi_markov_non_stat5(length, mu, sigma):
    nbc_x = 2
    nbc_u = 5
    nb_class = nbc_x * nbc_u
    hidden = np.zeros(length, dtype=int)
    visible = np.zeros((length, mu.shape[-1]))
    mu = np.repeat(mu, nbc_u, axis=0)
    sigma = np.repeat(sigma, nbc_u, axis=0)
    p0 = np.random.uniform(0, 1, (nbc_x * nbc_u,))
    p0 = p0 / p0.sum()
    test = np.random.multinomial(1, p0)
    hidden[0] = np.argmax(test)
    visible[0] = multivariate_normal.rvs(mu[hidden[0]], sigma[hidden[0]])
    for l in range(1, length):

        deltai = (l + 1) / length
        lawu = np.tile(np.array([1 -deltai, deltai/6,deltai/6,deltai/6,deltai/2]),(nbc_x,nbc_x,1))
        a = np.array([[deltai, 1 - deltai], [1 - deltai, deltai]])
        b = np.repeat(np.eye(nbc_x, nbc_x, k=0), nbc_u, axis=0)
        idx = [i for i in range(b.shape[0]) if ((i + 1) % nbc_u == 0)]
        for i, e in enumerate(idx):
            b[e] = a[i]

        a = lawu
        ut = [[np.eye(nbc_u, k=1) for n1 in range(nbc_x)] for n2 in range(int(nb_class / nbc_u))]
        for i, e in enumerate(ut):
            for j, p in enumerate(e):
                p[-1] = a[i, j]
        ut = np.block(ut)
        for i in range(1, nbc_x + 1):
            ut[:, (i - 1) * nbc_u:i * nbc_u] = (
                    ut[:, (i - 1) * nbc_u:i * nbc_u].T * b[:, i - 1]).T
        T = ut
        test = np.random.multinomial(1, T[hidden[l - 1], :])
        hidden[l] = np.argmax(test)
        visible[l] = multivariate_normal.rvs(mu[hidden[l]], sigma[hidden[l]])
    return convert_multcls_vectors(hidden, (nbc_u, nbc_x))[:, 1], visible


def generate_signal_markov_stat(length, mu, sigma, law=np.array([[0.19,0.1],[0.1,0.79]])):
    hidden = np.zeros(length, dtype=int)
    visible = np.zeros((length, mu.shape[-1]))
    p0 = law.sum(axis=1)
    T = (law.T / p0).T
    test = np.random.multinomial(1, p0)
    hidden[0] = np.argmax(test)
    visible[0] = multivariate_normal.rvs(mu[hidden[0]], sigma[hidden[0]])
    for i in range(1, length):
        test = np.random.multinomial(1, T[hidden[i - 1], :])
        hidden[i] = np.argmax(test)
        visible[i] = multivariate_normal.rvs(mu[hidden[i]], sigma[hidden[i]])
    return hidden, visible


# signaltest = np.array(
#     [0] * 20 + [0, 1] * 40 + [0] * 20 + [1] * 20 + [0] * 20 + [1] * 20 + [0] * 20 + [1] * 20 + [0] * 20 + [1] * 20 + [
#             0] * 20 + [0, 1] * 40 + [0] * 20 + [1] * 20 + [0] * 20 + [1] * 20 + [0] * 20 + [1] * 20 + [0] * 20 + [
#         1] * 20 + [0] * 20 + [0, 1] * 40 + [0] * 20 + [1] * 20 + [0] * 20 + [1] * 20 + [0] * 20 + [1] * 20 + [
#         0] * 20 + [1] * 20 + [0] * 20 + [0, 1] * 40 + [0] * 20 + [1] * 20 + [0] * 20 + [1] * 20 + [0] * 20 + [
#         1] * 20 + [0] * 20 + [1] * 20)



resfolder = './img/res_test2/artificial_signals'
# signals = [{'name': 'signal21', 'signal': generate_signal21},
#            {'name': 'signal22', 'signal': generate_signal22}, {'name': 'signal31', 'signal': generate_signal31}, {'name': 'signal_non_stat', 'signal': generate_nonstat_noise}]

signals = [{'name': 'signal_semi_markov_non_stat5', 'signal': generate_semi_markov_non_stat5}]


gauss_noise = [{'corr': False, 'mu1': 0, 'mu2': 1, 'sig1': 1, 'sig2': 1}]

models = [
    {'name': 'hmc', 'model': HMC_ctod(2), 'params': None},
    {'name': 'hsmc', 'model': HSMC_ctod(2, 5), 'params': None},
    {'name': 'hemc', 'model': HEMC_ctod(2), 'params': None},
    {'name': 'hsemc', 'model': HSEMC_ctod(2, 5), 'params': None},
    {'name': 'hesmc', 'model': HESMC_ctod(2, 5), 'params': None}]
kmeans_clusters = 2

if not os.path.exists(resfolder):
    os.makedirs(resfolder)


if not os.path.exists(os.path.join(resfolder, 'terr.txt')):
    terr = {}
    with open(os.path.join(resfolder, 'terr.txt'), 'w') as f:
        json.dump(terr, f, ensure_ascii=False, indent=2)




for signal in signals:
    for noise in gauss_noise:
        corr = ''
        corr_param = ''
        noise_param = '(' + str(noise['mu1']) + ',' + str(noise['sig1']) + ')' + '_' + '(' + str(
            noise['mu2']) + ',' + str(noise['sig2']) + ')'
        true_signal,signal_noisy = signal['signal'](signal_length, mu = np.array([[noise['mu1']], [noise['mu2']]]), sigma = np.array([[[noise['sig1']]], [[noise['sig2']]]]))

        img = np.zeros(resolution)
        img[test[0], test[1]] = true_signal
        cv.imwrite(resfolder + '/' + signal['name'] + '_' + corr + '_' + corr_param + noise_param + '_true_signal.bmp',
                   img * max_val)

        img = np.zeros(resolution)
        img[test[0], test[1]] = signal_noisy.reshape((signal_noisy.shape[0],))
        cv.imwrite(resfolder + '/' + signal['name'] + '_' + corr + '_' + corr_param + noise_param + '.bmp',
                   sigmoid_np(img) * max_val)

        data = signal_noisy.reshape(-1, 1)
        kmeans = KMeans(n_clusters=kmeans_clusters).fit(data)
        seg = kmeans.labels_
        # terr = {signal['name'] + '_' + corr + '_' + corr_param + noise_param + '_seg_kmeans':calc_err(seg, true_signal)}
        with open(os.path.join(resfolder, 'terr.txt'), 'r') as f:
            content = json.load(f)
        with open(os.path.join(resfolder, 'terr.txt'), 'w') as f:
            content[signal['name'] + '_' + corr + '_' + corr_param + noise_param + '_seg_kmeans'] = calc_err(seg, true_signal)
            json.dump(content, f, ensure_ascii=False, indent=2)

        img = np.zeros(resolution)
        img[test[0], test[1]] = seg
        cv.imwrite(resfolder + '/' + signal['name'] + '_' + corr + '_' + corr_param + noise_param + '_seg_kmeans' + '.bmp',
                   img * int(max_val / (kmeans_clusters - 1)))
        for model in models:
            if not model['params']:
                model['model'].init_data_prior(data)
            else:
                model['model'].give_param(*model['params'])
            model['model'].get_param_EM(data,
                                        10000,
                                        early_stopping=10 ** -10)  # estimation des paramètres avec ICE, (on peut utiliser SEM ou EM avec get_param_EM ou get_param_SEM)

            seg = model['model'].seg_mpm(data)  # Remplir notre matrice avec les valeurs de la segmentation
            # terr = {signal['name'] + '_' + corr + '_' + corr_param + noise_param + '_seg_' + model['name']: calc_err(seg, true_signal)}
            with open(os.path.join(resfolder, 'terr.txt'), 'r') as f:
                content = json.load(f)
            with open(os.path.join(resfolder, 'terr.txt'), 'w') as f:
                content[signal['name'] + '_' + corr + '_' + corr_param + noise_param + '_seg_' + model['name']] = calc_err(seg,
                                                                                                                 true_signal)
                json.dump(content, f, ensure_ascii=False, indent=2)
            img = np.zeros(resolution)
            img[test[0], test[1]] = seg
            cv.imwrite(
                resfolder + '/' + signal['name'] + '_' + corr + '_' + corr_param + noise_param + '_seg_' + model[
                    'name'] + '.bmp',
                img * max_val)
            param_s = {'p': model['model'].p.tolist(), 't': model['model'].t.tolist(), 'mu': model['model'].mu.tolist(),
                       'sig': model['model'].sigma.tolist()}

            with open(os.path.join(resfolder + '/' + signal['name'] + '_' + corr + '_' + corr_param + noise_param + '_param_' + model['name'] + '.txt'),
                      'w') as f:
                json.dump(param_s, f, ensure_ascii=False)
