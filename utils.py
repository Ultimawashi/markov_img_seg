import numpy as np
from math import log2
from scipy import signal
from itertools import groupby
import cv2 as cv


def sigmoid_np(x):
    return 1 / (1 + np.exp(-x))


def heaviside_np(x):
    thresold = np.max(x)/2
    return (x < thresold) * 0 + (x >= thresold)


def standardize_np(x):
    return (x-np.mean(x))/np.std(x)


def get_peano_index(dSize):
    xTmp = 0
    yTmp = 0
    dirTmp = 0
    dirLookup = np.array(
        [[3, 0, 0, 1], [0, 1, 1, 2], [1, 2, 2, 3], [2, 3, 3, 0], [1, 0, 0, 3], [2, 1, 1, 0], [3, 2, 2, 1],
         [0, 3, 3, 2]]).T
    dirLookup = dirLookup + np.array(
        [[4, 0, 0, 4], [4, 0, 0, 4], [4, 0, 0, 4], [4, 0, 0, 4], [0, 4, 4, 0], [0, 4, 4, 0], [0, 4, 4, 0],
         [0, 4, 4, 0]]).T
    orderLookup = np.array(
        [[0, 2, 3, 1], [1, 0, 2, 3], [3, 1, 0, 2], [2, 3, 1, 0], [1, 3, 2, 0], [3, 2, 0, 1], [2, 0, 1, 3],
         [0, 1, 3, 2]]).T
    offsetLookup = np.array([[1, 1, 0, 0], [1, 0, 1, 0]])
    for i in range(int(log2(dSize))):
        xTmp = np.array([(xTmp - 1) * 2 + offsetLookup[0, orderLookup[0, dirTmp]] + 1,
            (xTmp - 1) * 2 + offsetLookup[0, orderLookup[1, dirTmp]] + 1,
            (xTmp - 1) * 2 + offsetLookup[0, orderLookup[2, dirTmp]] + 1,
            (xTmp - 1) * 2 + offsetLookup[0, orderLookup[3, dirTmp]] + 1])

        yTmp = np.array([(yTmp - 1) * 2 + offsetLookup[1, orderLookup[0, dirTmp]] + 1,
            (yTmp - 1) * 2 + offsetLookup[1, orderLookup[1, dirTmp]] + 1,
            (yTmp - 1) * 2 + offsetLookup[1, orderLookup[2, dirTmp]] + 1,
            (yTmp - 1) * 2 + offsetLookup[1, orderLookup[3, dirTmp]] + 1])

        dirTmp = np.array([dirLookup[0, dirTmp],dirLookup[1, dirTmp], dirLookup[2, dirTmp], dirLookup[3, dirTmp]])

        xTmp = xTmp.T.flatten()
        yTmp = yTmp.T.flatten()
        dirTmp = dirTmp.flatten()

    x = - xTmp
    y = - yTmp
    return x,y


def generate_semipos_sym_mat(size, var_bound):
    aux = np.random.rand(*size) * (var_bound[1] - var_bound[0]) + var_bound[0]
    return np.dot(aux, aux.transpose())


def convert_multcls_vectors(data, rand_vect_param):
    classes = range(np.max(data).astype(int) + 1)
    assert (len(classes) <= np.prod(rand_vect_param)), 'Les paramètres du vecteur aléatoire ne correspondent pas'
    res = np.zeros((len(data),len(rand_vect_param)))
    aux = [convertcls_vect(cls, rand_vect_param) for cls in classes]
    for c in classes:
        res[data==c] = aux[c]
    return res.astype('int')


def convertcls_vect(cls, rand_vect_param):
    aux = cls
    res=np.zeros((len(rand_vect_param)))
    for i in reversed(range(len(rand_vect_param))):
        res[len(rand_vect_param) - i - 1] = aux % rand_vect_param[i]
        aux = aux // rand_vect_param[len(rand_vect_param) - i - 1]
    return res


def convert_vect_multcls(data, rand_vect_param):
    vectors = np.stack([a.flatten() for a in reversed(np.indices(rand_vect_param))]).T
    res = np.zeros((len(data)))
    for i,v in enumerate(vectors):
        res[(data == v).all(axis=1)] = i
    return res.astype('int')


def np_multivariate_normal_pdf(x, mu, cov):
    broadc = (len(mu.shape) - len(x.shape) + 1)
    x = x.reshape((x.shape[0],) + (1,) * broadc + x.shape[1:])
    part1 = 1 / (((2 * np.pi) ** (mu.shape[-1] / 2)) * (np.linalg.det(cov) ** (1 / 2)))
    part2 = (-1 / 2) * np.einsum('...j,...j',np.einsum('...j,...ji',(x - mu),np.linalg.inv(cov)),(x - mu))
    return part1 * np.exp(part2)


def np_multivariate_normal_pdf_marginal(x, mu, cov, j, i=0):
    broadc = (len(mu.shape) - len(x.shape) + 1)
    x = x.reshape((x.shape[0],) + (1,) * broadc + x.shape[1:])
    part1 = 1 / (((2 * np.pi) ** (mu[...,i:j+1].shape[-1] / 2)) * (np.linalg.det(cov[...,i:j+1,i:j+1]) ** (1 / 2)))
    part2 = (-1 / 2) * np.einsum('...j,...j',np.einsum('...j,...ji',(x - mu[...,i:j+1]),np.linalg.inv(cov[...,i:j+1,i:j+1])),(x - mu[...,i:j+1]))
    return part1 * np.exp(part2)


def multinomial_rvs(n, p):
    """
    Sample from the multinomial distribution with multiple p vectors.

    * count must be an (n-1)-dimensional numpy array.
    * p must an n-dimensional numpy array, n >= 1.  The last axis of p
      holds the sequence of probabilities for a multinomial distribution.

    The return value has the same shape as p.
    """

    count = np.full(p.shape[:-1], n)
    out = np.zeros(p.shape, dtype=int)
    ps = p.cumsum(axis=-1)
    # Conditional probabilities
    with np.errstate(divide='ignore', invalid='ignore'):
        condp = p / ps
    condp[np.isnan(condp)] = 0.0
    for i in range(p.shape[-1]-1, 0, -1):
        binsample = np.random.binomial(count, condp[..., i])
        out[..., i] = binsample
        count -= binsample
    out[..., 0] = count
    return out


def moving_average(x, neighbour, a):
    assert (neighbour == 4 or neighbour == 8), 'please choose only between 4 and 8 neighbour'
    if neighbour == 4:
        filter = np.array([[0,a,0], [a,1,a], [0,a,0]])
    else:
        filter = np.array([[a, a, a], [a, 1, a], [a, a, a]])
    return signal.convolve2d(x, filter, mode='same')


def calc_product(list_mat):
    res=list_mat[len(list_mat)-1]
    for i in reversed(range(len(list_mat)-1)):
        res=(res.flatten()*list_mat[i].T).T
    return res.reshape(int(np.sqrt(np.prod(res.shape))), int(np.sqrt(np.prod(res.shape))))


def algorithm_u(ns, m):
    def visit(n, a):
        ps = [[] for i in range(m)]
        for j in range(n):
            ps[a[j + 1]].append(ns[j])
        return ps

    def f(mu, nu, sigma, n, a):
        if mu == 2:
            yield visit(n, a)
        else:
            for v in f(mu - 1, nu - 1, (mu + sigma) % 2, n, a):
                yield v
        if nu == mu + 1:
            a[mu] = mu - 1
            yield visit(n, a)
            while a[nu] > 0:
                a[nu] = a[nu] - 1
                yield visit(n, a)
        elif nu > mu + 1:
            if (mu + sigma) % 2 == 1:
                a[nu - 1] = mu - 1
            else:
                a[mu] = mu - 1
            if (a[nu] + sigma) % 2 == 1:
                for v in b(mu, nu - 1, 0, n, a):
                    yield v
            else:
                for v in f(mu, nu - 1, 0, n, a):
                    yield v
            while a[nu] > 0:
                a[nu] = a[nu] - 1
                if (a[nu] + sigma) % 2 == 1:
                    for v in b(mu, nu - 1, 0, n, a):
                        yield v
                else:
                    for v in f(mu, nu - 1, 0, n, a):
                        yield v

    def b(mu, nu, sigma, n, a):
        if nu == mu + 1:
            while a[nu] < mu - 1:
                yield visit(n, a)
                a[nu] = a[nu] + 1
            yield visit(n, a)
            a[mu] = 0
        elif nu > mu + 1:
            if (a[nu] + sigma) % 2 == 1:
                for v in f(mu, nu - 1, 0, n, a):
                    yield v
            else:
                for v in b(mu, nu - 1, 0, n, a):
                    yield v
            while a[nu] < mu - 1:
                a[nu] = a[nu] + 1
                if (a[nu] + sigma) % 2 == 1:
                    for v in f(mu, nu - 1, 0, n, a):
                        yield v
                else:
                    for v in b(mu, nu - 1, 0, n, a):
                        yield v
            if (mu + sigma) % 2 == 1:
                a[nu - 1] = 0
            else:
                a[mu] = 0
        if mu == 2:
            yield visit(n, a)
        else:
            for v in b(mu - 1, nu - 1, (mu + sigma) % 2, n, a):
                yield v

    n = len(ns)
    a = [0] * (n + 1)
    for j in range(1, m + 1):
        a[n - m + j] = j - 1
    return f(m, n, 0, n, a)


def calc_err(ref_im, seg_im):
    terr = np.sum(seg_im != ref_im) / np.prod(ref_im.shape)
    return (terr <= 0.5) * terr + (terr > 0.5) * (1 - terr)


def split_in(num, list):
    return [list[x:x + num] for x in range(0, len(list), num)]


def cut_diff(inp):
    return [list(g) for k, g in groupby(inp, key=lambda i: i)]


def calc_matDS(m, lx):
    nbc_x, nbc_u = lx.shape
    return np.moveaxis((m[np.newaxis,:,:]*lx[:,:,np.newaxis]).reshape(nbc_u*nbc_x, nbc_u)[np.newaxis,...]*lx[:,np.newaxis,:],0,1).reshape(nbc_u*nbc_x, nbc_u*nbc_x)


def test_calc_cond_DS(c):
    aux = np.array([c[0, :], [0, 0, 0], [0, 0, 0]])
    a = c[0, 0] * aux[0, 0] + aux[0, 0] * c[0, 0] + c[0, 0] * aux[0, 2] + aux[0, 0] * c[0, 2] + aux[2, 0] * c[0, 0] + c[
        2, 0] * aux[0, 0] + aux[2, 2] * c[0, 0] + c[2, 2] * aux[0, 0]
    b = c[0, 1] * aux[0, 1] + aux[0, 1] * c[0, 1] + c[0, 1] * aux[0, 2] + aux[0, 1] * c[0, 2] + aux[2, 0] * c[0, 1] + c[
        2, 0] * aux[0, 1] + aux[2, 2] * c[0, 1] + c[2, 2] * aux[0, 1]
    d = c[0, 2] * aux[0, 2] + aux[0, 2] * c[0, 2] + c[2, 2] * aux[0, 2] + aux[2, 2] * c[0, 2] + aux[2, 2] * c[2, 2] + c[
        2, 2] * aux[2, 2]

    e = c[1, 0] * aux[1, 0] + aux[1, 0] * c[1, 0] + c[1, 0] * aux[1, 2] + aux[1, 0] * c[1, 2] + aux[2, 1] * c[1, 0] + c[
        2, 1] * aux[1, 0] + aux[2, 2] * c[1, 0] + c[2, 2] * aux[1, 0]
    f = c[1, 1] * aux[1, 1] + aux[1, 1] * c[1, 1] + c[1, 1] * aux[1, 2] + aux[1, 1] * c[1, 2] + aux[2, 1] * c[1, 1] + c[
        2, 1] * aux[1, 1] + aux[2, 2] * c[1, 1] + c[2, 2] * aux[1, 1]
    g = c[1, 2] * aux[1, 2] + aux[1, 2] * c[1, 2] + c[2, 2] * aux[1, 2] + aux[2, 2] * c[1, 2] + aux[2, 2] * c[2, 2] + c[
        2, 2] * aux[2, 2]

    h = c[2, 0] * aux[2, 0] + aux[2, 0] * c[2, 0] + c[2, 0] * aux[2, 2] + aux[2, 0] * c[2, 2] + aux[1, 2] * c[2, 0] + c[
        1, 2] * aux[2, 0]
    i = c[2, 1] * aux[2, 1] + aux[2, 1] * c[2, 1] + c[2, 1] * aux[2, 2] + aux[2, 1] * c[2, 2] + aux[1, 2] * c[2, 0] + c[
        1, 2] * aux[2, 0]
    j = c[2, 2] * aux[2, 2] + aux[2, 2] * c[2, 2]
    test1 = np.array([[a, b, d], [e, f, g], [h, i, j]])

    aux = np.array([[0, 0, 0], c[1, :], [0, 0, 0]])
    a = c[0, 0] * aux[0, 0] + aux[0, 0] * c[0, 0] + c[0, 0] * aux[0, 2] + aux[0, 0] * c[0, 2] + aux[2, 0] * c[0, 0] + c[
        2, 0] * aux[0, 0] + aux[2, 2] * c[0, 0] + c[2, 2] * aux[0, 0]
    b = c[0, 1] * aux[0, 1] + aux[0, 1] * c[0, 1] + c[0, 1] * aux[0, 2] + aux[0, 1] * c[0, 2] + aux[2, 0] * c[0, 1] + c[
        2, 0] * aux[0, 1] + aux[2, 2] * c[0, 1] + c[2, 2] * aux[0, 1]
    d = c[0, 2] * aux[0, 2] + aux[0, 2] * c[0, 2] + c[2, 2] * aux[0, 2] + aux[2, 2] * c[0, 2] + aux[2, 2] * c[2, 2] + c[
        2, 2] * aux[2, 2]

    e = c[1, 0] * aux[1, 0] + aux[1, 0] * c[1, 0] + c[1, 0] * aux[1, 2] + aux[1, 0] * c[1, 2] + aux[2, 1] * c[1, 0] + c[
        2, 1] * aux[1, 0] + aux[2, 2] * c[1, 0] + c[2, 2] * aux[1, 0]
    f = c[1, 1] * aux[1, 1] + aux[1, 1] * c[1, 1] + c[1, 1] * aux[1, 2] + aux[1, 1] * c[1, 2] + aux[2, 1] * c[1, 1] + c[
        2, 1] * aux[1, 1] + aux[2, 2] * c[1, 1] + c[2, 2] * aux[1, 1]
    g = c[1, 2] * aux[1, 2] + aux[1, 2] * c[1, 2] + c[2, 2] * aux[1, 2] + aux[2, 2] * c[1, 2] + aux[2, 2] * c[2, 2] + c[
        2, 2] * aux[2, 2]

    h = c[2, 0] * aux[2, 0] + aux[2, 0] * c[2, 0] + c[2, 0] * aux[2, 2] + aux[2, 0] * c[2, 2] + aux[1, 2] * c[2, 0] + c[
        1, 2] * aux[2, 0]
    i = c[2, 1] * aux[2, 1] + aux[2, 1] * c[2, 1] + c[2, 1] * aux[2, 2] + aux[2, 1] * c[2, 2] + aux[1, 2] * c[2, 0] + c[
        1, 2] * aux[2, 0]
    j = c[2, 2] * aux[2, 2] + aux[2, 2] * c[2, 2]
    test2 = np.array([[a, b, d], [e, f, g], [h, i, j]])

    aux = np.array([[0, 0, 0], [0, 0, 0], c[2, :]])
    a = c[0, 0] * aux[0, 0] + aux[0, 0] * c[0, 0] + c[0, 0] * aux[0, 2] + aux[0, 0] * c[0, 2] + aux[2, 0] * c[0, 0] + c[
        2, 0] * aux[0, 0] + aux[2, 2] * c[0, 0] + c[2, 2] * aux[0, 0]
    b = c[0, 1] * aux[0, 1] + aux[0, 1] * c[0, 1] + c[0, 1] * aux[0, 2] + aux[0, 1] * c[0, 2] + aux[2, 0] * c[0, 1] + c[
        2, 0] * aux[0, 1] + aux[2, 2] * c[0, 1] + c[2, 2] * aux[0, 1]
    d = c[0, 2] * aux[0, 2] + aux[0, 2] * c[0, 2] + c[2, 2] * aux[0, 2] + aux[2, 2] * c[0, 2] + aux[2, 2] * c[2, 2] + c[
        2, 2] * aux[2, 2]

    e = c[1, 0] * aux[1, 0] + aux[1, 0] * c[1, 0] + c[1, 0] * aux[1, 2] + aux[1, 0] * c[1, 2] + aux[2, 1] * c[1, 0] + c[
        2, 1] * aux[1, 0] + aux[2, 2] * c[1, 0] + c[2, 2] * aux[1, 0]
    f = c[1, 1] * aux[1, 1] + aux[1, 1] * c[1, 1] + c[1, 1] * aux[1, 2] + aux[1, 1] * c[1, 2] + aux[2, 1] * c[1, 1] + c[
        2, 1] * aux[1, 1] + aux[2, 2] * c[1, 1] + c[2, 2] * aux[1, 1]
    g = c[1, 2] * aux[1, 2] + aux[1, 2] * c[1, 2] + c[2, 2] * aux[1, 2] + aux[2, 2] * c[1, 2] + aux[2, 2] * c[2, 2] + c[
        2, 2] * aux[2, 2]

    h = c[2, 0] * aux[2, 0] + aux[2, 0] * c[2, 0] + c[2, 0] * aux[2, 2] + aux[2, 0] * c[2, 2]
    i = c[2, 1] * aux[2, 1] + aux[2, 1] * c[2, 1] + c[2, 1] * aux[2, 2] + aux[2, 1] * c[2, 2]
    j = c[2, 2] * aux[2, 2] + aux[2, 2] * c[2, 2]
    test3 = np.array([[a, b, d], [e, f, g], [h, i, j]])

    testf = test1 + test2 + test3
    testf = testf/(testf.sum(axis=1)[...,np.newaxis])
    testf[np.isnan(testf)] = 0

    return testf


# def test_calc_cond_DS(c):
#     res = (c.T/c.sum(axis=1)).T
#     res[np.isnan(res)] = 0
#
#     return res

# def test_calc_cond_DS(c):
#
#     return c


def calc_transDS(m, lx):
    nbc_x, nbc_u = lx.shape
    return np.tile(np.moveaxis((m[np.newaxis,:,:]*lx[:,np.newaxis,:]),0,1).reshape(nbc_u, nbc_u*nbc_x), (nbc_x,1))


def calc_vectDS(pm,lx):
    nbc_x, nbc_u = lx.shape
    return (pm[np.newaxis,...]*lx).reshape((nbc_u*nbc_x))


def pad_gray_im_to_square(im):
    height, width = im.shape
    x = height if height > width else width
    y = height if height > width else width
    square = np.zeros((x, y), np.uint8)
    square[int((y - height) / 2):int(y - (y - height) / 2), int((x - width) / 2):int(x - (x - width) / 2)] = im
    return square


def resize_gray_im_to_square(im):
    height, width = im.shape
    x = height if height > width else width
    y = height if height > width else width
    return cv.resize(im, (x,y))

def calc_cacheDS(lx, hidden, nbc_u1):
    nbc_u2 = lx.shape[1]
    res = np.zeros((hidden.shape[0],nbc_u2))
    for i in range(hidden.shape[0]):
        res[i] = np.any(lx[hidden[i]:(hidden[i]+1)*nbc_u1],axis=0)
    return res




