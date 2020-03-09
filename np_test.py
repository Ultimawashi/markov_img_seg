import numpy as np
import itertools
import cv2 as cv
from math import log2
from utils import generate_semipos_sym_mat, convertcls_vect, np_multivariate_normal_pdf, multinomial_rvs, \
    convert_multcls_vectors, \
    convert_vect_multcls, np_multivariate_normal_pdf_marginal, moving_average, get_peano_index, calc_product, \
    algorithm_u, split_in, cut_diff, calc_matDS, calc_transDS, calc_vectDS, pad_gray_im_to_square, \
    resize_gray_im_to_square
from scipy.stats import norm, multivariate_normal
from scipy.linalg import eig
import seaborn as sns
import matplotlib.pyplot as plt

# data = np.array([[1,2], [3,4], [5,6]]).reshape((-1,2))
# data2 = np.array([1,2,3]).reshape((-1,1))
#
# print(np.mean(data,axis=0), np.cov(data,rowvar=False))
# print(np.mean(data2,axis=0), np.cov(data2,rowvar=False).reshape(1,1))

# data = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]).reshape((-1,1))
# data2= np.array([1,2,3,4])
# data_aug = np.concatenate([np.repeat(data2, 4).reshape(-1,1), data], axis=1)# a = np.array([[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]])
# print(data_aug)

# a = np.array([1,2])
# b = np.array([3,4])
# print(np.outer(a,b).flatten())

# mu = np.array([[[0,0],[1,2]],[[0,1],[1,2]]])
# sig = np.array([[[[1,0.1],[0.1,1]], [[1,0.1],[0.1,1]]], [[[1,0.1],[0.1,1]], [[1,0.1],[0.1,1]]]])
#
# print(np.linalg.det(sig) > 0)

# test = np.array([[1,2,3,4,5,6],[1,2,3,4,5,6],[1,2,3,4,5,6],[1,2,3,4,5,6],[1,2,3,4,5,6],[1,2,3,4,5,6]])
# test = test.reshape((2,3,2,3))
# print(test)
# print(np.sum(test, axis=(1,3)))
# test = np.array([[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]])
#
# print(moving_average(test,4,1))
# nbc_x = 2
# nbc_u = 2**nbc_x
# x = np.stack([convertcls_vect(i, (2,)*nbc_x) for i in range(nbc_u)],axis=0)
# mu = np.array([[0], [1]])
# sig = np.array([[[1]], [[1]]])
# mu_t = np.repeat(mu, nbc_u, axis=0)
# mu_t2 = repeat_DS(mu, x)
#
# sig_t = np.repeat(sig, nbc_u, axis=0)
# sig_t2 = repeat_DS(sig, x)
#

# nbc_x = 2
# nbc_u = 2**nbc_x
# x = np.stack([convertcls_vect(i, (2,)*nbc_x) for i in range(nbc_u)],axis=0)
# aux = np.full((nbc_u - 1,nbc_u - 1), 1 / (2 * (nbc_u - 2)))
# aux = aux - np.diag(np.diag(aux))
# aux = np.diag(np.array([1 / 2] * (nbc_u - 1))) + aux
# a = np.zeros((nbc_u, nbc_u))
# a[1:, 1:] = aux
# calc_matDS
# print(a)
# p = (np.expand_dims(u, axis=1)*x).T.flatten()
# t = calc_matDS(a, x)
# print(np.outer(x,x))
# print(p)
# print(t)
#
# print(sum_matDS(t,nbc_u))

# xdim = 2
# sample = np.array([0,2,5,4,3,6,7,0,1,1,1,4,5,2,3,0,7,4,5,6,0,4,5,6,1,2,3,4,5,6,7])
# res = process_sampleDS(sample, xdim)
# print(res)

# def compnorm(datai, mu, sig):
#     res = np.zeros((mu.shape[0],))
#     for i in range(mu.shape[0]):
#         res[i] = multivariate_normal.pdf(datai, mu[i], sig[i])
#     return res
#
# def compmultinorm(datai, mu, sig):
#     res = np.zeros((mu.shape[0],mu.shape[1]))
#     for i in range(mu.shape[0]):
#         for j in range(mu.shape[1]):
#             res[i,j] = multivariate_normal.pdf(datai, mu[i,j], sig[i,j])
#     return res


# mu = np.array([[0], [1]])
# mu2 = np.array([[[0,1],[2,3]],[[4,5], [6,7]]])
#
# sig = np.array([[[1]], [[1]]])
# sig2 = np.array([[[[1,0.1],[0.1,1]],[[1,0.1],[0.1,1]]],[[[1,0.1],[0.1,1]],[[1,0.1],[0.1,1]]]])
#
# # data = np.random.normal(1,1,1000)
# data = np.array([1,2,3,4,5,6])
# data = data.reshape(-1,1)
# data2 = data.reshape(-1,2)
#
# test = np_multivariate_normal_pdf(data,mu,sig)
# comp11 = np.array([compnorm(e, mu, sig) for e in data])
# print(test)
# print(comp11)
# print(test==comp11)
#
#
# test2 = np_multivariate_normal_pdf(data2,mu2,sig2)
# comp22 = np.array([compmultinorm(e, mu2, sig2) for e in data2])
# print(test2)
# print(comp22)
# print(test2==comp22)


# test =
# pp = np.broadcast(data, mu[np.newaxis,...])
# print(pp.shape)

# test2 = np_multivariate_normal_pdf2(data,mu,sig)
# test3 = np.array([np_multivariate_normal_pdf(e,mu,sig) for e in data])
# print(test2.shape)
# print(test3.shape)
# print(test2)
# print(test3)


# data = np.random.normal(1,1,1000)
# data=data.reshape((1,-1))
# print(data.shape)
# print(np_multivariate_normal_pdf(data,mu,sig))
# test = np.array([[1,2],[3,4]])
# test2 = np.array([[0,1],[2,3],[4,5]])
# test2 = test2.T
# print(test2)
# res1 = test2[np.newaxis,:,:] * test[:,:,np.newaxis]
# res2 = test2[:,np.newaxis,:] * test[:,:,np.newaxis]
#
# for i in range(res1.shape[-1]):
#     print(res1[:,:,i])
#
# for i in range(res2.shape[-1]):
#     print(res2[:,:,i])


# print(a)
# p = (np.expand_dims(u, axis=1)*x).T.flatten()
# t = calc_matDS(a, x)
# print(np.outer(x,x))
# print(p)
# print(t)
#
# print(sum_matDS(t,nbc_u))

# xdim = 2
# sample = np.array([0,2,5,4,3,6,7,0,1,1,1,4,5,2,3,0,7,4,5,6,0,4,5,6,1,2,3,4,5,6,7])
# res = process_sampleDS(sample, xdim)
# print(res)

# for i in range(0, ut.shape[0], a.shape[0]):
#     for j in range(0, ut.shape[1], a.shape[1]):
#         ut[i:i+a.shape[0],j:j+a.shape[1]] = ut[i:i+a.shape[0],j:j+a.shape[1]] * a
#
# ut = np.block(ut)
# print(ut.shape, p.shape, nbc_x*nbc_u)
# print(ut,p)


# test = [0.1,0.2,0.5,0.1,0.1]
# sns.distplot(test)
# plt.show()
#
# listc = [b for a in cut_diff(test) for b in split_in(d,a)]
# list_idx = []
# for i in range(1,len(listc)):
#     list_idx.append((listc[i-1][0], listc[i][0], len(listc[i]) - 1))
#
# res = [np.zeros((x,x,d)) for n in range(len(list_idx))]
#
# for i,a in enumerate(list_idx):
#     res[i][a[0],a[1], a[2]] = 1
#
# res = sum(res)/len(res)
# res = (res.T/np.sum(res, axis=-1)).T
# # res = (res.T / np.sum(res, axis=1)).T
#
# print(res)


# res = algorithm_u(range(10), 2)
#
# fres = [a for a in res if all(len(item) == int(5) for item in a)]
#
# print(fres)
# print(len(fres))


# data = np.array([0,1,2,3])
# test1 = convert_multcls_vectors(data, (2, 2))
# test2 = convert_vect_multcls(test1, (2, 2))
# print(test1,test2)

# nbc_u = 2
# nbc_x = 2
# nb_class = nbc_u*nbc_x
#
# p = np.array([1 / nb_class] * nb_class)
# # a = np.full((nb_class, nbc_x), 1 / (2 * (nbc_x - 1)))
# # b = np.block([[np.eye(nbc_x)] * int(nb_class / nbc_x)]).T * 1 / (2 * (nbc_x - 1))
# # c = np.block([[np.eye(nbc_x)] * int(nb_class / nbc_x)]).T * 1 / 2
# # d = np.block([[np.eye(nbc_x)] * int(nb_class / nbc_x)]).T * 1 / 2
#
#
# a = np.full((nbc_x, nbc_x), 1 / (2 * (nbc_x - 1)))
# a = a - np.diag(np.diag(a))
# a = np.diag(np.array([1 / 2] * nbc_x)) + a
# b = np.repeat(np.eye(nbc_x,nbc_x, k=0),nbc_u,axis=0)
# idx = [i for i in range(b.shape[0]) if ((i+1) % nbc_u == 0)]
# for i,e in enumerate(idx):
#     b[e] = a[i]
# print(b)
# a = np.full((nb_class*nbc_x, nbc_u), 1 / (2 * (nbc_u - 1)))
# a = a - np.block([[np.eye(nbc_u)] * int(nb_class*nbc_x/nbc_u)]).T * 1 / (2 * (nbc_u - 1))
# a = a + np.block([[np.eye(nbc_u)] * int(nb_class*nbc_x/nbc_u)]).T * 1 / 2
# ut = [[np.eye(nbc_u, k=1) for n1 in range(nbc_x)] for n2 in range(int(nb_class/nbc_u))]
# ind=0
# for e in ut:
#     for p in e:
#         p[-1] = a[ind]
#         ind = ind + 1
# ut = np.block(ut)
# print(ut)
# for i in range(1,nbc_x+1):
#     print(b[:,i-1])
#     ut[:,(i-1)*nbc_u:i*nbc_u] = (ut[:,(i-1)*nbc_u:i*nbc_u].T * b[:,i-1]).T
#
# print(ut)


# print(((b.T.flatten()*ut.T)).reshape(nb_class, nb_class))

# res=(b.T.flatten()*ut.T).reshape(nb_class, nb_class)
#
# print(res)

#
# f = (d.flatten()*ut.T).T.reshape(nb_class,nb_class)
#
# print(f)
# test = moving_average(a,(2,2))
# print(test)

# a = np.array([1,2,3,4]).reshape(-1,1)
# b = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]).reshape(-1,1)
#
# c = np.concatenate([np.repeat(a, 4).reshape(-1,1), b], axis=1)
#
#
# print(a.shape,b.shape, c.shape)
# print(c)


# mu = np.array([[[0,1],[1,0]],[[1,0],[0,1]]])
# sigma = np.array([[[[1,0.5],[0.5,1]],[[1,0.5],[0.5,1]]],[[[1,0.5],[0.5,1]],[[1,0.5],[0.5,1]]]])
#
# x1 =np.array([1.3,0.8])
#
# test = np_multivariate_normal_pdf(x1,mu, sigma)
# test2 = np_multivariate_normal_pdf_marginal(x1[0], mu, sigma, 0)


# test=np.array([[0,0,0],[0,0,0],[0,0,0],[0,0,0],[1,0,1],[1,0,1],[1,0,1],[1,0,1]])
#
# rev_test= test[::4]
# print(test)
# print(rev_test)
#
# print((1,) + (4,))
# nb_class = 10
# b = np.full((nb_class,nb_class),{'mu':np.zeros((2)), 'sigma':np.zeros((2,2))})
# print(b)
# numbers = dict(x=5, y=0)
# print('numbers = ',numbers)
# print(type(numbers))
# ae = np.array([[-1, 1], [-1, 1]])
# p1e = np.array([[1, -1], [1, -1]])
# print(ln_sum(ae-p1e,p1e-ae))
# print(np.NINF - np.NINF)


# p = np.array([[0.75,0.25],[0.5,0.5],[0.60,0.40],[0.1,0.9],[0.3,0.7], [0.75,0.25],[0.5,0.5],[0.60,0.40],[0.1,0.9],[0.3,0.7], [0.75,0.25],[0.5,0.5],[0.60,0.40],[0.1,0.9],[0.3,0.7]])
# test = multinomial_rvs(1,p)
# test2 = np.argmax(test, axis=1)
# test3 = np.stack((test2[:-1], test2[1:]), axis=1)
# print(test3.shape, test3.T.shape)
# test4 = np.indices((2,2))
# test5 = np.stack((test4[0], test4[1]), axis=-1)
# test6 = np.moveaxis(test4,0,-1)
#
# print(test5, test6)
# broadc = (len(test5.shape) - len(test3.shape) + 1)
# test3 = test3.reshape((test3.shape[0],) + (1,) * broadc + test3.shape[1:])
# test6 = (test3 == test5)
# test7 = np.all(test6, axis=-1)
# print(test3)
# print(test7.sum(axis=0))
# data = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14])
# data = data.reshape(-1,1)
# mu = np.array([[0], [1]])
# broadc = (len(mu.shape) - len(data.shape) + 1)
# test3 = (test2[...,np.newaxis] == np.indices((mu.shape[0],)))*data
# print(test2.shape)
# print((test2[...,np.newaxis] == np.indices((mu.shape[0],))).shape)
# print(test3.shape)
# print(test3)
# test = np.array([[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]])
# test2 = test[1:,:,np.newaxis] * test[1:].sum(axis=1)[:,np.newaxis,np.newaxis]
# print(test[1:,:,np.newaxis])
# print(test2)

# nbc_x=2
# nbc_u=(nbc_x**2) - 1
# test = np.array([[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]])
# test2 = np.array([1,2,3,4,5,6])
# mu = np.array([[0], [1]])
# sig = np.array([[[1]], [[1]]])
# lx = np.stack([convertcls_vect(i + 1, (2,) * nbc_x) for i in range(nbc_u)], axis=0).T
# card = 1 / np.sum(lx, axis=0)
# m = np.array([[0.10, 0.04, 0.01],[0.04, 0.74, 0.01],[0.02, 0.02, 0.02]])
# test1 = calc_matDS(m,lx)
# test2 = np.moveaxis((m[np.newaxis,:,:]*lx[:,np.newaxis,:]),0,1).reshape(nbc_u, nbc_u*nbc_x)
# test3 = np.tile(np.moveaxis((m[np.newaxis,:,:]*lx[:,np.newaxis,:]),0,1).reshape(nbc_u, nbc_u*nbc_x), (nbc_x,1))
# print(test1.shape, test1)
# print(test2)
# print(test3.shape, test3)

# test1 =(m[np.newaxis,:,:]*lx[:,:,np.newaxis]).reshape(nbc_u*nbc_x, nbc_u)
# test2 = test1[np.newaxis,...]*lx[:,np.newaxis,:]
# test3 = np.moveaxis(test2,0,1)
# test4 = np.moveaxis((m[np.newaxis,:,:]*lx[:,:,np.newaxis]).reshape(nbc_u*nbc_x, nbc_u)[np.newaxis,...]*lx[:,np.newaxis,:],0,1).reshape(nbc_u*nbc_x, nbc_u*nbc_x)
# print(test4)
# test5 = np.outer(card,card) * test4.reshape(nbc_x,nbc_u,nbc_x,nbc_u).sum(axis=(0,2))
# print(test5.sum(axis=1))
# print(m.sum(axis=1))
# test5 = (pm[np.newaxis,...]*lx).reshape((nbc_u*nbc_x))
# print(pm)
# print(card * test5.reshape((nbc_x,nbc_u)).sum(axis=0))
# test6 = np.stack([test4, test4, test4, test4],axis=0)
# print((1/test6.shape[0]) * np.outer(card,card) * test6.reshape((test6.shape[0],nbc_x,nbc_u,nbc_x,nbc_u)).sum(axis=(0,1,3)))
# nbc_x = 2
# nbc_u1 = 2**nbc_x - 1
# nbc_u2 = 5
# u1 = np.array([[0.19, 0.01, 0], [0.01, 0.79, 0], [0, 0, 0]])
# # u1 = np.array([[0.18, 0.009, 0.001], [0.009, 0.78, 0.001], [0.0075, 0.0075, 0.005]])
# u2 = np.array([[[0.1, 0.2, 0.4, 0.2, 0.1], [0.1, 0.2, 0.4, 0.2, 0.1], [0.1, 0.2, 0.4, 0.2, 0.1]],
#                [[0.1, 0.2, 0.4, 0.2, 0.1], [0.1, 0.2, 0.4, 0.2, 0.1], [0.1, 0.2, 0.4, 0.2, 0.1]],
#                [[0.1, 0.2, 0.4, 0.2, 0.1], [0.1, 0.2, 0.4, 0.2, 0.1], [0.1, 0.2, 0.4, 0.2, 0.1]]])
# mu = np.array([[0], [1]])
# sigma = np.array([[[1]], [[1]]])
#
# lx = np.repeat(np.stack([convertcls_vect(i + 1, (2,) * nbc_x) for i in range(nbc_u1)], axis=0),
#                             nbc_u2,
#                             axis=0).T
# nb_class = nbc_u1 * nbc_u2
# p = (np.sum(u1 * u2.T, axis=1)).T.flatten()
# a = (u1.T / np.sum(u1, axis=1)).T
# a[np.isnan(a)] = 0
# print(p)
# print(a)
# b = np.repeat(np.eye(nbc_u1, nbc_u1, k=0), nbc_u2, axis=0)
# idx = [i for i in range(b.shape[0]) if ((i + 1) % nbc_u2 == 0)]
# for i, e in enumerate(idx):
#     b[e] = a[i]
#
# a = u2
# ut = [[np.eye(nbc_u2, k=1) for n1 in range(nbc_u1)] for n2 in range(int(nb_class / nbc_u2))]
# for i, e in enumerate(ut):
#     for j, p in enumerate(e):
#         p[-1] = a[i, j]
# ut = np.block(ut)
# print(ut)
# for i in range(1, nbc_u1 + 1):
#     ut[:, (i - 1) * nbc_u2:i * nbc_u2] = (
#             ut[:, (i - 1) * nbc_u2:i * nbc_u2].T * b[:, i - 1]).T
# t = ut
# t[np.isnan(t)] = 0
# mu = mu
# sigma = sigma
#
# c = np.array([[0.19, 0.01], [0.01, 0.79]])
# u = np.array([[[0.1, 0.2, 0.4, 0.2, 0.1], [0.1, 0.2, 0.4, 0.2, 0.1]],[[0.1, 0.2, 0.4, 0.2, 0.1], [0.1, 0.2, 0.4, 0.2, 0.1]]])
# nbc_u = 5
# nb_class = nbc_x * nbc_u
# p = (np.sum(c * u.T, axis=1)).T.flatten()
# a = (c.T / np.sum(c, axis=1)).T
# print(p)
# print(a)
# b = np.repeat(np.eye(nbc_x,nbc_x, k=0), nbc_u, axis=0)
# idx = [i for i in range(b.shape[0]) if ((i + 1) % nbc_u == 0)]
# for i, e in enumerate(idx):
#     b[e] = a[i]
#
# a = u
# ut = [[np.eye(nbc_u, k=1) for n1 in range(nbc_x)] for n2 in range(int(nb_class / nbc_u))]
# for i, e in enumerate(ut):
#     for j, p in enumerate(e):
#         p[-1] = a[i, j]
# ut = np.block(ut)
# print(ut)
# for i in range(1, nbc_x + 1):
#     ut[:, (i - 1) * nbc_u:i * nbc_u] = (
#             ut[:, (i - 1) * nbc_u:i * nbc_u].T * b[:, i - 1]).T
# t2 = ut
# mu = mu
# sigma = sigma
# print(t2)
# print(t)


# print(sum_DSvect(test2, 2))
# print(test2.reshape(3,2))
# print(test2.reshape(3,2).sum(axis=0))
# length = 100
# c = np.array([[0.19, 0.01], [0.01, 0.79]])
# mu = np.array([[[5], [10]],[[15], [20]]])
# sig = np.array([[[[3]], [[3]]], [[[3]], [[3]]]])
# dt = np.array([i for i in range(1,length+1)])
# dt = dt.reshape(-1,1)
# uprime = np_multivariate_normal_pdf(dt,mu,sig).T.reshape(-1,length)
# uprime = (uprime.T/uprime.sum(axis=1)).T
# print(uprime)

# mu = np.array([[0], [0,1]])
# sig = np.array([[[1]], [[[1,0.5],[0.5,1]]]])
# data = np.array([[1], [0,1]])
# print(np_multivariate_normal_pdf(data,mu,sig))

# img = cv.imread('./img/' + 'promenade2' + '.bmp')  # Charger l'image
# img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# test1 = pad_gray_im_to_square(img)
# test2 = resize_gray_im_to_square(img)
#
# cv.imshow("original_img", img)
# cv.imshow('img_padded',test1)
# cv.imshow("resized", test2)
# cv.waitKey(0)

# ttest = np.array([[0,1],[0,1],[0,1],[0,1],[0,1]])
# print(ttest)
# print(ttest.reshape((10,)))
# print(ttest.T.reshape((10,)))
# print(ttest.T.reshape((10,)).T)



# u = np.array([[0,0.2,0,0,0],[0,0,0.2,0,0],[0,0,0,0.2,0],[0,0,0,0,0.2],[0.04,0.04,0.04,0.04,0.04]])
# print(u.sum())
#
#
# hidden = np.array(
#     [0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0,
#      1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1])
#
# hiddenc = np.stack((hidden[:-1], hidden[1:]), axis=-1)
# uprime = np.repeat(u[np.newaxis,:,:],hiddenc.shape[0], axis=0)
# aux = np.moveaxis(np.indices((2,2)), 0, -1)
# broadc = (len(aux.shape) - len(hiddenc.shape) + 1)
# testhidden = np.all(hiddenc.reshape((hiddenc.shape[0],) + (1,) * broadc + hiddenc.shape[1:]) == aux, axis=-1)
#
#
# test = (uprime[:,np.newaxis,:,np.newaxis,:]*testhidden[:,:,np.newaxis,:,np.newaxis]).sum(axis=0)/(uprime[:,np.newaxis,:,np.newaxis,:]).sum(axis=0)
# test[np.isnan(test)]=0
# test2 = u[np.newaxis,:,np.newaxis,:]*test
# print()
# c = test2.reshape(10,10)
# print(test2)
# print(c)
#
# test2 = (((psi[:,:,np.newaxis] * (hidden[..., np.newaxis] == np.indices((2,)))[:,np.newaxis,:]).sum(axis=0)) / (psi[:,:,np.newaxis].sum(axis=0)))
#
# p = (test2*p[:,np.newaxis]).T.reshape((20,))
# p[np.isnan(p)]=0
#
#
# print(p)
#
nbc_x = 2
nbc_u = 5

lawu = np.random.uniform(0, 1, (nbc_x, nbc_x, nbc_u))
max_vals = np.sum(lawu, axis=-1)
lawu = lawu / max_vals[:, :, np.newaxis]
print(lawu.sum(axis=-1))

# nbc_u2 = (nbc_u1 * nbc_x) + 1
# lx = np.vstack((np.eye(nbc_x * nbc_u1), np.ones((nbc_x * nbc_u1,)))).T
# hidden = np.array(
#     [0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0,
#      1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1])
#
# def calc_cacheDS(lx, hidden, nbc_u1):
#     nbc_u2 = lx.shape[1]
#     res = np.zeros((hidden.shape[0],nbc_u2))
#     for i in range(hidden.shape[0]):
#         res[i] = np.any(lx[hidden[i]*nbc_u1:(hidden[i]+1)*nbc_u1],axis=0)
#     return res
#
# print(calc_cacheDS(lx, hidden, nbc_u1))


# p = np.sum(u, axis=1)
# t = (u.T / p).T
# t[np.isnan(t)] = 0
# print(t)
# # print(calc_transDS(t,lx))
# print(calc_matDS(t,lx))


# card = np.sum(lx, axis=0)
# card[card == np.inf] = 0
#
# lxprime = lx*alpha
# print(lxprime)
# a = np.full((nbc_x, nbc_x), 1 / (2 * (nbc_x - 1)))
# a = a - np.diag(np.diag(a))
# t = np.diag(np.array([1 / 2] * nbc_x)) + a
# p = np.array([1 / nbc_x] * nbc_x)
# c = (t.T * p).T
# u = lxprime.T @ c @ lxprime
# print(u)
# print(u.sum(axis=1))
# print(u.sum())

# p1 = np.array([0.0285119390799247, 0.7093357036556708, 0.13107617863220403])
# t1 = np.array([[0.8742852780157239, 0.03892534182224672, 0.043233022859543516],
#                [0.0007387209432667431, 0.993904361623281, 0.002678444258580758],
#                [0.011673380291735034, 0.01206545968793269, 0.48803179060103374]])
#
# p2 = np.array([0.029290925944635404, 0.7091874372261652, 0.13076081841460221])
# t2 = np.array([[0.8739170170338303, 0.0015991733168090386, 0.009772741921187569],
#                [0.017708644024864485, 0.9939215117893622, 0.014503807262382387],
#                [0.05466327654926296, 0.002198102557111149, 0.48792114622540267]])


# epsilon = 0.005
# alpha = 0.0001
# c = np.array([[0.19, 0.01], [0.01, 0.79]])
# u = np.array([[c[0, 0] - epsilon, c[0, 1] - alpha, (alpha / 2)], [c[1, 0] - alpha, c[1, 1] - epsilon, (epsilon / 2)],
#               [(alpha / 2), (epsilon / 2), alpha + epsilon]])
# ut = (u.T/u.sum(axis=1)).T
# p = (u*card).sum(axis=1)
#
#
# testut = calc_transDS(ut,lx)
# testp = calc_vectDS(p, lx)
# testf = testp@testut
#
# print(testp)
# print(testf)

# u = np.array([[0.0205, 0, 0.0028],[0, 0.7074, 0.0045],[0.0028, 0.0045, 0.2573]])


# test8 = calc_vectDS(psecond,lx)
# test2 = test1.sum(axis=1)
# test3 = (test1.T/test2).T
# test3[np.isnan(test3)]=0
# test4 = calc_transDS(t,lx)
# pprime= test2.reshape(nbc_x,nbc_u).sum(axis=0)
# print(test2)
# print(test2@test3)
# print(test4)
# print(p)
# print(pprime)
# print(psecond)
# print(test8)
# print(test8@test4)
# a = np.array([[1,2],[3,1]])
# b = np.array([3,4])
#
# print(a@b.T)
# print(b@a)
# print(a@b)

# print(test11,test12,test13)
# print(test21,test22,test23)

# perturbation_param=0.5
# nbc_x = 2
# nbc_u1 = 5
# nbc_u2 = nbc_x*nbc_u1 + 1
#
#
# lx = np.vstack((np.eye(nbc_x*nbc_u1), np.ones((nbc_x*nbc_u1,)))).T
# card = 1 / np.sum(lx, axis=0)
#
# u1 = np.ones((nbc_x, nbc_x, nbc_u1)) * (1 / nbc_u1)
# a = np.full((nbc_x, nbc_x), 1 / (2 * (nbc_x - 1)))
# a = a - np.diag(np.diag(a))
# x = np.array([1 / nbc_x] * nbc_x)
# a = np.diag(np.array([1 / 2] * nbc_x)) + a
# p1 = (np.sum((a.T * x).T * u1.T, axis=1)).T.flatten()
# b = np.repeat(np.eye(nbc_x, nbc_x, k=0), nbc_u1, axis=0)
# idx = [i for i in range(b.shape[0]) if ((i + 1) % nbc_u1 == 0)]
# for i, e in enumerate(idx):
#     b[e] = a[i]
# a = u1
# ut = [[np.eye(nbc_u1, k=1) for n1 in range(nbc_x)] for n2 in range(int((nbc_x*nbc_u1) / nbc_u1))]
# for i, e in enumerate(ut):
#     for j, p in enumerate(e):
#         p[-1] = a[i, j]
# ut = np.block(ut)
# for i in range(1, nbc_x + 1):
#     ut[:, (i - 1) * nbc_u1:i * nbc_u1] = (
#             ut[:, (i - 1) * nbc_u1:i * nbc_u1].T * b[:, i - 1]).T
# t1 = ut
#
# c = (t1.T*p1).T
#
# index1 = lx.T.sum(axis=1) == 1
# index2 = lx.T.sum(axis=1) != 1
# res = np.copy(lx.T)
# res[index1] = res[index1] * (1 - perturbation_param)
# res[index2] = (res[index2].T * perturbation_param).T
# u2 = res @ c @ res.T
#
# p2 = card * u2.sum(axis=1)
# t2 = ((np.outer(card, card) * u2).T / p2).T
#
# test = calc_transDS(t2, lx)
# print(test)

# card = 1 / np.sum(lx, axis=0)
# nb_class = self.nbc_u1 * self.nbc_u2
# pu1 = (card * np.sum(u1, axis=1))
# a = ((np.outer(card, card) * u1).T / pu1).T
# self.p = (np.sum((a.T * pu1).T * u2.T, axis=1)).T.flatten()
# b = np.repeat(np.eye(self.nbc_u1, self.nbc_u1, k=0), self.nbc_u2, axis=0)
# idx = [i for i in range(b.shape[0]) if ((i + 1) % self.nbc_u2 == 0)]
# for i, e in enumerate(idx):
#     b[e] = a[i]
#
# a = u2
# ut = [[np.eye(self.nbc_u2, k=1) for n1 in range(self.nbc_u1)] for n2 in range(int(nb_class / self.nbc_u2))]
# for i, e in enumerate(ut):
#     for j, p in enumerate(e):
#         p[-1] = a[i, j]
# ut = np.block(ut)
# for i in range(1, self.nbc_u1 + 1):
#     ut[:, (i - 1) * self.nbc_u2:i * self.nbc_u2] = (
#             ut[:, (i - 1) * self.nbc_u2:i * self.nbc_u2].T * b[:, i - 1]).T
# self.t = ut
# self.t[np.isnan(self.t)] = 0
# self.mu = mu
# self.sigma = sigma
