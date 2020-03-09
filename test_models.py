import numpy as np

np.set_printoptions(threshold=np.inf)
from utils import get_peano_index, convert_multcls_vectors, moving_average, calc_matDS
from dbn import DBN, RBM_dtod, RBM_ctod
from hmm import HMC_ctod, HMC_multiR_ctod, HSMC_ctod, HSMC_class_ctod, HEMC_ctod, HESMC_ctod, HSEMC_ctod, HEMC2_ctod
from pmm import PMC_ctod, PSMC_ctod
from tmm import TMC_ctod, GSMC_ctod
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# terr = {}
sample_length = 1000
#
# c = np.array([[0.19, 0.01], [0.01, 0.79]])
# mu = np.array([[0], [1]])
# sig = np.array([[[1]], [[1]]])
#
# hmc = HMC_ctod(2)
# hmc.give_param(c,mu,sig)
# sample_hidden, sample_visible = hmc.generate_sample(sample_length)
# seg_hidden = hmc.seg_mpm(sample_visible)
# err = np.sum(seg_hidden != sample_hidden) / sample_length
# terr['hmc'] = (err <= 0.5) * err + (err > 0.5) * (1 - err)
# hmc.init_data_prior(sample_visible)
# print({'iter': 0, 'p': hmc.p, 't': hmc.t, 'mu': hmc.mu, 'sigma': hmc.sigma})
# hmc.init_kmeans(sample_visible)
# print({'iter': 0, 'p': hmc.p, 't': hmc.t, 'mu': hmc.mu, 'sigma': hmc.sigma})
# hmc.init_data_prior(sample_visible)
# hmc.get_param_EM(sample_visible, 100)
# seg_hidden = hmc.seg_mpm(sample_visible)
# err = np.sum(seg_hidden != sample_hidden) / sample_length
# terr['hmc_estim_param_EM'] = (err <= 0.5) * err + (err > 0.5) * (1 - err)
# hmc.init_data_prior(sample_visible, 2)
# hmc.get_param_ICE(sample_visible, 100, 10)
# seg_hidden = hmc.seg_mpm(sample_visible)
# err = np.sum(seg_hidden != sample_hidden) / sample_length
# terr['hmc_estim_param_ICE'] = (err <= 0.5) * err + (err > 0.5) * (1 - err)
# hmc.init_data_prior(sample_visible, 2)
# hmc.get_param_SEM(sample_visible, 100)
# seg_hidden = hmc.seg_mpm(sample_visible)
# err = np.sum(seg_hidden != sample_hidden) / sample_length
# terr['hmc_estim_param_SEM'] = (err <= 0.5) * err + (err > 0.5) * (1 - err)
#
# u = np.array([[[0.1, 0.2, 0.4, 0.2, 0.1], [0.1, 0.2, 0.4, 0.2, 0.1]],[[0.1, 0.2, 0.4, 0.2, 0.1], [0.1, 0.2, 0.4, 0.2, 0.1]]])
# mu = np.array([[0], [1]])
# sig = np.array([[[1]], [[1]]])
# hsmc = HSMC_ctod(nbc_x=2, nbc_u=5)
# hsmc.give_param(c,u,mu,sig)
# sample_hidden, sample_visible = hsmc.generate_sample(sample_length, x_only=True)
# sample_visible = sample_visible.reshape(-1, 1)
# seg_hidden = hsmc.seg_mpm(sample_visible)
# err = np.sum(seg_hidden != sample_hidden) / sample_length
# terr['hsmc'] = (err <= 0.5) * err + (err > 0.5) * (1 - err)
# hsmc.init_data_prior(sample_visible, 2)
# print({'iter': 0, 'p': hsmc.p, 't': hsmc.t, 'mu': hsmc.mu, 'sigma': hsmc.sigma})
# hsmc.init_kmeans(sample_visible)
# print({'iter': 0, 'p': hsmc.p, 't': hsmc.t, 'mu': hsmc.mu, 'sigma': hsmc.sigma})
# hsmc.init_data_prior(sample_visible, 2)
# hsmc.get_param_EM(sample_visible, 100)
# seg_hidden = hsmc.seg_mpm(sample_visible)
# err = np.sum(seg_hidden != sample_hidden) / sample_length
# terr['hsmc_estim_param_EM'] = (err <= 0.5) * err + (err > 0.5) * (1 - err)
# hsmc.init_data_prior(sample_visible, 2)
# hsmc.get_param_ICE(sample_visible, 100, 10)
# seg_hidden = hsmc.seg_mpm(sample_visible)
# err = np.sum(seg_hidden != sample_hidden) / sample_length
# terr['hsmc_estim_param_ICE'] = (err <= 0.5) * err + (err > 0.5) * (1 - err)
# hsmc.init_data_prior(sample_visible, 2)
# hsmc.get_param_SEM(sample_visible, 100)
# seg_hidden = hsmc.seg_mpm(sample_visible)
# err = np.sum(seg_hidden != sample_hidden) / sample_length
# terr['hsmc_estim_param_SEM'] = (err <= 0.5) * err + (err > 0.5) * (1 - err)
#

c = np.array([[0.20, 0.05],
              [0.05, 0.70]])
epsilon = 0
alpha = 0
u = np.array([[c[0, 0] - epsilon, c[0, 1] - alpha, (alpha / 2)], [c[1, 0] - alpha, c[1, 1] - epsilon, (epsilon / 2)],
              [(alpha / 2), (epsilon / 2), alpha + epsilon]])
mu = np.array([[0], [1]])
sig = np.array([[[1]], [[1]]])
hemc = HEMC_ctod(nbc_x=2)
hemc.give_param(u, mu, sig)
print(hemc.p, hemc.t)
sample_hidden, sample_visible = hemc.generate_sample(sample_length, x_only=True)
plt.plot(sample_hidden)
plt.show()
# sample_visible = sample_visible.reshape(-1, 1)
# seg_hidden = hemc.seg_mpm(sample_visible)
# err = np.sum(seg_hidden != sample_hidden) / sample_length
# terr['hemc'] = (err <= 0.5) * err + (err > 0.5) * (1 - err)
# hemc.init_data_prior(sample_visible)
# print({'iter': 0, 'p': hemc.p, 't': hemc.t, 'mu': hemc.mu, 'sigma': hemc.sigma})
# hemc.init_kmeans(sample_visible)
# print({'iter': 0, 'p': hemc.p, 't': hemc.t, 'mu': hemc.mu, 'sigma': hemc.sigma})
# hemc.init_kmeans(sample_visible)
# hemc.get_param_EM(sample_visible,100)
# seg_hidden = hemc.seg_mpm(sample_visible)
# err = np.sum(seg_hidden != sample_hidden)/sample_length
# terr['hemc_estim_param_EM'] = (err <= 0.5) * err + (err > 0.5) * (1 - err)
# hemc.init_data_prior(sample_visible)
# hemc.get_param_ICE(sample_visible,100,10)
# seg_hidden = hemc.seg_mpm(sample_visible)
# err = np.sum(seg_hidden != sample_hidden)/sample_length
# terr['hemc_estim_param_ICE'] = (err <= 0.5) * err + (err > 0.5) * (1 - err)
# hemc.init_data_prior(sample_visible)
# hemc.get_param_SEM(sample_visible,100)
# seg_hidden = hemc.seg_mpm(sample_visible)
# err = np.sum(seg_hidden != sample_hidden)/sample_length
# terr['hemc_estim_param_SEM'] = (err <= 0.5) * err + (err > 0.5) * (1 - err)
#
#


# hmc.get_param_EM(sample_visible, 100)
# seg_hidden = hmc.seg_mpm(sample_visible)
# err = np.sum(seg_hidden != sample_hidden) / sample_length
# terr['hmc_estim_param_EM'] = (err <= 0.5) * err + (err > 0.5) * (1 - err)
# hmc.init_data_prior(sample_visible, 2)
# hmc.get_param_ICE(sample_visible, 100, 10)
# seg_hidden = hmc.seg_mpm(sample_visible)
# err = np.sum(seg_hidden != sample_hidden) / sample_length
# terr['hmc_estim_param_ICE'] = (err <= 0.5) * err + (err > 0.5) * (1 - err)
# hmc.init_data_prior(sample_visible, 2)
# hmc.get_param_SEM(sample_visible, 100)
# seg_hidden = hmc.seg_mpm(sample_visible)
# err = np.sum(seg_hidden != sample_hidden) / sample_length
# terr['hmc_estim_param_SEM'] = (err <= 0.5) * err + (err > 0.5) * (1 - err)
# #
# terr = {}
# sample_length = 1000
# c = np.array([[0.19, 0.01], [0.01, 0.79]])
# u = np.array([[[0.1, 0.2, 0.4, 0.2, 0.1], [0.1, 0.2, 0.4, 0.2, 0.1]],[[0.1, 0.2, 0.4, 0.2, 0.1], [0.1, 0.2, 0.4, 0.2, 0.1]]])
# mu = np.array([[0], [1]])
# sig = np.array([[[1]], [[1]]])
# hsmc = HSMC_ctod(nbc_x=2, nbc_u=5)
# hsmc.give_param(c,u,mu,sig)
# sample_hidden, sample_visible = hsmc.generate_sample(sample_length, x_only=True)
# sample_visible = sample_visible.reshape(-1, 1)
# seg_hidden = hsmc.seg_mpm(sample_visible)
# err = np.sum(seg_hidden != sample_hidden) / sample_length
# terr['hsmc'] = (err <= 0.5) * err + (err > 0.5) * (1 - err)
# hsmc.init_data_prior(sample_visible, 2)
# hsmc.get_param_supervised(sample_visible,sample_hidden, 100)
# seg_hidden = hsmc.seg_mpm(sample_visible)
# err = np.sum(seg_hidden != sample_hidden) / sample_length
# terr['hsmc_estim_param_sup'] = (err <= 0.5) * err + (err > 0.5) * (1 - err)
# print(terr)


# sample_hidden = convert_multcls_vectors(sample_hidden, (hsmc.nbc_u, hsmc.nbc_x))[:, 1]
# sample_visible = sample_visible.reshape(-1, 1)
# seg_hidden = hsmc.seg_mpm(sample_visible)
# err = np.sum(seg_hidden != sample_hidden) / sample_length
# terr['hsmc'] = (err <= 0.5) * err + (err > 0.5) * (1 - err)
# hsmc.init_data_prior(sample_visible, 2)
# hsmc.get_param_EM(sample_visible, 100)
# seg_hidden = hsmc.seg_mpm(sample_visible)
# err = np.sum(seg_hidden != sample_hidden) / sample_length
# terr['hsmc_estim_param_EM'] = (err <= 0.5) * err + (err > 0.5) * (1 - err)
# hsmc.init_data_prior(sample_visible, 2)
# hsmc.get_param_ICE(sample_visible, 100, 10)
# seg_hidden = hsmc.seg_mpm(sample_visible)
# err = np.sum(seg_hidden != sample_hidden) / sample_length
# terr['hsmc_estim_param_ICE'] = (err <= 0.5) * err + (err > 0.5) * (1 - err)
# hsmc.init_data_prior(sample_visible, 2)
# hsmc.get_param_SEM(sample_visible, 100)
# seg_hidden = hsmc.seg_mpm(sample_visible)
# err = np.sum(seg_hidden != sample_hidden) / sample_length
# terr['hsmc_estim_param_SEM'] = (err <= 0.5) * err + (err > 0.5) * (1 - err)
# #
# c = np.array([[0.20, 0.05], [0.05, 0.70]])
# u = np.array([[[0.6, 0.2, 0.1, 0.05, 0.05], [0.6, 0.2, 0.1, 0.05, 0.05]],[[0.6, 0.2, 0.1, 0.05, 0.05], [0.6, 0.2, 0.1, 0.05, 0.05]]])
# u = np.array([[[0.2, 0.2, 0.2, 0.2, 0.2], [0.2, 0.2, 0.2, 0.2, 0.2]],[[0.2, 0.2, 0.2, 0.2, 0.2], [0.2, 0.2, 0.2, 0.2, 0.2]]])
# u = np.array([[[0.05, 0.05, 0.1, 0.2, 0.6], [0.05, 0.05, 0.1, 0.2, 0.6]],[[0.05, 0.05, 0.1, 0.2, 0.6], [0.05, 0.05, 0.1, 0.2, 0.6]]])
# mu = np.array([[0], [1]])
# sig = np.array([[[1]], [[1]]])
# hsmc2 = HSMC_ctod(2,5)
# hsmc2.give_param(c,u,mu,sig)
# sample_hidden, sample_visible = hsmc2.generate_sample(sample_length)
# plt.plot(sample_hidden)
# plt.show()
# # sample_hidden = convert_multcls_vectors(sample_hidden, (hsmc2.nbc_u, hsmc2.nbc_x))[:, 1]
# # sample_visible = sample_visible.reshape(-1, 1)
# # seg_hidden = hsmc2.seg_mpm(sample_visible)
# # err = np.sum(seg_hidden != sample_hidden) / sample_length
# # terr['hsmc_class'] = (err <= 0.5) * err + (err > 0.5) * (1 - err)
# # hsmc2.init_data_prior(sample_visible, 2)
# # hsmc2.get_param_EM(sample_visible, 100)
# # seg_hidden = hsmc2.seg_mpm(sample_visible)
# # err = np.sum(seg_hidden != sample_hidden) / sample_length
# # terr['hsmc_class_estim_param_EM'] = (err <= 0.5) * err + (err > 0.5) * (1 - err)
# # hsmc2.init_data_prior(sample_visible, 2)
# # hsmc2.get_param_ICE(sample_visible, 100, 10)
# # seg_hidden = hsmc2.seg_mpm(sample_visible)
# # err = np.sum(seg_hidden != sample_hidden) / sample_length
# # terr['hsmc_class_estim_param_ICE'] = (err <= 0.5) * err + (err > 0.5) * (1 - err)
# # hsmc2.init_data_prior(sample_visible, 2)
# # hsmc2.get_param_SEM(sample_visible, 100)
# # seg_hidden = hsmc2.seg_mpm(sample_visible)
# # err = np.sum(seg_hidden != sample_hidden) / sample_length
# # terr['hsmc_class_estim_param_SEM'] = (err <= 0.5) * err + (err > 0.5) * (1 - err)
# #
# epsilon = 0.01
# alpha = 0.005
# c = np.array([[0.19, 0.01], [0.01, 0.79]])
# u = np.array([[c[0, 0] - epsilon, c[0, 1] - alpha, (alpha / 2)], [c[1, 0] - alpha, c[1, 1] - epsilon, (epsilon / 2)],
#               [(alpha / 2), (epsilon / 2), alpha + epsilon]])
# ut = (u.T / u.sum(axis=1)).T
# phi1 = np.array([[u[0, 0], 0, u[0, 2], 0, u[0, 1], u[0, 2]],
#                  [0, 0, 0, 0, 0, 0],
#                  [u[2, 0], 0, u[2, 2], 0, u[2, 1], u[2, 2]],
#                  [0, 0, 0, 0, 0, 0],
#                  [u[1, 0], 0, u[1, 2], 0, u[1, 1], u[1, 2]],
#                  [u[2, 0], 0, u[2, 2], 0, u[2, 1], u[2, 2]]])
# phi2 = np.array([[ut[0, 0], 0, ut[0, 2], 0, ut[0, 1], ut[0, 2]],
#                  [ut[1, 0], 0, ut[1, 2], 0, ut[1, 1], ut[1, 2]],
#                  [ut[2, 0], 0, ut[2, 2], 0, ut[2, 1], ut[2, 2]],
#                  [ut[0, 0], 0, ut[0, 2], 0, ut[0, 1], ut[0, 2]],
#                  [ut[1, 0], 0, ut[1, 2], 0, ut[1, 1], ut[1, 2]],
#                  [ut[2, 0], 0, ut[2, 2], 0, ut[2, 1], ut[2, 2]]
#                  ])
# phi2prime = np.array([[ut[0, 0], 0, ut[0, 2], 0, ut[0, 1], ut[0, 2]],
#                  [0, 0, 0, 0, 0, 0],
#                  [ut[2, 0], 0, ut[2, 2], 0, ut[2, 1], ut[2, 2]],
#                  [0, 0, 0, 0, 0, 0],
#                  [ut[1, 0], 0, ut[1, 2], 0, ut[1, 1], ut[1, 2]],
#                  [ut[2, 0], 0, ut[2, 2], 0, ut[2, 1], ut[2, 2]]
#                  ])
# print(phi2.sum(axis=1))
# print(phi2prime.sum(axis=1))
# beta3 = np.array([1,1,1,1,1,1])
# beta2 = phi2 @ beta3
# beta1 = phi2 @ beta2
# p1 = beta1/beta1.sum()
# p21 = ((phi2*beta2).T/beta1).T
# p21[np.isnan(p21)] = 0
# p32 = ((phi2*beta3).T/beta2).T
# p32[np.isnan(p21)] = 0
# print(p21)
# print(p32)
# print((p1.T*p21).T)
# # # u = np.array([[0.18, 0.009, 0.001],[0.009, 0.78, 0.001],[0.0075, 0.0075, 0.005]])
# # # u = np.array([[0.15, 0.03, 0.005],[0.03, 0.75, 0.005],[0.003, 0.003, 0.024]])
# u = np.array([[0.0205, 0, 0.0028],[0, 0.7074, 0.0045],[0.0028, 0.0045, 0.2573]])
# # # print((u.T / u.sum(axis=1)).T)


# hemc.get_param_EM(sample_visible,100)
# seg_hidden = hemc.seg_mpm(sample_visible)
# err = np.sum(seg_hidden != sample_hidden)/sample_length
# terr['hemc_estim_param_EM'] = (err <= 0.5) * err + (err > 0.5) * (1 - err)
# hemc.init_data_prior(sample_visible)
# hemc.get_param_ICE(sample_visible,100,10)
# seg_hidden = hemc.seg_mpm(sample_visible)
# err = np.sum(seg_hidden != sample_hidden)/sample_length
# terr['hemc_estim_param_ICE'] = (err <= 0.5) * err + (err > 0.5) * (1 - err)
# hemc.init_data_prior(sample_visible)
# hemc.get_param_SEM(sample_visible,100)
# seg_hidden = hemc.seg_mpm(sample_visible)
# err = np.sum(seg_hidden != sample_hidden)/sample_length
# terr['hemc_estim_param_SEM'] = (err <= 0.5) * err + (err > 0.5) * (1 - err)
#
# u1 = u
# u2 = np.array([[[0.1, 0.2, 0.4, 0.2, 0.1], [0.1, 0.2, 0.4, 0.2, 0.1], [0.4, 0.2, 0.2, 0.1, 0.1]],
#                [[0.1, 0.2, 0.4, 0.2, 0.1], [0.1, 0.2, 0.4, 0.2, 0.1], [0.4, 0.2, 0.2, 0.1, 0.1]],
#                [[0.4, 0.2, 0.2, 0.1, 0.1], [0.4, 0.2, 0.2, 0.1, 0.1], [0.4, 0.2, 0.2, 0.1, 0.1]]])
# hesmc = HESMC_ctod(nbc_x=2, nbc_u=5)
# hesmc.give_param(u1, u2, mu, sig)
# sample_hidden, sample_visible = hesmc.generate_sample(sample_length, x_only=True)
# sample_visible = sample_visible.reshape(-1, 1)
# seg_hidden = hesmc.seg_mpm(sample_visible)
# err = np.sum(seg_hidden != sample_hidden)/sample_length
# terr['hesmc'] = (err <= 0.5) * err + (err > 0.5) * (1 - err)
# hesmc.init_data_prior(sample_visible)
# hesmc.get_param_EM(sample_visible,100)
# seg_hidden = hesmc.seg_mpm(sample_visible)
# err = np.sum(seg_hidden != sample_hidden)/sample_length
# terr['hesmc_estim_param_EM'] = (err <= 0.5) * err + (err > 0.5) * (1 - err)
# hesmc.init_data_prior(sample_visible)
# hesmc.get_param_ICE(sample_visible,100,10)
# seg_hidden = hesmc.seg_mpm(sample_visible)
# err = np.sum(seg_hidden != sample_hidden)/sample_length
# terr['hesmc_estim_param_ICE'] = (err <= 0.5) * err + (err > 0.5) * (1 - err)
# hesmc.init_data_prior(sample_visible)
# hesmc.get_param_SEM(sample_visible,100)
# seg_hidden = hesmc.seg_mpm(sample_visible)
# err = np.sum(seg_hidden != sample_hidden)/sample_length
# terr['hesmc_estim_param_SEM'] = (err <= 0.5) * err + (err > 0.5) * (1 - err)

# test = np.array(
#     [0] * 20 + [0, 1] * 40 + [0] * 20 + [1] * 20 + [0] * 20 + [1] * 20 + [0] * 20 + [1] * 20 + [0] * 20 + [1] * 20 + [
#         0] * 20 + [0, 1] * 40 + [0] * 20 + [1] * 20 + [0] * 20 + [1] * 20 + [0] * 20 + [1] * 20 + [0] * 20 + [
#         1] * 20 + [0] * 20 + [0, 1] * 40 + [0] * 20 + [1] * 20 + [0] * 20 + [1] * 20 + [0] * 20 + [1] * 20 + [
#         0] * 20 + [1] * 20 + [0] * 20 + [0, 1] * 40 + [0] * 20 + [1] * 20 + [0] * 20 + [1] * 20 + [0] * 20 + [
#         1] * 20 + [0] * 20 + [1] * 20)
# plt.plot(test)
# plt.show()
# mu = np.array([[0], [1]])
# sig = np.array([[[1]], [[1]]])
# test_noisy = (test == 0) * np.random.normal(mu[0, 0], sig[0, 0], test.shape) + (
#         test == 1) * np.random.normal(mu[1, 0], sig[1, 0], test.shape)
# plt.plot(test_noisy)
# plt.show()
# sample_visible = test_noisy.reshape(-1, 1)
# hmc = HMC_ctod(2)
# hmc.init_data_prior(sample_visible)
# hmc.get_param_EM(sample_visible, 100, early_stopping=10 ** -10)
# seg_hidden_hmc = hmc.seg_mpm(sample_visible)
# errhmc = np.sum(seg_hidden_hmc != test) / test.shape[0]
# #
# hsmc = HSMC_ctod(2, 5)
# hsmc.init_data_prior(sample_visible)
# hsmc.get_param_EM(sample_visible, 100, early_stopping=10 ** -10)
# seg_hidden_hsmc = hsmc.seg_mpm(sample_visible)
# errhsmc = np.sum(seg_hidden_hsmc != test) / test.shape[0]
# #
# hemc = HEMC_ctod(nbc_x=2)
# hemc.init_data_prior(sample_visible)
# hemc.get_param_supervised(sample_visible, test, 1000)
# seg_hidden_hemc = hemc.seg_mpm(sample_visible)
# errhemc = np.sum(seg_hidden_hemc != test) / test.shape[0]
# #
# hesmc = HESMC_ctod(nbc_x=2, nbc_u=5)
# hesmc.init_data_prior(sample_visible)
# hesmc.get_param_EM(sample_visible, 100, early_stopping=10 ** -10)
# seg_hidden_hesmc = hesmc.seg_mpm(sample_visible)
# errhesmc = np.sum(seg_hidden_hesmc != test) / test.shape[0]
# #
# plt.plot(seg_hidden_hmc)
# plt.show()
# plt.plot(seg_hidden_hsmc)
# plt.show()
# plt.plot(seg_hidden_hemc)
# plt.show()
# # plt.plot(seg_hidden_hesmc)
# # plt.show()
# # print(errhmc)
# # print(errhsmc)
# print(errhemc)
# # print(errhesmc)
# #

# # u1 = np.array([[0.19, 0.01, 0], [0.01, 0.79, 0], [0, 0, 0]])
# # u1 = np.array([[0.18, 0.009, 0.001], [0.009, 0.78, 0.001], [0.0075, 0.0075, 0.005]])
# # u1 = np.array([[0.15, 0.03, 0.005],[0.03, 0.75, 0.005],[0.003, 0.003, 0.024]])
# # u1 = np.array([[0.15, 0.03, 0.005],[0.03, 0.75, 0.005],[0.003, 0.003, 0.024]])
# u1 = np.array([[0.0200, 0.0005, 0.0028],[0.0074, 0.7000, 0.0045],[0.0028, 0.0045, 0.2573]])
# u2 = np.array([[[0.1, 0.2, 0.4, 0.2, 0.1], [0.1, 0.2, 0.4, 0.2, 0.1], [0.4, 0.2, 0.2, 0.1, 0.1]],
#                [[0.1, 0.2, 0.4, 0.2, 0.1], [0.1, 0.2, 0.4, 0.2, 0.1], [0.4, 0.2, 0.2, 0.1, 0.1]],
#                [[0.4, 0.2, 0.2, 0.1, 0.1], [0.4, 0.2, 0.2, 0.1, 0.1], [0.4, 0.2, 0.2, 0.1, 0.1]]])
# mu = np.array([[0], [1]])
# sig = np.array([[[1]], [[1]]])
# hesmc = HESMC_ctod(nbc_x=2, nbc_u=5)
# hesmc.give_param(u1, u2, mu, sig)
# sample_hidden, sample_visible = hesmc.generate_sample(sample_length, x_only=False)
# sample_hidden = convert_multcls_vectors(sample_hidden, (hesmc.nbc_u1*hesmc.nbc_u2, hesmc.nbc_x))[:, 1]
# sample_visible = sample_visible.reshape(-1, 1)
# seg_hidden = hesmc.seg_mpm(sample_visible)
# err = np.sum(seg_hidden != sample_hidden)/sample_length
# terr['hesmc'] = (err <= 0.5) * err + (err > 0.5) * (1 - err)
# hesmc.init_data_prior(sample_visible)
# hesmc.get_param_EM(sample_visible,100)
# seg_hidden = hesmc.seg_mpm(sample_visible)
# err = np.sum(seg_hidden != sample_hidden)/sample_length
# terr['hesmc_estim_param_EM'] = (err <= 0.5) * err + (err > 0.5) * (1 - err)
# hesmc.init_data_prior(sample_visible)
# hesmc.get_param_ICE(sample_visible,100,10)
# seg_hidden = hesmc.seg_mpm(sample_visible)
# err = np.sum(seg_hidden != sample_hidden)/sample_length
# terr['hesmc_estim_param_ICE'] = (err <= 0.5) * err + (err > 0.5) * (1 - err)
# hesmc.init_data_prior(sample_visible)
# hesmc.get_param_SEM(sample_visible,100)
# seg_hidden = hesmc.seg_mpm(sample_visible)
# err = np.sum(seg_hidden != sample_hidden)/sample_length
# terr['hesmc_estim_param_SEM'] = (err <= 0.5) * err + (err > 0.5) * (1 - err)

# hmc = HMC_ctod()
# hmc.init_data_prior(sample_visible, 2)
# hmc.get_param_EM(sample_visible, 50)
# seg_hidden = hmc.seg_mpm(sample_visible)
# err = np.sum(seg_hidden != sample_hidden) / sample_length
# terr['hmc_estim_param_ICE'] = (err <= 0.5) * err + (err > 0.5) * (1 - err)

# c = np.array([[0.19, 0.01], [0.01, 0.79]])
# p = np.sum(c, axis=1)
# t = (c.T / p).T
# mu = np.array([[0], [1]])
# sig = np.array([[[1]], [[1]]])
# hmt = HMT_ctod(2, p=p, t=t, mu=mu, sigma=sig)
#
# hidden, visible = hmt.generate_sample(sample_length, 10)
#
#
# visible = [v.reshape(-1, 1) for v in visible]
# # hmt.init_data_prior(visible,2)
# hmt.get_param_EM(visible, 30)
# seg_hidden = hmt.seg_mpm(visible)
# terr_aux = sum([np.sum(seg_hidden[k]!=hidden[k]) for k in range(len(hidden))])/sum([hidden[k].shape[0] for k in range(len(hidden))])
# terr = np.sum(seg_hidden[0]!=hidden[0])/hidden[0].shape[0]
# print(terr_aux, terr)
# d = 4
# c = np.array([[0.49, 0.01], [0.01, 0.49]])
# u = np.full((d, d), 1 / d)
# mu = np.array([[[n1, n2] for n1 in range(2 * d)] for n2 in range(2 * d)])
# sig = np.array([[[[1, 0.1], [0.1, 1]] for n1 in range(2 * d)] for n2 in range(2 * d)])
# tmc = TMC_ctod(nbc_x=2, nbc_u=d)
# tmc.give_param(c, u, mu, sig)
# print(tmc.p, tmc.t)


# c = np.array([[0, 0, 0, 0], [0, 0.40, 0.04, 0.01],[0, 0.04, 0.40, 0.01],[0, 0.05, 0.04, 0.01]])
# mu = np.array([[0], [1]])
# sig = np.array([[[1]], [[1]]])
# hemc = HEMC_ctod(2)
# hemc.give_param(c,mu,sig)
# sample_hidden, sample_visible = hemc.generate_sample(sample_length)
# sample_visible = sample_visible.reshape(-1, 1)
# sample_hidden = process_sampleDS(sample_hidden, 2)
# seg_hidden = hemc.seg_mpm(sample_visible)
# err = np.sum(seg_hidden != sample_hidden)/sample_length
# print('sup',err)
# hemc.init_data_prior(sample_visible)
# hemc.get_param_EM(sample_visible,30)
# seg_hidden = hemc.seg_mpm(sample_visible)
# err = np.sum(seg_hidden != sample_hidden)/sample_length
# print('non-sup',err)

# err = np.sum(seg_hidden != sample_hidden) / sample_length
# terr['hmc'] = (err <= 0.5) * err + (err > 0.5) * (1 - err)
# hmc.init_data_prior(sample_visible, 2)
# hmc.get_param_ICE(sample_visible, 30, 10, True)
# seg_hidden = hmc.seg_mpm(sample_visible)
# err = np.sum(seg_hidden != sample_hidden) / sample_length
# terr['hmc_estim_param_ICE'] = (err <= 0.5) * err + (err > 0.5) * (1 - err)
# tmc = TMC_ctod(2, 4)
# tmc.init_data_prior(sample_visible)
# tmc.get_param_ICE(sample_visible,30,10)
# seg_hidden = tmc.seg_mpm(sample_visible, True)
# err = [np.sum(seg_hidden[:,i]!=sample_hidden)/sample_length for i in range(seg_hidden.shape[1])]
# err = [(e <= 0.5)*e + (e > 0.5)*(1-e) for e in err]
# min_err = min(err)
# print(err)
# print(min_err)
# terr['tmc_estim_param_ICE'] = (err <= 0.5)*err + (err > 0.5)*(1-err)
# hmc.init_data_prior(sample_visible, 2)
# hmc.get_param_EM(sample_visible,30)
# seg_hidden = hmc.seg_mpm(sample_visible)
# err = np.sum(seg_hidden!=sample_hidden)/sample_length
# terr['hmc_estim_param_EM'] = (err <= 0.5)*err + (err > 0.5)*(1-err)
# hmc.init_data_prior(sample_visible, 2)
# hmc.get_param_SEM(sample_visible,30,10)
# seg_hidden = hmc.seg_mpm(sample_visible)
# err = np.sum(seg_hidden!=sample_hidden)/sample_length
# terr['hmc_estim_param_SEM'] = (err <= 0.5)*err + (err > 0.5)*(1-err)

# c = np.array([[0.49, 0.01], [0.01, 0.49]])
# u = np.array([[0.2, 0.2, 0.2, 0.2, 0.2], [0.2, 0.2, 0.2,0.2, 0.2]])
# mu = np.array([[0], [1]])
# sig = np.array([[[1]], [[1]]])
# hsmc = HSMC_ctod(nbc_x=2, nbc_u=5)
# hsmc.give_param(c,u,mu,sig)
# print(hsmc.p, hsmc.t)
# sample_hidden, sample_visible = hsmc.generate_sample(sample_length)
# sample_hidden = convert_multcls_vectors(sample_hidden, (hsmc.nbc_u, hsmc.nbc_x))[:, 1]
# sample_visible = sample_visible.reshape(-1, 1)
# seg_hidden = hsmc.seg_mpm(sample_visible)
# err = np.sum(seg_hidden != sample_hidden) / sample_length
# terr['hsmc'] = (err <= 0.5) * err + (err > 0.5) * (1 - err)
# hsmc.init_data_prior(sample_visible, 2)
# hsmc.get_param_ICE(sample_visible, 30, 10, True)
# seg_hidden = hsmc.seg_mpm(sample_visible)
# err = np.sum(seg_hidden != sample_hidden) / sample_length
# terr['hsmc_estim_param_ICE'] = (err <= 0.5) * err + (err > 0.5) * (1 - err)

# c=np.array([[0.1,0.4],[0.4,0.1]])
# p = np.sum(c,axis=1)
# t = (c.T/p).T
# mu=np.array([[[0,0],[2,2]],[[4,4], [6,6]]])
# sig=np.array([[[[1,0.1],[0.1,1]],[[1,0.1],[0.1,1]]],[[[1,0.1],[0.1,1]],[[1,0.1],[0.1,1]]]])
# pmc = PMC_ctod(p=p, t=t, mu=mu, sigma=sig)
# sample_hidden, sample_visible = pmc.generate_sample(sample_length,1)
# sample_visible = sample_visible.reshape(-1,1)
# seg_hidden = pmc.seg_mpm(sample_visible)
# err = np.sum(seg_hidden!=sample_hidden)/sample_length
# terr['pmc'] = (err <= 0.5)*err + (err > 0.5)*(1-err)
# pmc.init_data_prior(sample_visible, 2)
# pmc.get_param_ICE(sample_visible,30,10, True)
# seg_hidden = pmc.seg_mpm(sample_visible)
# err = np.sum(seg_hidden!=sample_hidden)/sample_length
# terr['pmc_estim_param_ICE'] = (err <= 0.5)*err + (err > 0.5)*(1-err)
# pmc.init_data_prior(sample_visible, 2)
# pmc.get_param_SEM(sample_visible,30)
# seg_hidden = pmc.seg_mpm(sample_visible)
# err = np.sum(seg_hidden!=sample_hidden)/sample_length
# terr['pmc_estim_param_SEM'] = (err <= 0.5)*err + (err > 0.5)*(1-err)

# c = np.array([[0.49, 0.01], [0.01, 0.49]])
# u = np.array([[0.05, 0.05, 0.05, 0.05, 0.05], [0.05, 0.05, 0.05, 0.05, 0.05], [0.05, 0.05, 0.05, 0.05, 0.05], [0.05, 0.05, 0.05, 0.05, 0.05], [0.05, 0.05, 0.05, 0.05, 0.05]])
# mu=np.array([[[0,0],[2,2]],[[4,4], [6,6]]])
# sig=np.array([[[[1,0.1],[0.1,1]],[[1,0.1],[0.1,1]]],[[[1,0.1],[0.1,1]],[[1,0.1],[0.1,1]]]])
# psmc = PSMC_ctod(nbc_x=2, nbc_u=5)
# psmc.give_param(c,u,mu,sig)
# sample_hidden, sample_visible = psmc.generate_sample(sample_length,1)
# sample_visible = sample_visible.reshape(-1,1)
# seg_hidden = psmc.seg_mpm(sample_visible)
# err = np.sum(seg_hidden!=sample_hidden)/sample_length
# terr['psmc'] = (err <= 0.5)*err + (err > 0.5)*(1-err)
# psmc.init_data_prior(sample_visible, 2)
# psmc.get_param_ICE(sample_visible,30,10, True)
# seg_hidden = psmc.seg_mpm(sample_visible)
# err = np.sum(seg_hidden!=sample_hidden)/sample_length
# terr['psmc_estim_param_ICE'] = (err <= 0.5)*err + (err > 0.5)*(1-err)
#
# c = np.array([[0.43, 0.01, 0.01, 0.01], [0.01, 0.01, 0.01, 0.01], [0.01, 0.01, 0.01, 0.01], [0.01, 0.01, 0.01, 0.43]])
# p = np.sum(c,axis=1)
# t = (c.T/p).T
# mu = np.array([[[0, 0], [0, 1], [1, 0], [1, 1]], [[2, 2], [2, 3], [3, 2], [3, 3]], [[4, 4], [4, 5], [5, 4], [5, 5]],
#                [[6, 6], [6, 7], [7, 6], [7, 7]]])
# sig = np.array([[[[1, 0.1], [0.1, 1]], [[1, 0.1], [0.1, 1]], [[1, 0.1], [0.1, 1]], [[1, 0.1], [0.1, 1]]],
#                 [[[1, 0.1], [0.1, 1]], [[1, 0.1], [0.1, 1]], [[1, 0.1], [0.1, 1]], [[1, 0.1], [0.1, 1]]],
#                 [[[1, 0.1], [0.1, 1]], [[1, 0.1], [0.1, 1]], [[1, 0.1], [0.1, 1]], [[1, 0.1], [0.1, 1]]],
#                 [[[1, 0.1], [0.1, 1]], [[1, 0.1], [0.1, 1]], [[1, 0.1], [0.1, 1]], [[1, 0.1], [0.1, 1]]]])
# tmc = TMC_ctod(2, 2, p=p, t=t, mu=mu, sigma=sig)
# sample_hidden, sample_visible = tmc.generate_sample(sample_length, 1)
# sample_hidden = convert_multcls_vectors(sample_hidden, (tmc.nbc_x, tmc.nbc_u))[:, 1]
# sample_visible = sample_visible.reshape(-1,1)
# seg_hidden = tmc.seg_mpm(sample_visible)
# err = np.sum(seg_hidden!=sample_hidden)/sample_length
# terr['tmc'] = (err <= 0.5)*err + (err > 0.5)*(1-err)
# tmc.init_data_prior(sample_visible)
# tmc.get_param_ICE(sample_visible,30,10)
# seg_hidden = tmc.seg_mpm(sample_visible)
# err = np.sum(seg_hidden!=sample_hidden)/sample_length
# terr['tmc_estim_param_ICE'] = (err <= 0.5)*err + (err > 0.5)*(1-err)
# tmc.init_data_prior(sample_visible)
# tmc.get_param_SEM(sample_visible,30)
# seg_hidden = tmc.seg_mpm(sample_visible)
# err = np.sum(seg_hidden!=sample_hidden)/sample_length
# terr['tmc_estim_param_SEM'] = (err <= 0.5)*err + (err > 0.5)*(1-err)

# c = np.array([[0.43, 0.01, 0.01, 0.01], [0.01, 0.01, 0.01, 0.01], [0.01, 0.01, 0.01, 0.01], [0.01, 0.01, 0.01, 0.43]])
# u = np.array([[0.05, 0.05, 0.05, 0.05, 0.05], [0.05, 0.05, 0.05, 0.05, 0.05], [0.05, 0.05, 0.05, 0.05, 0.05], [0.05, 0.05, 0.05, 0.05, 0.05], [0.05, 0.05, 0.05, 0.05, 0.05]])
# mu = np.array([[[0, 0], [0, 1], [1, 0], [1, 1]], [[2, 2], [2, 3], [3, 2], [3, 3]], [[4, 4], [4, 5], [5, 4], [5, 5]],
#                [[6, 6], [6, 7], [7, 6], [7, 7]]])
# sig = np.array([[[[1, 0.1], [0.1, 1]], [[1, 0.1], [0.1, 1]], [[1, 0.1], [0.1, 1]], [[1, 0.1], [0.1, 1]]],
#                 [[[1, 0.1], [0.1, 1]], [[1, 0.1], [0.1, 1]], [[1, 0.1], [0.1, 1]], [[1, 0.1], [0.1, 1]]],
#                 [[[1, 0.1], [0.1, 1]], [[1, 0.1], [0.1, 1]], [[1, 0.1], [0.1, 1]], [[1, 0.1], [0.1, 1]]],
#                 [[[1, 0.1], [0.1, 1]], [[1, 0.1], [0.1, 1]], [[1, 0.1], [0.1, 1]], [[1, 0.1], [0.1, 1]]]])
# gsmc = GSMC_ctod(nbc_x=2, nbc_u=5)
# gsmc.give_param(c,u,mu,sig)
# sample_hidden, sample_visible = gsmc.generate_sample(sample_length, 1)
# sample_visible = sample_visible.reshape(-1,1)
# seg_hidden = gsmc.seg_mpm(sample_visible)
# err = np.sum(seg_hidden!=sample_hidden)/sample_length
# terr['tmc'] = (err <= 0.5)*err + (err > 0.5)*(1-err)
# gsmc.init_data_prior(sample_visible)
# gsmc.get_param_ICE(sample_visible,30,10)
# seg_hidden = gsmc.seg_mpm(sample_visible)
# err = np.sum(seg_hidden!=sample_hidden)/sample_length
# terr['gsmc_estim_param_ICE'] = (err <= 0.5)*err + (err > 0.5)*(1-err)

# c = np.array([[0.43, 0.01, 0.01, 0.01], [0.01, 0.01, 0.01, 0.01], [0.01, 0.01, 0.01, 0.01], [0.01, 0.01, 0.01, 0.43]])
# mu=np.array([[0,0],[0,1], [1,0], [1,1]])
# sig=np.array([[[1,0.1],[0.1,1]],[[1,0.1],[0.1,1]],[[1,0.1],[0.1,1]],[[1,0.1],[0.1,1]]])
# mhmc = HMC_multiR_ctod(resoffset=(4,), resnbc=(2,2), c=c, mu=mu, sigma=sig)
# sample_hidden, sample_visible = mhmc.generate_sample(sample_length)
# sample_hidden = convert_multcls_vectors(sample_hidden, mhmc.resnbc)[:, 0]
# sample_visible = sample_visible.reshape(-1,2)
# seg_hidden = mhmc.seg_mpm(sample_visible, 0)
# err = np.sum(seg_hidden!=sample_hidden)/sample_length
# terr['mhmc'] = (err <= 0.5)*err + (err > 0.5)*(1-err)
# mhmc.init_data_prior(sample_visible)
# mhmc.get_param_ICE(sample_visible,30,10, True)
# seg_hidden = mhmc.seg_mpm(sample_visible, 0)
# err = np.sum(seg_hidden!=sample_hidden)/sample_length
# terr['mhmc_estim_param_ICE_2obs'] = (err <= 0.5)*err + (err > 0.5)*(1-err)
# mhmc.init_data_prior(sample_visible[:,0:1])
# mhmc.get_param_ICE(sample_visible[:,0:1],30,10, True)
# seg_hidden = mhmc.seg_mpm(sample_visible[:,0:1], 0)
# err = np.sum(seg_hidden!=sample_hidden)/sample_length
# terr['mhmc_estim_param_ICE_no2obs'] = (err <= 0.5)*err + (err > 0.5)*(1-err)


# print(terr)
