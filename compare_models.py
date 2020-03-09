import numpy as np
import cv2 as cv
import os
import json
from hmm import HMC_ctod, HSMC_ctod, HEMC_ctod, HESMC_ctod, HSEMC_ctod
from pmm import PMC_ctod, PSMC_ctod
from tmm import TMC_ctod
from utils import standardize_np, get_peano_index, resize_gray_im_to_square, convert_multcls_vectors, moving_average, \
    heaviside_np, sigmoid_np

resfolder = './img/res_test2/models_comparison'
ref_img = './img/img_reelles/img420.bmp'
terr = {}
resolution = (64, 64)
sample_length = np.prod(resolution)
test = get_peano_index(resolution[0])  # Parcours de peano
# test = [a.flatten() for a in np.indices(resolution)] #Parcours ligne par ligne
max_val = 255
epsilon = 0.20
alpha = 0.04
c = np.array([[0.40, 0.05],
              [0.05, 0.50]])
u = np.array([[c[0, 0] - epsilon, c[0, 1] - alpha, (alpha / 2)],
              [c[1, 0] - alpha, c[1, 1] - epsilon, (epsilon / 2)],
              [(alpha / 2), (epsilon / 2), alpha + epsilon]])
u1 = np.array(
              [[[0.6,0.2,0.1,0.05,0.05], [0.6,0.2,0.1,0.05,0.05]],
               [[0.6,0.2,0.1,0.05,0.05], [0.6,0.2,0.1,0.05,0.05]]])
u2 = np.array(
              [[[0.6,0.2,0.1,0.05,0.05], [0.6,0.2,0.1,0.05,0.05], [0.6,0.2,0.1,0.05,0.05]],
               [[0.6,0.2,0.1,0.05,0.05], [0.6,0.2,0.1,0.05,0.05], [0.6,0.2,0.1,0.05,0.05]],
               [[0.6,0.2,0.1,0.05,0.05], [0.6,0.2,0.1,0.05,0.05], [0.6,0.2,0.1,0.05,0.05]]
               ])
perturbation_param = 0.7
models = [{'name': 'hmc', 'model': HMC_ctod(2),'params': [c]},
          {'name': 'hsmc', 'model': HSMC_ctod(2, 5), 'params': [c, u1]},
          {'name': 'hemc', 'model': HEMC_ctod(2),'params': [u]},
          {'name': 'hsemc', 'model': HSEMC_ctod(2, 5),'params': [u, u2]},
          {'name': 'hesmc', 'model': HESMC_ctod(2, 5),'params': [c, u1, perturbation_param]}
          ]

gauss_noise = [
               {'corr': False, 'mu1': 0, 'mu2': 3, 'sig1': 1, 'sig2': 1, 'corr_param': None},
               {'corr': False, 'mu1': 0, 'mu2': 2, 'sig1': 1, 'sig2': 1, 'corr_param': None},
               {'corr': False, 'mu1': 0, 'mu2': 1, 'sig1': 1, 'sig2': 1, 'corr_param': None}]

if not os.path.exists(resfolder):
    os.makedirs(resfolder)

img = cv.imread(ref_img)  # Charger l'image # Charger l'image
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # Si cette ligne est décommentée on travaille en niveau de gris
img = cv.resize(img, resolution)
img = heaviside_np(img)
test = get_peano_index(img.shape[0])  # Parcours de peano
# test = [a.flatten() for a in np.indices(img.shape)] #Parcours ligne par ligne


for noise in gauss_noise:
    for model in models:
        if not model['params']:
            if not noise['corr']:
                img_noisy = (img == 0) * np.random.normal(noise['mu1'], noise['sig1'], img.shape) + (
                        img == 1) * np.random.normal(noise['mu2'], noise['sig2'],
                                                     img.shape)
                corr = ''
                corr_param = ''
            else:
                img_noisy = moving_average((img == 0) * np.random.normal(noise['mu1'], noise['sig1'], img.shape),
                                           noise['corr_param'][0], noise['corr_param'][1]) + moving_average((
                                                                                                                    img == 1) * np.random.normal(
                    noise['mu2'], noise['sig2'],
                    img.shape), noise['corr_param'][0], noise['corr_param'][1])
                corr = 'corr'
                corr_param = str(noise['corr_param'][0]) + '_' + str(noise['corr_param'][1])
            noise_param = '(' + str(noise['mu1']) + ',' + str(noise['sig1']) + ')' + '_' + '(' + str(
                noise['mu2']) + ',' + str(noise['sig2']) + ')'
            data = img_noisy[test[0], test[1]].reshape(-1, 1)

            model['model'].init_data_prior(data)
            model['model'].get_param_EM(data,
                                        100)  # estimation des paramètres avec ICE, (on peut utiliser SEM ou EM avec get_param_EM ou get_param_SEM)
        else:
            corr = ''
            corr_param = ''
            noise_param = '(' + str(noise['mu1']) + ',' + str(noise['sig1']) + ')' + '_' + '(' + str(
                noise['mu2']) + ',' + str(noise['sig2']) + ')'
            params = model['params'] + [np.array([[noise['mu1']], [noise['mu2']]]),
                                        np.array([[[noise['sig1']]], [[noise['sig2']]]])]
            model['model'].give_param(*params)
        sample_hidden, sample_visible = model['model'].generate_sample(sample_length)
        img_save = np.zeros(resolution)
        img_save[test[0], test[1]] = sample_hidden
        cv.imwrite(resfolder + '/' + 'gen_' + model[
            'name'] + '_' + corr + '_' + corr_param + '_' + noise_param + '.bmp',
                   img_save * max_val)

        img_save = np.zeros(resolution)
        img_save[test[0], test[1]] = sample_visible.reshape((sample_visible.shape[0],))
        cv.imwrite(resfolder + '/' + 'gen_' + model[
            'name'] + '_' + corr + '_' + corr_param + '_' + noise_param + 'noisy.bmp',
                   sigmoid_np(img_save) * max_val)

        sample_visible = sample_visible.reshape(-1, 1)
        param_s = {'p': model['model'].p.tolist(), 't': model['model'].t.tolist(), 'mu': model['model'].mu.tolist(),
                   'sig': model['model'].sigma.tolist()}
        with open(os.path.join(resfolder, 'gen_' + model[
            'name'] + '_' + corr + '_' + corr_param + '_' + noise_param + '_param.txt'),
                  'w') as f:
            json.dump(param_s, f, ensure_ascii=False)
        for m in models:
            if m['name'] != model['name']:
                m['model'].init_data_prior(sample_visible)
                m['model'].get_param_EM(sample_visible, 10000, 10**-10)
            seg_hidden = m['model'].seg_mpm(sample_visible)
            img_save = np.zeros(resolution)
            img_save[test[0], test[1]] = seg_hidden
            cv.imwrite(resfolder + '/' + 'gen_' + model['name'] + '_' + corr + '_' + corr_param + '_' + noise_param + '_seg_' + m['name'] + '.bmp',
                       img_save * max_val)
            err = np.sum(seg_hidden != sample_hidden) / sample_length
            terr['gen_' + model['name'] + '_' + corr + '_' + corr_param + '_' + noise_param + '_seg_' + m['name']] = (
                                                                                                                             err <= 0.5) * err + (
                                                                                                                             err > 0.5) * (
                                                                                                                             1 - err)
            param_s = {'p': m['model'].p.tolist(), 't': m['model'].t.tolist(), 'mu': m['model'].mu.tolist(),
                       'sig': m['model'].sigma.tolist()}
            with open(os.path.join(resfolder, 'gen_' + model[
                'name'] + '_' + corr + '_' + corr_param + '_' + noise_param + '_seg_' + m['name'] + '_param.txt'),
                      'w') as f:
                json.dump(param_s, f, ensure_ascii=False)

with open(resfolder + '/terr.txt', 'w') as f:
    json.dump(terr, f, ensure_ascii=False)
