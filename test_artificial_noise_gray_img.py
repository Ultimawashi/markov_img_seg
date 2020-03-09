import numpy as np
import cv2 as cv
import os
import json

np.set_printoptions(threshold=np.inf)
from utils import get_peano_index, convert_multcls_vectors, moving_average, cut_diff, split_in, sigmoid_np, heaviside_np
from dbn import DBN, RBM_dtod, RBM_ctod
from hmm import HMC_ctod, HSMC_ctod, HESMC_ctod, HEMC_ctod, HSEMC_ctod, HEMC2_ctod
from pmm import PMC_ctod, PSMC_ctod
from tmm import TMC_ctod, GSMC_ctod
from sklearn.cluster import KMeans

resfolder = './img/res_test2'
# imgfs = ['alfa2', 'beee2', 'cible2', 'city2', 'country2', 'promenade2', 'veau2', 'zebre2', 'img409']
imgfs = ['beee2', 'cible2', 'alfa2']
resolutions = [(128, 128)]
max_val = 255
gauss_noise = [
               {'corr': False, 'mu1': 0, 'mu2': 1, 'sig1': 1, 'sig2': 1, 'corr_param': None},
               {'corr': False, 'mu1': 0, 'mu2': 0.6, 'sig1': 1, 'sig2': 1, 'corr_param': None},
               {'corr': False, 'mu1': 0, 'mu2': 0.4, 'sig1': 1, 'sig2': 1, 'corr_param': None}]

# models = [{'name': 'hmc', 'model': HMC_ctod(2), 'params': None},
#           {'name': 'hsmc', 'model': HSMC_ctod(2,10), 'params': None},
#           {'name': 'hemc', 'model': HEMC_ctod(2), 'params': None},
#           {'name': 'hsemc', 'model': HSEMC_ctod(2,10), 'params': None}]
models = [{'name': 'hmc', 'model': HMC_ctod(2), 'params': None},
          {'name': 'hsmc', 'model': HSMC_ctod(2,10), 'params': None}]
kmeans_clusters = 2

for resolution in resolutions:
    for imgf in imgfs:
        if not os.path.exists(resfolder + '/' + imgf):
            os.makedirs(resfolder + '/' + imgf)

        img = cv.imread('./img/' + imgf + '.bmp')  # Charger l'image
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # Si cette ligne est décommentée on travaille en niveau de gris
        img = cv.resize(img, resolution)
        img = heaviside_np(img)
        cv.imwrite(resfolder + '/' + imgf + '/' + str(resolution[0]) + '_' + str(resolution[1]) + '.bmp', img * max_val)
        test = get_peano_index(img.shape[0])  # Parcours de peano
        # test = [a.flatten() for a in np.indices(resolution)] #Parcours ligne par ligne
        hidden = img[test[0], test[1]]

        for noise in gauss_noise:
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
            cv.imwrite(
                resfolder + '/' + imgf + '/' + str(resolution[0]) + '_' + str(
                    resolution[1]) + '_' + corr + '_' + corr_param + noise_param + '.bmp', sigmoid_np(img_noisy) * max_val)

            data = img_noisy[test[0], test[1]].reshape(-1, 1)
            kmeans = KMeans(n_clusters=kmeans_clusters).fit(data)
            seg_kmeans = np.zeros(
                (img.shape[0], img.shape[1]))
            seg_kmeans[test[0], test[1]] = kmeans.labels_
            cv.imwrite(resfolder + '/' + imgf + '/' + str(resolution[0]) + '_' + str(
                resolution[1]) + '_' + corr + '_' + corr_param + noise_param + '_seg_kmeans' + '.bmp', seg_kmeans * int(max_val/(kmeans_clusters-1)))

            for model in models:
                if not model['params']:
                    # if not model['name']=='hmc':
                    #     model['model'].init_from_markov_chain(data)
                    # else:
                    #     model['model'].init_data_prior(data)
                    model['model'].init_data_prior(data)
                else:
                    model['model'].give_param(*model['params'])
                # if 'hmc' not in model['name']:
                #     model['model'].get_param_supervised(data,hidden, 100, early_stopping=10**-10)  # estimation des paramètres avec ICE, (on peut utiliser SEM ou EM avec get_param_EM ou get_param_SEM)
                # else:
                #     model['model'].get_param_supervised(data, hidden)
                model['model'].get_param_EM(data, 500, early_stopping=10 ** -10)
                seg = np.zeros(
                    (img.shape[0], img.shape[1]))  # Création d'une matrice vide qui va recevoir l'image segmentée
                seg[test[0], test[1]] = model['model'].seg_mpm(
                    data)  # Remplir notre matrice avec les valeurs de la segmentation
                cv.imwrite(
                    resfolder + '/' + imgf + '/' + str(resolution[0]) + '_' + str(
                        resolution[1]) + '_' + corr + '_' + corr_param + noise_param + '_seg_' + model['name'] + '.bmp',
                    seg * max_val)  # Sauvegarder l'image
                param_s = {'p': model['model'].p.tolist(), 't': model['model'].t.tolist(), 'mu': model['model'].mu.tolist(),
                           'sig': model['model'].sigma.tolist()}
                if hasattr(model['model'], 'nbc_u'):
                    seg_u = np.zeros(
                        (img.shape[0], img.shape[1]))  # Création d'une matrice vide qui va recevoir l'image segmentée
                    seg_u[test[0], test[1]] = model['model'].seg_mpm_u(
                        data)
                    cv.imwrite(
                        resfolder + '/' + imgf + '/' + str(resolution[0]) + '_' + str(
                            resolution[1]) + '_' + corr + '_' + corr_param + noise_param + '_smg_u_' + model[
                            'name'] + '.bmp',
                        seg_u * int(max_val/(model['model'].nbc_u-1)))  # Sauvegarder l'image
                with open(os.path.join(resfolder + '/' + imgf, str(resolution[0]) + '_' + str(
                            resolution[1]) + '_' + corr + '_' + corr_param + noise_param + '_param_' + model['name'] + '.txt'),
                              'w') as f:
                        json.dump(param_s, f, ensure_ascii=False)
