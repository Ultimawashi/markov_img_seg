import numpy as np
import cv2 as cv
import os
import json

np.set_printoptions(threshold=np.inf)
from utils import get_peano_index, convert_multcls_vectors, moving_average, cut_diff, split_in, standardize_np
from dbn import DBN, RBM_dtod, RBM_ctod
from hmm import HMC_ctod, HSMC_ctod, HEMC_ctod, HESMC_ctod
from pmm import PMC_ctod, PSMC_ctod
from tmm import TMC_ctod, GSMC_ctod
from sklearn.cluster import KMeans

resfolder = './img/res_test2/test_real_gray_images'
imgfs = ['img415']
max_val = 255
resolution = (256, 256)
models = [{'name':'hmc', 'model':HMC_ctod(2)},
          {'name':'hemc', 'model':HEMC_ctod(2)}]


if not os.path.exists(resfolder):
    os.makedirs(resfolder)

for imgf in imgfs:

    img = cv.imread('./img/img_reelles/' + imgf + '.bmp') # Charger l'image
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # Si cette ligne est décommentée on travaille en niveau de gris
    img = cv.resize(img, resolution)
    img = standardize_np(img)

    test = get_peano_index(img.shape[0])  # Parcours de peano
    # test = [a.flatten() for a in np.indices(img.shape)] #Parcours ligne par ligne

    data = img[test[0], test[1]].reshape(-1, 1)

    for model in models:

        model['model'].init_data_prior(data)
        model['model'].get_param_EM(data,
                         50)  # estimation des paramètres avec ICE, (on peut utiliser SEM ou EM avec get_param_EM ou get_param_SEM)
        seg = np.zeros(
            (img.shape[0], img.shape[1]))  # Création d'une matrice vide qui va recevoir l'image segmentée
        seg[test[0], test[1]] = model['model'].seg_mpm(data)  # Remplir notre matrice avec les valeurs de la segmentation
        cv.imwrite(
            resfolder + '/' + imgf + '_seg_' + model['name'] + '.bmp',
            seg * max_val)  # Sauvegarder l'image

