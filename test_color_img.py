import numpy as np
import cv2 as cv
np.set_printoptions(threshold=np.inf)
from utils import get_peano_index, convert_multcls_vectors
from dbn import DBN, RBM_dtod, RBM_ctod
from hmm import HMC_ctod, HMC_multiR_ctod
from pmm import PMC_ctod
from tmm import TMC_ctod
from sklearn.cluster import KMeans


img = cv.imread('./img/testroute1.jpg') #Charger l'image
#img = cv.cvtColor(img, cv.COLOR_BGR2GRAY) #Si cette ligne est décommentée on travaille en niveau de gris

dimY = 3 #Dimension des observations (couleur = 3, niveau de gris = 1)
shape_seg_vect = (2,)*3 #Cela va servir a associer les classes avec des couleurs différentes si on a plus de deux classes (cela fonctionne jusqu'a 8 classes)

#Prétraiement de l'image (on ne travaille qu'avec des dimensions carrée en puissance de 2 pour l'instant (ex:(64,64), (256,256))
img_noisy = cv.resize(img,(256,256))
max_val = np.max(img_noisy)
img_noisy = img_noisy / max_val
im_res2 = cv.resize(img_noisy,(64,64))

test = get_peano_index(img_noisy.shape[0]) #Parcours de peano
#test = [a.flatten() for a in np.indices((256, 256))] #Parcours ligne par ligne
data = img_noisy[test[0], test[1]].reshape(-1,dimY)

#On segmente par kmeans pour initialiser les modèles suivants (on garde la segmentation kmeans pour la comparer avec les autres)
kmeans = KMeans(n_clusters=4, random_state=0).fit(data)
seg_kmeans = np.zeros((img_noisy.shape[0], img_noisy.shape[1], 3))
seg_kmeans[test[0], test[1]] = convert_multcls_vectors(kmeans.labels_, shape_seg_vect)
cv.imwrite('./img/res/testroute1_seg_kmeans_color.jpg', seg_kmeans*max_val)

# #Creation de la chaine de markov caché
# hmc = HMC_ctod()
# hmc.init_kmeans(data, kmeans.labels_) #initialisation grace au kmeans
# hmc.get_param_ICE(data, 10, 10) #estimation des paramètres avec ICE, (on peut utiliser SEM ou EM avec get_param_EM ou get_param_SEM)
# seg_hmc = np.zeros((img_noisy.shape[0], img_noisy.shape[1], 3)) #Création d'une matrice vide qui va recevoir l'image segmentée
# seg_hmc[test[0], test[1]] = convert_multcls_vectors(hmc.seg_mpm(data),shape_seg_vect) #Remplir notre matrice avec les valeurs de la segmentation
# cv.imwrite('./img/res/testroute1_seg_hmc_color.jpg', seg_hmc*max_val) #Sauvegarder l'image

#Creation de la chaine de markov caché multiresolution

mhmc = HMC_multiR_ctod(resoffset=(4,), resnbc=(2,2)) #resoffset correspond au décallage entre les deux résolution et resnbc correspond au nombre de classe par résoltuion
mhmc.init_rand_prior(dimY,(-1,1),(0,1),indep=True) #initialisation aléatoire (pas de kmeans pour l'instant)
mhmc.get_param_ICE(data, 10, 1) #estimation des paramètres avec ICE, (on peut utiliser SEM ou EM avec get_param_EM ou get_param_SEM)
seg_mhmc = [np.zeros((int(img_noisy.shape[0]/(2**n)), int(img_noisy.shape[1]/(2**n)), 3)) for n in range(len(mhmc.resnbc))] #Création des matrices vides pour chaque resolution qui vont recevoir l'image segmentée
peano_res=[get_peano_index(int(img_noisy.shape[0]/(2**n))) for n in range(len(mhmc.resnbc))]
seg_aux = mhmc.seg_mpm(data)
list_res = ((1,)+mhmc.resoffset)
for i,a in enumerate(seg_mhmc):
    print(a.shape)
    print(len(peano_res[i][0]),len(peano_res[i][1]))
    a[peano_res[i][0], peano_res[i][1]] = convert_multcls_vectors(convert_multcls_vectors(seg_aux, mhmc.resnbc)[:, i],shape_seg_vect)[::np.prod(list_res[:i+1])] #Remplir les matrices avec les valeurs de la segmentation pour chaque resolution
    cv.imwrite('./img/res/testroute1_seg_mhmc_color_res_'+str(i)+'.jpg', a*max_val) #Sauvegarder les images

#Creation de la chaine de markov couple
pmc = PMC_ctod()
# pmc.init_kmeans(data, kmeans.labels_) #initialisation grace au kmeans
pmc.init_rand_prior(4,dimY, (-1,1),(0,1))
pmc.get_param_ICE(data, 10, 1) #estimation des paramètres avec ICE, (on peut utiliser SEM avec get_param_SEM mais pas EM)
seg_pmc = np.zeros((img_noisy.shape[0], img_noisy.shape[1], 3))  #Création d'une matrice vide qui va recevoir l'image segmentée
seg_pmc[test[0], test[1]] = convert_multcls_vectors(pmc.seg_mpm(data),shape_seg_vect) #Remplir notre matrice avec les valeurs de la segmentation
cv.imwrite('./img/res/testroute1_seg_pmc_color.jpg', seg_pmc*max_val) #Sauvegarder l'image

#Creation de la chaine de markov triplet
tmc = TMC_ctod(nbc_x=4, nbc_u=2)
tmc.init_rand_prior(dimY, (-1,1),(0,1))  #initialisation aléatoire (pas de kmeans pour l'instant)
tmc.get_param_ICE(data, 10, 10) #estimation des paramètres avec ICE, (on peut utiliser SEM avec get_param_SEM mais pas EM)
seg_tmc = np.zeros((img_noisy.shape[0], img_noisy.shape[1], 3)) #Création d'une matrice vide qui va recevoir l'image segmentée
seg_tmc[test[0], test[1]] = convert_multcls_vectors(tmc.seg_mpm(data), shape_seg_vect) #Remplir notre matrice avec les valeurs de la segmentation
cv.imwrite('./img/res/testroute1_seg_tmc_color.jpg', seg_tmc*max_val) #Sauvegarder l'image

# #Afficher les images
# cv.imshow('img_noisy',img_noisy)
# cv.imshow('seg_kmeans',seg_kmeans)
# cv.imshow('seg_hmc',seg_hmc)
# for i,a in enumerate(seg_mhmc):
#     cv.imshow('seg_mhmc_'+str(i), a)
# cv.imshow('seg_pmc', seg_pmc)
# cv.imshow('seg_tmc', seg_tmc)
# cv.waitKey(0)


# Le reste de la section est dédié au rbm / dbn. Il est impossible d'appliquer le modèle directement car 1 trop de paramètre