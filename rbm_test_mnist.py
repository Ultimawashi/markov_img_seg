from rbm import RBM_dtod
import numpy as np
import cv2 as cv
import os

resfolder = './img/gen_rbm'
original_im_size = (28,28)
file = "D:/CIFRE_code/pfe_champs_rbm/train_set0.csv"

data = np.genfromtxt(file, delimiter=',').astype("int")
data = (data <= 128) * 0 + (data > 128) * 255
max_val = np.max(data)
data = data / max_val
rbm = RBM_dtod()
rbm.init_rand_prior(data.shape[1], 64)
rbm.get_param_CD(data, 64, 0.1, 100)
res = rbm.gen_visible(100, 10)

if not os.path.exists(resfolder):
    os.makedirs(resfolder)

res = res.reshape((res.shape[0],original_im_size[0],original_im_size[1]))

for i in range(res.shape[0]):
    cv.imwrite(resfolder + '/' + 'im_persist_' + str(i) + '.bmp', res[i,:,:]*max_val)
