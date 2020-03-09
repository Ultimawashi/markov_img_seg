from rbm import RBM_ctod
import numpy as np
import cv2 as cv
import os

resfolder = './img/gen_rbm_ctod'
original_im_size = (16,16)
file = "./img/mnist/train_set0_resized.csv"

data = np.genfromtxt(file, delimiter=',').astype("int")
data = (data <= 128) * 0 + (data > 128) * 255
max_val = np.max(data)
data = data / max_val
data_noisy = (data == 0) * np.random.normal(0, 1, data.shape) + (data == 1) * np.random.normal(1, 1, data.shape)
rbm = RBM_ctod()
rbm.init_rand_prior(data.shape[1], data.shape[1])
rbm.get_param_CD(data_noisy, 64, 0.001, 100)
res = rbm.gen_hidden(100, 10)

if not os.path.exists(resfolder):
    os.makedirs(resfolder)

res = res.reshape((res.shape[0],original_im_size[0],original_im_size[1]))

for i in range(res.shape[0]):
    cv.imwrite(resfolder + '/' + 'im_persist_' + str(i) + '.bmp', res[i,:,:]*max_val)
