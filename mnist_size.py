import numpy as np
from skimage.transform import resize
from skimage import img_as_bool
import cv2 as cv
import os


resfolder = './img/mnist'
original_im_size = (28,28)
new_res = (16,16)
file = "D:/CIFRE_code/pfe_champs_rbm/train_set0.csv"

if not os.path.exists(resfolder):
    os.makedirs(resfolder)

data = np.genfromtxt(file, delimiter=',').astype("int")
data = data.reshape((data.shape[0],original_im_size[0],original_im_size[1]))
data = (data <= 128) * 0 + (data > 128) * 255
max_val = np.max(data)
data = data / max_val
data_resized = np.zeros((data.shape[0],new_res[0],new_res[1]))

for i in range(data.shape[0]):
    data_resized[i,:,:] = img_as_bool(resize(data[i,:,:], new_res, anti_aliasing=True))

data_resized = data_resized.reshape((data_resized.shape[0], data_resized.shape[1]*data_resized.shape[2])) * max_val

np.savetxt(resfolder + '/train_set0_resized.csv', data_resized, delimiter=",")