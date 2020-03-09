import json
import numpy as np
import cv2 as cv

path1 = 'D:/CIFRE_code/imseg/img/res_test/cible2/u.txt'

with open(path1, 'r') as f:
    test = json.load(f)

test = np.array(test)

u = np.sum(test, axis=1)

print(u)

resfolder = './img/res_test'
min_max_time = [2, 5, 10]
imgf = 'cible2'
resolution = (128, 128)

ref_im = cv.cvtColor(
            cv.imread(resfolder + '/' + imgf + '/' + str(resolution[0]) + '_' + str(resolution[1]) + '.bmp'),
            cv.COLOR_BGR2GRAY)




