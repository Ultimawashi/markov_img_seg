import numpy as np
import cv2 as cv
import json
import os
from utils import calc_err


resfolder = './img/res_test2'
# imgfs = ['alfa2', 'beee2', 'cible2', 'city2', 'country2', 'promenade2', 'veau2', 'zebre2', 'img409']
imgfs = ['img409']
resolutions = [(128, 128)]

for resolution in resolutions:
    for imgf in imgfs:
        if os.path.exists(resfolder + '/' + imgf):
            terr = {}
            ref_im = cv.cvtColor(
                cv.imread(resfolder + '/' + imgf + '/' + str(resolution[0]) + '_' + str(resolution[1]) + '.bmp'),
                cv.COLOR_BGR2GRAY)
            content = [f for f in os.listdir(resfolder + '/' + imgf) if ('seg' in f) and (str(resolution[0]) + '_' + str(resolution[1]) in f)]
            for c in content:
                if os.path.isfile(os.path.join(resfolder + '/' + imgf, c)):
                    seg_im = cv.cvtColor(cv.imread(os.path.join(resfolder + '/' + imgf, c)), cv.COLOR_BGR2GRAY)
                    terr[c.replace('.bmp', '')] = calc_err(ref_im, seg_im)
                elif os.path.isdir(os.path.join(resfolder + '/' + imgf, c)):
                    list_im = [(f, calc_err(ref_im, cv.cvtColor(cv.imread(os.path.join(os.path.join(resfolder + '/' + imgf, c), f)),
                                                                cv.COLOR_BGR2GRAY))) for f in
                               os.listdir(os.path.join(resfolder + '/' + imgf, c)) if
                               os.path.isfile(os.path.join(os.path.join(resfolder + '/' + imgf, c), f))]
                    res = min(list_im, key=lambda t: t[1])
                    terr[c] = res[1]
                    os.rename(os.path.join(os.path.join(resfolder + '/' + imgf, c), res[0]),
                              os.path.join(resfolder + '/' + imgf, c + '.bmp'))
            with open(os.path.join(resfolder + '/' + imgf, 'terr.txt'), 'w') as f:
                json.dump(terr, f, ensure_ascii=False)
