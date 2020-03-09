import numpy as np
import matplotlib.pyplot as plt
import json
import os


resfolder = './img/res_test'
imgfs = ['cible2']
resolution = (128, 128)
deltamnoise =[2, 1, 0.6, 0.4]
min_max_time=[2, 5, 10]
gaussname='gauss'
models = ['hsmc', 'psmc']

for imgf in imgfs:
    for idx, deltam in enumerate(deltamnoise):
        for deltat in min_max_time:
            for model in models:

                with open(os.path.join(resfolder + '/' + imgf, str(resolution[0]) + '_' + str(
                        resolution[1]) + '_' + gaussname + '_' + str(
                    deltam) + '_param_' + model + '_' + str(deltat) + '_sup.txt'), 'r') as f:
                    param_sup = json.load(f)

                param_sup = {k: np.array(v) for k, v in param_sup.items()}
                c_sup = (param_sup['t'].T * param_sup['p']).T
                px_sup = np.sum(param_sup['p'].reshape((2, deltat)), axis=1)
                cx_sup = np.sum(c_sup.reshape((2, deltat, 2, deltat)), axis=(1, 3))
                tx_sup = (cx_sup.T / px_sup).T
                u_sup = np.sum(c_sup.reshape((2, deltat, 2, deltat)), axis=(0, 2, 3))
                with open(os.path.join(resfolder + '/' + imgf, str(resolution[0]) + '_' + str(
                        resolution[1]) + '_' + gaussname + '_' + str(
                    deltam) + '_param_' + model + '_u' + str(deltat) + '_sup.txt'), 'w') as f:
                    json.dump(u_sup.tolist(), f, ensure_ascii=False)


                with open(os.path.join(resfolder + '/' + imgf, str(resolution[0]) + '_' + str(
                        resolution[1]) + '_' + gaussname + '_' + str(
                    deltam) + '_param_' + model + '_' + str(deltat) + '.txt'), 'r') as f:
                    param = json.load(f)
                param = {k: np.array(v) for k, v in param.items()}

                c = (param['p'] * param['t'].T).T
                px = np.sum(param['p'].reshape((2, deltat)), axis=1)
                cx = np.sum(c.reshape((2, deltat, 2, deltat)), axis=(1, 3))
                tx = (cx.T / px).T
                u = np.sum(c.reshape((2, deltat, 2, deltat)), axis=(0, 2, 3))
                with open(os.path.join(resfolder + '/' + imgf, str(resolution[0]) + '_' + str(
                        resolution[1]) + '_' + gaussname + '_' + str(
                    deltam) + '_param_' + model + '_u' + str(deltat) + '.txt'), 'w') as f:
                    json.dump(u.tolist(), f, ensure_ascii=False)

