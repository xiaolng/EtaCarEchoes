import numpy as np
import matplotlib.pyplot as plt
import pickle

modelname = 'gpmodels_for_publication/lrn_gmodel.pickle'
with open(modelname,'rb') as f:
    model = pickle.load(f)

x = np.atleast_2d(np.linspace(-15,200,100)).T

y_pred, sigma = model.predict(x, return_std=True)

plt.plot(x, y_pred, c='red', label='Prediction')
plt.fill(np.concatenate([x, x[::-1]]),np.concatenate([y_pred - 1.9600 * sigma,(y_pred + 1.9600 * sigma)[::-1]]),alpha=.5, fc='red', ec='None', label='95% confidence interval')

plt.xlim(-20,200)
plt.gca().invert_yaxis()

plt.xlabel('Days since peak',size=14)
plt.ylabel('Magnitudes below peak',size=14)

plt.savefig(modelname.replace('.pickle','.pdf'), bbox_inches='tight')
