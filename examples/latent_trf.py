import numpy as np
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split

from pystruct.models import LatentTRF
#from pystruct.learners import LatentSSVM
from pystruct.learners import LatentSubgradientSSVM

import scipy.io
data = scipy.io.loadmat('ltrf_feat.mat')
X1 = data['class1']
X2 = data['class2']
num1 = X1.size
num2 = X2.size
h,w,f = X1[0,0].shape

X = np.ndarray((num1+num2, h, w, f))
Y = np.ndarray(num1+num2)
cnt = 0
for i in xrange(num1+num2):
    if i < X1.size:
        X[i,:,:,:] = X1[0,i]
        Y[i] = 0
    else:
        X[i] = X2[0,i-X1.size]
        Y[i] = 1

crf = LatentTRF(n_states = 4, n_classes = 2, n_features = 200, inference_method='qpbo')

import pickle

try:
    clf = pickle.load(open("clf.p", "rb"))
except:
    clf = LatentSubgradientSSVM(
        model=crf, max_iter=500, C=10., verbose=2,
        n_jobs=-1, learning_rate=0.1, show_loss_every=10)
    clf.fit(X, Y)

import pickle
pickle.dump(clf, open("clf_c12_t4_qpbo.p", "wb"))

#import ipdb
#ipdb.set_trace()

energies = np.ndarray(num1+num2)
y_pred = np.ndarray(num1+num2)
h_pred = np.ndarray(X.shape[:3])
loss = 0

for i in xrange(num1+num2):
    print "predict image %d" % i
    y_pred[i], energies[i], h_temp = crf.inference(X[i], clf.w, return_energy=True)
    h_pred[i] = np.reshape(h_temp, X.shape[1:3])
    if y_pred[i] != Y[i]:
        loss = loss + 1
