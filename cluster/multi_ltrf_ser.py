import numpy as np
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from pystruct.models import LatentTRF
#from pystruct.learners import LatentSSVM
from pystruct.learners import LatentSubgradientSSVM
import scipy.io

tot_classes = 15
n_machines = tot_classes * (tot_classes - 1) / 2 # c^n_2

all_clf = []
all_crf = []
all_X_train = []
all_X_test = []
all_Y_train = []
all_Y_test = []

all_energies_tr = []
all_y_pred_tr = []
all_h_pred_tr = []
all_loss_tr = []

all_energies_te = []
all_y_pred_te = []
all_h_pred_te = []
all_loss_te = []

#import ipdb
#ipdb.set_trace()

for i in xrange(tot_classes):
    for j in xrange(i+1, tot_classes):
        print '------------ training machine (%d,%d) -------------' % (i,j)

        # data
        data1 = scipy.io.loadmat('./15sc_data/15sc_c' + '%02d' % (i+1) + '.mat')
        data2 = scipy.io.loadmat('./15sc_data/15sc_c' + '%02d' % (j+1) + '.mat')
        X1 = data1['llc_feat']
        X2 = data2['llc_feat']
        num1 = X1.size
        num2 = X2.size

        h,w,f = X1[0,0].shape
        X_pos = np.ndarray((num1, h, w, f))
        Y_pos = np.ndarray(num1)
        for i in xrange(num1):
            X_pos[i,:,:,:] = X1[0,i]
            Y_pos[i] = 0
        X_train_pos, X_test_pos, Y_train_pos, Y_test_pos = train_test_split(X_pos, Y_pos, train_size=100)

        h,w,f = X2[0,0].shape
        X_neg = np.ndarray((num2, h, w, f))
        Y_neg = np.ndarray(num2)
        for i in xrange(num2):
            X_neg[i,:,:,:] = X2[0,i]
            Y_neg[i] = 1
        X_train_neg, X_test_neg, Y_train_neg, Y_test_neg = train_test_split(X_neg, Y_neg, train_size=100)

        X_train = np.vstack((X_train_pos, X_train_neg))
        X_test = np.vstack((X_test_pos, X_test_neg))
        Y_train = np.hstack((Y_train_pos, Y_train_neg))
        Y_test = np.hstack((Y_test_pos, Y_test_neg))

        # crf
        crf = LatentTRF(n_states = 4, n_classes = 2, n_features = 200, inference_method='qpbo')

        # learner
        clf = LatentSubgradientSSVM(
            model=crf, max_iter=500, C=10., verbose=2,
            n_jobs=-1, learning_rate=0.1, show_loss_every=10)
        # training
        clf.fit(X_train, Y_train)

        # testing
        energies_tr = np.ndarray(X_train.shape[0])
        y_pred_tr = np.ndarray(X_train.shape[0])
        h_pred_tr = np.ndarray(X_train.shape[:3])
        loss_tr = 0

        for k in xrange(X_train.shape[0]):
            print "predict training image %d" % k
            y_pred_tr[k], energies_tr[k], h_temp = crf.inference(X_train[k], clf.w, return_energy=True)
            h_pred_tr[k] = np.reshape(h_temp, X_train.shape[1:3])
            if y_pred_tr[k] != Y_train[k]:
                loss_tr = loss_tr + 1

        print '*-* training loss for (%d,%d) is %d' % (i,j,loss_tr)

        energies_te = np.ndarray(X_test.shape[0])
        y_pred_te = np.ndarray(X_test.shape[0])
        h_pred_te = np.ndarray(X_test.shape[:3])
        loss_te = 0

        for k in xrange(X_test.shape[0]):
            print "predict testing image %d" % k
            y_pred_te[k], energies_te[k], h_temp = crf.inference(X_test[k], clf.w, return_energy=True)
            h_pred_te[k] = np.reshape(h_temp, X_test.shape[1:3])
            if y_pred_te[k] != Y_test[k]:
                loss_te = loss_te + 1

        print '*-* testing loss for (%d,%d) is %d' % (i,j,loss_te)

        # collect data
        all_crf.append(crf)
        all_clf.append(clf)
        all_X_train.append(X_train)
        all_X_test.append(X_test)
        all_Y_train.append(Y_train)
        all_Y_test.append(Y_test)

        all_energies_tr.append(energies_tr)
        all_y_pred_tr.append(y_pred_tr)
        all_h_pred_tr.append(h_pred_tr)
        all_loss_tr.append(loss_tr)

        all_energies_te.append(energies_te)
        all_y_pred_te.append(y_pred_te)
        all_h_pred_te.append(h_pred_te)
        all_loss_te.append(loss_te)

import pickle
pickle.dump(all_clf, open("15sc_all_clf_t4_qpbo.p", "wb"))
pickle.dump(all_crf, open("15sc_all_crf_t4_qpbo.p", "wb"))

pickle.dump(all_energies_tr, open("15sc_all_energy_tr_t4_qpbo.p", "wb"))
pickle.dump(all_y_pred_tr, open("15sc_all_ypred_tr_t4_qpbo.p", "wb"))
pickle.dump(all_h_pred_tr, open("15sc_all_hpred_tr_t4_qpbo.p", "wb"))
pickle.dump(all_loss_tr, open("15sc_all_loss_tr_t4_qpbo.p", "wb"))

pickle.dump(all_energies_te, open("15sc_all_energy_te_t4_qpbo.p", "wb"))
pickle.dump(all_y_pred_te, open("15sc_all_ypred_te_t4_qpbo.p", "wb"))
pickle.dump(all_h_pred_te, open("15sc_all_hpred_te_t4_qpbo.p", "wb"))
pickle.dump(all_loss_te, open("15sc_all_loss_te_t4_qpbo.p", "wb"))

pickle.dump(all_X_train, open("15sc_all_X_train.p", "wb"))
pickle.dump(all_X_test, open("15sc_all_X_test.p", "wb"))
pickle.dump(all_Y_train, open("15sc_all_Y_train.p", "wb"))
pickle.dump(all_Y_test, open("15sc_all_Y_test.p", "wb"))

#import ipdb
#ipdb.set_trace()

