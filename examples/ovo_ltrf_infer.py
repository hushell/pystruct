import numpy as np
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from pystruct.models import LatentTRF
#from pystruct.learners import LatentSSVM
from pystruct.learners import LatentSubgradientSSVM
import scipy.io
import pickle
import progressbar

tot_classes = 15
n_machines = tot_classes * (tot_classes - 1) / 2 # c^n_2

## model
#all_clf = pickle.load(open("./store/15sc_all_clf_t4_qpbo.p", "rb"))
#all_crf = pickle.load(open("./store/15sc_all_crf_t4_qpbo.p", "rb"))

## train pred
#all_y_pred_tr = pickle.load(open("./store/15sc_all_ypred_tr_t4_qpbo.p", "rb"))
#all_h_pred_tr = pickle.load(open("./store/15sc_all_hpred_tr_t4_qpbo.p", "rb"))
#
## test pred
#all_y_pred_te = pickle.load(open("./store/15sc_all_ypred_te_t4_qpbo.p", "rb"))
#all_h_pred_te = pickle.load(open("./store/15sc_all_hpred_te_t4_qpbo.p", "rb"))
#
## GT
#all_Y_train = pickle.load(open("./store/15sc_all_Y_train.p", "rb"))
#all_Y_test = pickle.load(open("./store/15sc_all_Y_test.p", "rb"))
#
## data
#all_X_train = pickle.load(open("./store/15sc_all_X_train.p", "rb"))
#all_X_test = pickle.load(open("./store/15sc_all_X_test.p", "rb"))

#import ipdb
#ipdb.set_trace()

all_votes = []
all_Y_hat = []
all_loss = []
crf = LatentTRF(n_states = 4, n_classes = 2, n_features = 200, inference_method='qpbo')

for q in xrange(tot_classes):
    print '*-* class %d' % q
    data1 = scipy.io.loadmat('./15sc_data/15sc_c' + '%02d' % (q+1) + '.mat')
    X1 = data1['llc_feat']
    num1 = X1.size
    h,w,f = X1[0,0].shape

    X = np.ndarray((num1, h, w, f))
    #Y_pos = np.ndarray(num1)
    for k in xrange(num1):
        X[k,:,:,:] = X1[0,k]
        #Y_pos[k] = q
    #X_train_pos, X_test_pos, Y_train_pos, Y_test_pos = train_test_split(X_pos, Y_pos, train_size=100)

    # TODO: save these
    #all_energies_q = []
    #all_y_pred_q = []
    #all_h_pred_q = []

    votes = np.zeros((num1, tot_classes))
    cnt = 0
    for i in xrange(tot_classes):
        for j in xrange(i+1, tot_classes):
            print 'class %d, evaluating machine (%d,%d)' % (q,i,j)
            clf = pickle.load(open('./15sc_store/15sc_clf_c%d%d.p' % (i+1,j+1), 'rb'))
            cnt = cnt + 1

            energies = np.ndarray(X.shape[0])
            y_pred = np.ndarray(X.shape[0])
            h_pred = np.ndarray(X.shape[:3])

            bar = progressbar.ProgressBar(maxval=num1, \
                        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
            bar.start()
            for k in xrange(num1):
                #print "class %d, machine (%d,%d): predict training image %d" % (q,i,j,k)
                y_pred[k], energies[k], h_temp = crf.inference(X[k], clf.w, return_energy=True)
                #h_pred[k] = np.reshape(h_temp, X.shape[1:3])
                if y_pred[k] == 0:
                    votes[k,i] = votes[k,i] + 1
                elif y_pred[k] == 1:
                    votes[k,j] = votes[k,j] + 1

                bar.update(k+1)
            bar.finish()

            #all_energies_q.append(energies)
            #all_y_pred_q.append(y_pred)
            #all_h_pred_q.append(h_pred)

    Y_hat = np.ndarray(num1)
    loss = 0
    for k in xrange(num1):
        Y_hat[k] = np.argmax(votes[k])
        if Y_hat[k] != q:
            loss = loss + 1

    print '*-* loss for class %d is %d' % (q,loss)

    all_votes.append(votes)
    all_Y_hat.append(Y_hat)
    all_loss.append(loss)




