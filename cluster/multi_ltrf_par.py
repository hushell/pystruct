import numpy as np

from sklearn.cross_validation import train_test_split
from pystruct.models import LatentTRF
#from pystruct.learners import LatentSSVM
from pystruct.learners import LatentSubgradientSSVM

import sys
import scipy.io
import pickle

rt_path = '/nfs/stak/students/h/huxu/python-cluster/git-dir/pystruct/nips/'

def run_experiment(datafile1, datafile2, outfile, n_states = 4, n_classes = 2, n_features = 200):
    # data
    data1 = scipy.io.loadmat(rt_path + datafile1)
    data2 = scipy.io.loadmat(rt_path + datafile2)
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
    crf = LatentTRF(n_states = n_states, n_classes = n_classes, n_features = n_features, inference_method='qpbo')

    # learner NOTE: C value need to be tuned
    clf = LatentSubgradientSSVM(
        model=crf, max_iter=500, C=10., verbose=2,
        n_jobs=-1, learning_rate=0.1, show_loss_every=10)
    # training
    clf.fit(X_train, Y_train)
    pickle.dump(clf, open(rt_path + outfile, 'wb'))

    #import ipdb
    #ipdb.set_trace()


if __name__ == "__main__":
    #if len(sys.argv) < 4:
    #    raise ValueError("num of argv is not enough!"))

    datafile1   =    sys.argv[1]
    datafile2   =    sys.argv[2]
    outfile    =     sys.argv[3]
    n_states   = int(sys.argv[4])
    n_classes  = int(sys.argv[5])
    n_features = int(sys.argv[6])

    print '^-^ input params are: '
    print (datafile1, datafile2, outfile, n_states, n_classes, n_features)
    run_experiment(datafile1=datafile1, datafile2=datafile2, outfile=outfile, \
            n_states=n_states, n_classes=n_classes, n_features=n_features)
