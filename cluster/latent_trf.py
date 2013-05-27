import numpy as np
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split

from pystruct.models import LatentTRF
#from pystruct.learners import LatentSSVM
from pystruct.learners import LatentSubgradientSSVM

import sys

def run_experiment(datafile, outfile, c1, c2, n_states = 4, n_classes = 2, n_features = 200):
    import scipy.io
    data = scipy.io.loadmat("/nfs/stak/students/h/huxu/python-cluster/git-dir/pystruct/nips/" + datafile)
    X1 = data[c1]
    X2 = data[c2]
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

    crf = LatentTRF(n_states = n_states, n_classes = n_classes, n_features = n_features)

    # NOTE: C value
    clf = LatentSubgradientSSVM(
        model=crf, max_iter=500, C=10., verbose=2,
        n_jobs=-1, learning_rate=0.1, show_loss_every=10)

    #import ipdb
    #ipdb.set_trace()

    clf.fit(X, Y)

    import pickle
    pickle.dump(clf, open("/nfs/stak/students/h/huxu/python-cluster/git-dir/pystruct/nips/" + outfile, "wb"))

if __name__ == "__main__":
    #if len(sys.argv) < 4:
    #    raise ValueError("num of argv is not enough!"))

    datafile   =     sys.argv[1]
    outfile    =     sys.argv[2]
    c1         =     sys.argv[3]
    c2         =     sys.argv[4]
    n_states   = int(sys.argv[5])
    n_classes  = int(sys.argv[6])
    n_features = int(sys.argv[7])

    run_experiment(datafile=datafile, outfile=outfile, c1=c1, c2=c2,
            n_states=n_states, n_classes=n_classes, n_features=n_features)