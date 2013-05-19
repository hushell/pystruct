import numpy as np

from .base import StructuredModel
from .utils import crammer_singer_psi
from . import GridCRF, DirectionalGridCRF
#from .latent_graph_crf import kmeans_init, LatentGraphCRF
from ..utils import make_grid_edges

class LatentTRF(GridCRF):
    """Latent topic TRF with grid graph.
    """
    def __init__(self, n_states, n_classes, n_features,
                 inference_method='lp'):
        GridCRF.__init__(self, n_states, n_features,
                          inference_method=inference_method)

        # TODO: allow different number of topics
        self.n_states = n_states # n_topics_per_class
        self.n_classes = n_classes
        self.n_features = n_features # size of dict

        # TODO: add class_weights

        # one weight-vector per class
        self.size_psi = n_classes * (n_features + n_states
                          + n_states * (n_states+1) / 2)

    def init_latent(self, X, Y):
        # treat all edges the same
        edges = [[self.get_edges(x)] for x in X]
        features = np.array([self.get_features(x) for x in X])
        return kmeans_init(features, Y, edges, n_labels=self.n_labels,
                           n_states_per_label=self.n_states_per_label)

    def get_edges(self, x):
        """ x, shape (h, w, n_features)
        """
        return make_grid_edges(x, neighborhood=self.neighborhood)

    def get_features(self, x):
        """ return shape (h*w, n_features + 1)
        """
        x = np.dstack((x, np.ones(x.shape[0],x.shape[1])))
        return x.reshape(-1, self.n_features + 1)

    def get_unary_potentials(self, x, y, w):
        """Computes unary potentials for x and w.

        Parameters
        ----------
        x : tuple
            Instance Representation.

        y : class

        w : ndarray, shape=(size_psi,)
            [ ..., beta^i_1,theta^i_1, ..., beta^i_K,theta^i_K, ... ]
            [ ..., M,       1,         ..., M,       1,         ... ]

        Returns
        -------
        unary : ndarray, shape=(h*w, n_states)
            Unary weights.
        """
        self._check_size_w(w)
        self._check_size_x(x)
        features, edges = self.get_features(x), self.get_edges(x)
        unit_len = self.n_states * self.n_features + self.n_states;
        unary_params = w[y * unit_len : (y+1) * unit_len].reshape(
            self.n_states, self.n_features + 1)

        return np.dot(features, unary_params.T)

    def get_pairwise_potentials(self, x, y, w):
        """Computes pairwise potentials for x and w.

        Parameters
        ----------
        x : tuple
            Instance Representation.

        y : class

        w : ndarray, shape=(size_psi,)
            Weight vector for CRF instance.

        Returns
        -------
        pairwise : ndarray, shape=(n_states, n_states)
            Pairwise weights.
        """
        self._check_size_w(w)
        self._check_size_x(x)

        unary_offset = self.n_classes * (self.n_states * self.n_features + self.n_states)
        unit_len = self.n_states * (self.n_states + 1) / 2
        pairwise_flat = np.asarray(w[unary_offset + y * unit_len : unary_offset + (y+1) * unit_len])

        pairwise_params = np.zeros((self.n_states, self.n_states))
        # set lower triangle of matrix, then make symmetric
        # we could try to redo this using ``scipy.spatial.distance`` somehow
        pairwise_params[np.tri(self.n_states, dtype=np.bool)] = pairwise_flat
        return (pairwise_params + pairwise_params.T -
                np.diag(np.diag(pairwise_params)))

    def psi(self, x, h):
        """Feature vector associated with instance (x, y).

        Feature representation psi, such that the energy of the configuration
        (x, y) and a weight vector w is given by np.dot(w, psi(x, y)).

        Parameters
        ----------
        x : ndarray, shape (height, width, n_features,)

        y : no need, see self.inference

        h : ndarray, shape (height, width,)

        Returns
        -------
        p : ndarray, shape (size_psi,)
            Feature vector associated with state (x, y, h).

        """
        self._check_size_x(x)
        features, edges = self.get_features(x), self.get_edges(x)
        n_nodes = features.shape[0]

        if isinstance(h, tuple):
            # TODO: refine later
            # h can also be continuous (from lp)
            # in this case, it comes with accumulated edge marginals
            # h is result of relaxation, tuple of unary and pairwise marginals
            unary_marginals, pw = y
            unary_marginals = unary_marginals.reshape(n_nodes, self.n_states)
            # accumulate pairwise
            pw = pw.reshape(-1, self.n_states, self.n_states).sum(axis=0)
        else:
            h = h.reshape(n_nodes)
            gx = np.ogrid[:n_nodes]

            #make one hot encoding
            unary_marginals = np.zeros((n_nodes, self.n_states), dtype=np.int)
            gx = np.ogrid[:n_nodes]
            unary_marginals[gx, h] = 1 # final matrix of hidden, sum column = 1

            ##accumulated pairwise
            pw = np.dot(unary_marginals[edges[:, 0]].T,
                        unary_marginals[edges[:, 1]])

        unaries_prior = np.zeros((self.n_states,1), dtype=np.int)
        for t in xrange(self.n_states):
            unaries_prior[t] = np.sum(h == t)

        unaries_acc = np.dot(unary_marginals.T, features)
        unaries_acc = np.hstack((unaries_acc, unaries_prior))

        pw = pw + pw.T - np.diag(np.diag(pw))  # make symmetric

        psi_vector = np.hstack([unaries_acc.ravel(),
                                pw[np.tri(self.n_states, dtype=np.bool)]])
        return psi_vector

        #result = np.zeros((self.n_classes, self.size_psi / self.n_classes))
        #result[y,:] = psi_vector
        #return result

        #result = np.zeros((self.n_states, self.n_features, self.n_classes))
        #for t in xrange(self.n_states):
        #    indx = np.mgrid[0:height,0:width];

        #    result[t,:,y] = np.sum(x & (h==t), axis=0)
        #return result.ravel()

    def inference(self, x, w, relaxed=None, return_energy=False):
        """Inference for x using parameters w.

        Finds armin_y np.dot(w, psi(x, h)), i.e. best possible prediction.

        Parameters
        ----------
        x : ndarray, shape (n_features,)
            Input sample features.

        w : ndarray, shape=(size_psi,)
            Parameters of the SVM.

        relaxed : ignored

        Returns
        -------
        y_pred : int
            Predicted class label.
        """
        self.inference_calls += 1
        scores = np.dot(w.reshape(self.n_classes, -1), self.psi(x, w))
        if return_energy:
            return np.argmax(scores), np.max(scores)
        return np.argmax(scores)

    def loss_augmented_inference(self, x, h, w, relaxed=False,
                                 return_energy=False):
        self.inference_calls += 1
        self._check_size_w(w)
        unary_potentials = self.get_unary_potentials(x, w)
        pairwise_potentials = self.get_pairwise_potentials(x, w)
        edges = self.get_edges(x)
        # do loss-augmentation
        for l in np.arange(self.n_states):
            # for each class, decrement features
            # for loss-agumention
            unary_potentials[self.label_from_latent(h)
                             != self.label_from_latent(l), l] += 1.

        return inference_dispatch(unary_potentials, pairwise_potentials, edges,
                                  self.inference_method, relaxed=relaxed,
                                  return_energy=return_energy)

    def latent(self, x, y, w):
        # TODO: let h encode y
        unary_potentials = self.get_unary_potentials(x, y, w)
        pairwise_potentials = self.get_pairwise_potentials(x, y, w)
        edges = self.get_edges(x)
        h = inference_dispatch(unary_potentials, pairwise_potentials, edges,
                               self.inference_method, relaxed=False)
        return h

    def loss(self, y, y_hat):
        return self.class_weight[y] * (y != y_hat)

    #def loss(self, h, h_hat):
    #    if isinstance(h_hat, tuple):
    #        return self.continuous_loss(h, h_hat[0])
    #    return GraphCRF.loss(self, self.label_from_latent(h),
    #                         self.label_from_latent(h_hat))

    #def continuous_loss(self, y, y_hat):
    #    # continuous version of the loss
    #    # y_hat is the result of linear programming
    #    y_hat_org = np.zeros((y_hat.shape[0], self.n_labels))
    #    for s in xrange(self.n_states):
    #        y_hat_org[:, self._states_map[s]] += y_hat[:, s]
    #    y_org = self.label_from_latent(y)
    #    return GraphCRF.continuous_loss(self, y_org, y_hat_org)

