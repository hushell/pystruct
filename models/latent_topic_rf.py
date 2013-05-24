import numpy as np
import pylab as plt

from .base import StructuredModel
from .utils import crammer_singer_psi
from . import GridCRF
#from .latent_graph_crf import kmeans_init, LatentGraphCRF
from ..utils import make_grid_edges
from ..inference import inference_dispatch

class LatentTRF(GridCRF):
    """Latent topic TRF with grid graph.
    """
    def __init__(self, n_states, n_classes, n_features, class_weight=None,
                 inference_method='lp'):
        """
        n_features : size of dict
        n_states   : num of topics per class
        """
        GridCRF.__init__(self, n_states, n_features,
                          inference_method=inference_method)

        # TODO: allow different number of topics
        self.n_states = n_states # n_topics_per_class
        self.n_classes = n_classes
        self.n_features = n_features # size of dict

        # TODO: add class_weights
        # self.rescale_C = rescale_C
        if class_weight is not None:
            if len(class_weight) != n_classes:
                raise ValueError("class_weight must have length n_classes or"
                                 " be None")
            class_weight = np.array(class_weight)
        else:
            class_weight = np.ones(n_classes)
        self.class_weight = class_weight

        # one weight-vector per class
        self.size_psi = n_classes * (n_states * n_features + n_states
                          + n_states * (n_states+1) / 2)

    def init_latent(self, X, Y):
        """ randomly or load from PLSA results
        """
        return np.random.randint(low=0, high=self.n_states, size=Y.shape)

    def get_edges(self, x):
        """ x, shape (h, w, n_features)
        """
        return make_grid_edges(x, neighborhood=self.neighborhood)

    def get_features(self, x):
        """ return shape (h*w, n_features + 1)
        """
        #x = np.dstack((x, np.ones(x.shape[:-1])))
        return x.reshape(-1, self.n_features)

    def get_unary_potentials(self, x, y, w):
        """Computes unary potentials for x and w.

        Parameters
        ----------
        x : tuple
            Instance Representation.

        y : class

        w : ndarray, shape=(size_psi,)
            [ ..., beta^i_1,theta^i_1, ..., beta^i_K,theta^i_K, eta^i,     ... ]
            [ ..., M,       1,         ..., M,       1,         K*(K+1)/2, ... ]

        Returns
        -------
        unary : ndarray, shape=(h*w, n_states)
            Unary weights.
        """
        self._check_size_w(w)
        #self._check_size_x(x)
        features, edges = self.get_features(x), self.get_edges(x)
        features = np.hstack((features, np.ones((features.shape[0], 1))))

        unary_len = self.n_states * self.n_features + self.n_states;
        pair_len = self.n_states * (self.n_states + 1) / 2
        unary_params = w[y * (unary_len+pair_len) : (y+1) * (unary_len+pair_len) - pair_len].reshape(
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

        unary_len = self.n_states * self.n_features + self.n_states;
        pair_len = self.n_states * (self.n_states + 1) / 2
        pairwise_flat = np.asarray(w[y * (unary_len+pair_len) + unary_len : (y+1) * (unary_len+pair_len)])

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

        h,y = h

        if isinstance(h, tuple):
            # TODO: refine later
            # h can also be continuous (from lp)
            # in this case, it comes with accumulated edge marginals
            # h is result of relaxation, tuple of unary and pairwise marginals
            unary_marginals, pw = h
            unary_marginals = unary_marginals.reshape(n_nodes, self.n_states)
            # accumulate pairwise
            pw = pw.reshape(-1, self.n_states, self.n_states).sum(axis=0)
        else:
            h = h.reshape(n_nodes)
            gx = np.ogrid[:n_nodes]

            #make one shot encoding
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

        # TODO: normalization
        psi_vector = np.hstack([unaries_acc.ravel(),
                                pw[np.tri(self.n_states, dtype=np.bool)]])
        #return psi_vector

        result = np.zeros((self.n_classes, self.size_psi / self.n_classes))
        result[y,:] = psi_vector
        return result.ravel()

    def inference(self, x, w, relaxed=None, return_energy=False, unaries_only=False):
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
        #self.inference_calls += 1

        scores = np.zeros(self.n_classes)
        h = np.zeros((self.n_classes, x.shape[0]*x.shape[1]), dtype=np.int) # h has been flatten already
        for i in xrange(self.n_classes):
            if unaries_only == False:
                h[i],_ = self.latent(x, i, w)
            else:
                h[i],_ = self.latent_unary(x, i, w)
            scores[i] = np.dot(w, self.psi(x, (h[i],i)))

        y = np.argmax(scores)
        if return_energy:
            return (y, np.max(scores), h[y])
        return y

    def loss_augmented_inference(self, x, h, w, relaxed=False,
                                 return_energy=False):
        """Loss-augmented inference for x and y using parameters w.

        Minimizes over y_hat:
        np.dot(psi(x, y_hat), w) + loss(y, y_hat)

        Parameters
        ----------
        x : ndarray, shape (n_features,)
            Unary evidence / input to augment.

        y : int
            Ground truth labeling relative to which the loss
            will be measured.

        w : ndarray, shape (size_psi,)
            Weights that will be used for inference.

        Returns
        -------
        y_hat : int
            Label with highest sum of loss and score.
        """
        self.inference_calls += 1

        h,y = h
        scores = np.zeros(self.n_classes)
        h_hat = np.zeros((self.n_classes, h.shape[0]), dtype=np.int) # h has been flatten already
        for i in xrange(self.n_classes):
            h_hat[i],_ = self.latent(x, i, w)
            scores[i] = np.dot(w, self.psi(x, (h_hat[i],i)))

        other_classes = np.arange(self.n_classes) != y
        scores[other_classes] += self.class_weight[y] #TODO: diff weights for diff classes
        #if self.rescale_C:
        #    scores[other_classes] += 1
        #else:
        #    scores[other_classes] += self.class_weight[y]

        if return_energy:
            return np.argmax(scores), np.max(scores)
        y_hat = np.argmax(scores)
        return (h_hat[y_hat], y_hat)

    def latent(self, x, y, w):
        unary_potentials = self.get_unary_potentials(x, y, w)
        pairwise_potentials = self.get_pairwise_potentials(x, y, w)
        edges = self.get_edges(x)
        h = inference_dispatch(unary_potentials, pairwise_potentials, edges,
                               self.inference_method, relaxed=False)
        h.astype(np.int)
        return (h, y)

    def latent_unary(self, x, y, w):
        unary_potentials = self.get_unary_potentials(x, y, w)
        pairwise_potentials = np.ones((self.n_states, self.n_states))
        edges = self.get_edges(x)
        h = inference_dispatch(unary_potentials, pairwise_potentials, edges,
                               self.inference_method, relaxed=False)
        h.astype(np.int)
        return (h, y)

    # TODO: continuous LOSS
    # TODO: lots of other tricks can be applied here instead of 0-1 loss
    def loss(self, h, h_hat):
        if isinstance(h, tuple):
            h,y = h
            h_hat,y_hat = h_hat
        else:
            y = h
            y_hat = h_hat
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

    def visual_topics(self, topics, cls, w):
        pass

    def visual_topic_priors(self, y, w):
        unary_len = self.n_states * self.n_features + self.n_states;
        pair_len = self.n_states * (self.n_states + 1) / 2
        unary_params = w[y * (unary_len+pair_len) : (y+1) * (unary_len+pair_len) - pair_len].reshape(
            self.n_states, self.n_features + 1)
        prior = unary_params[:,-1]
        xaxis = np.arange(1, self.n_states+1)
        width = 1
        plt.bar(xaxis, prior, width, color="y" )

    def visual_topic_inter(self, y, w):
        unary_len = self.n_states * self.n_features + self.n_states;
        pair_len = self.n_states * (self.n_states + 1) / 2
        pairwise_flat = np.asarray(w[y * (unary_len+pair_len) + unary_len : (y+1) * (unary_len+pair_len)])
        pairwise_params = np.zeros((self.n_states, self.n_states))
        pairwise_params[np.tri(self.n_states, dtype=np.bool)] = pairwise_flat
        pairwise_params = (pairwise_params + pairwise_params.T -
                np.diag(np.diag(pairwise_params)))

        cax = plt.matshow(pairwise_params )
        plt.colorbar(cax)
        plt.show()
