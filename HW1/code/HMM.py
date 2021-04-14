import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
import pdb

class HMM(object):
    """A class for implementing HMMs.

    Attributes
    ----------
    envShape : list
        A two element list specifying the shape of the environment
    states : list
        A list of states specified by their (x, y) coordinates
    observations : list
        A list specifying the sequence of observations
    T : numpy.ndarray
        An N x N array encoding the transition probabilities, where
        T[i,j] is the probability of transitioning from state i to state j.
        N is the total number of states (envShape[0]*envShape[1])
    M : numpy.ndarray
        An M x N array encoding the emission probabilities, where
        M[k,i] is the probability of observing k from state i.
    pi : numpy.ndarray
        An N x 1 array encoding the prior probabilities

    Methods
    -------
    train(observations)
        Estimates the HMM parameters using a set of observation sequences
    viterbi(observations)
        Implements the Viterbi algorithm on a given observation sequence
    setParams(T, M, pi)
        Sets the transition (T), emission (M), and prior (pi) distributions
    getParams
        Queries the transition (T), emission (M), and prior (pi) distributions
    sub2ind(i, j)
        Convert integer (i,j) coordinates to linear index.
    """

    def __init__(self, envShape, T=None, M=None, pi=None):
        """Initialize the class.

        Attributes
        ----------
        envShape : list
            A two element list specifying the shape of the environment
        T : numpy.ndarray, optional
            An N x N array encoding the transition probabilities, where
            T[i,j] is the probability of transitioning from state j to state i.
            N is the total number of states (envShape[0]*envShape[1])
        M : numpy.ndarray, optional
            An M x N array encoding the emission probabilities, where
            M[k,i] is the probability of observing k from state i.
        pi : numpy.ndarray, optional
            An N x 1 array encoding the prior probabilities
        """
        self.envShape = envShape 
        self.numStates = envShape[0] * envShape[1]

        if T is None:
            # Initial estimate of the transition function
            # where T[sub2ind(i',j'), sub2ind(i,j)] is the likelihood
            # of transitioning from (i,j) --> (i',j')
            self.T = np.zeros((self.numStates, self.numStates))

            # Self-transitions
            for i in range(self.numStates):
                self.T[i, i] = 0.2 

            # Black rooms
            self.T[self.sub2ind(0, 0), self.sub2ind(0, 0)] = 1.0
            self.T[self.sub2ind(1, 1), self.sub2ind(1, 1)] = 1.0
            self.T[self.sub2ind(0, 3), self.sub2ind(0, 3)] = 1.0
            self.T[self.sub2ind(3, 2), self.sub2ind(3, 2)] = 1.0

            # (2, 0) -->
            self.T[self.sub2ind(1, 0), self.sub2ind(2, 0)] = 0.8/3.0
            self.T[self.sub2ind(2, 1), self.sub2ind(2, 0)] = 0.8/3.0
            self.T[self.sub2ind(3, 0), self.sub2ind(2, 0)] = 0.8/3.0

            # (3, 0) -->
            self.T[self.sub2ind(2, 0), self.sub2ind(3, 0)] = 0.8/2.0
            self.T[self.sub2ind(3, 1), self.sub2ind(3, 0)] = 0.8/2.0

            # (0, 1) --> (0, 2)
            self.T[self.sub2ind(0, 2), self.sub2ind(0, 1)] = 0.8

            # (2, 1) -->
            self.T[self.sub2ind(2, 0), self.sub2ind(2, 1)] = 0.8/3.0
            self.T[self.sub2ind(3, 1), self.sub2ind(2, 1)] = 0.8/3.0
            self.T[self.sub2ind(2, 2), self.sub2ind(2, 1)] = 0.8/3.0

            # (3, 1) -->
            self.T[self.sub2ind(2, 1), self.sub2ind(3, 1)] = 0.8/2.0
            self.T[self.sub2ind(3, 0), self.sub2ind(3, 1)] = 0.8/2.0

            # (0, 2) -->
            self.T[self.sub2ind(0, 1), self.sub2ind(0, 2)] = 0.8/2.0
            self.T[self.sub2ind(1, 2), self.sub2ind(0, 2)] = 0.8/2.0

            # (1, 2) -->
            self.T[self.sub2ind(0, 2), self.sub2ind(1, 2)] = 0.8/3.0
            self.T[self.sub2ind(2, 2), self.sub2ind(1, 2)] = 0.8/3.0
            self.T[self.sub2ind(1, 3), self.sub2ind(1, 2)] = 0.8/3.0

            # (2, 2) -->
            self.T[self.sub2ind(1, 2), self.sub2ind(2, 2)] = 0.8/2.0
            self.T[self.sub2ind(2, 3), self.sub2ind(2, 2)] = 0.8/2.0

            # (1, 3) -->
            self.T[self.sub2ind(1, 2), self.sub2ind(1, 3)] = 0.8/2.0
            self.T[self.sub2ind(2, 3), self.sub2ind(1, 3)] = 0.8/2.0

            # (2, 3) -->
            self.T[self.sub2ind(1, 3), self.sub2ind(2, 3)] = 0.8/3.0
            self.T[self.sub2ind(3, 3), self.sub2ind(2, 3)] = 0.8/3.0
            self.T[self.sub2ind(2, 2), self.sub2ind(2, 3)] = 0.8/3.0

            # (3, 3) --> (2, 3)
            self.T[self.sub2ind(2, 3), self.sub2ind(3, 3)] = 0.8
        else:
            self.T = T

        if M is None:
            # Initial estimates of emission likelihoods, where
            # M[k, sub2ind(i,j)]: likelihood of observation k from state (i, j)
            self.M = np.ones((4, self.numStates)) * 0.1

            # Black states
            self.M[:, self.sub2ind(0, 0)] = 0.25
            self.M[:, self.sub2ind(1, 1)] = 0.25
            self.M[:, self.sub2ind(0, 3)] = 0.25
            self.M[:, self.sub2ind(3, 2)] = 0.25

            self.M[self.obs2ind('r'), self.sub2ind(0, 1)] = 0.7
            self.M[self.obs2ind('g'), self.sub2ind(0, 2)] = 0.7
            self.M[self.obs2ind('g'), self.sub2ind(1, 0)] = 0.7
            self.M[self.obs2ind('b'), self.sub2ind(1, 2)] = 0.7
            self.M[self.obs2ind('r'), self.sub2ind(1, 3)] = 0.7
            self.M[self.obs2ind('y'), self.sub2ind(2, 0)] = 0.7
            self.M[self.obs2ind('g'), self.sub2ind(2, 1)] = 0.7
            self.M[self.obs2ind('r'), self.sub2ind(2, 2)] = 0.7
            self.M[self.obs2ind('y'), self.sub2ind(2, 3)] = 0.7
            self.M[self.obs2ind('b'), self.sub2ind(3, 0)] = 0.7
            self.M[self.obs2ind('y'), self.sub2ind(3, 1)] = 0.7
            self.M[self.obs2ind('b'), self.sub2ind(3, 3)] = 0.7
        else:
            self.M = M

        if pi is None:
            # Initialize estimates of prior probabilities where
            # pi[(i, j)] is the likelihood of starting in state (i, j)
            self.pi = np.ones(16)/12
            self.pi[self.sub2ind(0, 0)] = 0.0
            self.pi[self.sub2ind(1, 1)] = 0.0
            self.pi[self.sub2ind(0, 3)] = 0.0
            self.pi[self.sub2ind(3, 2)] = 0.0
        else:
            self.pi = pi

    def setParams(self, T, M, pi):
        """Set the transition, emission, and prior probabilities."""
        self.T = T
        self.M = M
        self.pi = pi

    def getParams(self):
        """Get the transition, emission, and prior probabilities."""
        return (self.T, self.M, self.pi)

    # Estimate the transition and observation likelihoods and the
    # prior over the initial state based upon training data
    def train(self, observations):
        """Estimate HMM parameters from training data via Baum-Welch.

        Parameters
        ----------
        observations : list
            A list specifying a set of observation sequences
            where observations[i] denotes a distinct sequence
        """
        # This function should set self.T, self.M, and self.pi
        logliks = []
        xi_so_far = np.zeros((0, 200, 16, 16))
        gamma_so_far = np.zeros((0, 201, 16))
        z_so_far = np.zeros((0, 201))
        for idx, z in enumerate(observations):
            
            alpha = self.forward(z)
            beta = self.backward(z)
            gamma = self.computeGamma(alpha, beta)
            xi = self.computeXis(alpha, beta, z)
            logliks.append(self.log_obs_prob(alpha))
            px_t = alpha[-1].sum()
            
            xi = xi/px_t
            gamma = gamma/px_t

            xi_so_far = np.concatenate((xi_so_far, np.expand_dims(xi, axis=0)), axis=0)
            gamma_so_far = np.concatenate((gamma_so_far, np.expand_dims(gamma, axis=0)), axis=0)
            z_so_far = np.concatenate((z_so_far, np.expand_dims(np.array(z), axis=0)), axis=0)

            new_pi = gamma_so_far[:, 0, :].sum(axis=0)/gamma_so_far.shape[0]

            xi_sum = xi_so_far.sum(axis=(0,1))
            gamma_sum = gamma_so_far[:, :-1, :].sum(axis=(0,1)) 
            gamma_sum = np.where(gamma_sum <= 0, 1e-200, gamma_sum) #prevent division by zero

            gamma_sum2 = gamma_so_far.sum(axis=(0,1))
            gamma_sum2 = np.where(gamma_sum2 <= 0, 1e-200, gamma_sum2) #prrevent division by zero

            new_T = xi_sum/gamma_sum
            new_M = np.zeros((4, self.numStates))

            for m in 'rgby':
                z_equal_to_m = (z_so_far==m).astype(int)
                new_M[self.obs2ind(m), :] = (gamma_so_far*z_equal_to_m.reshape(z_equal_to_m.shape[0], -1, 1)).sum(axis=(0,1))/gamma_sum2
            
            self.setParams(new_T, new_M, new_pi)

        plt.plot(list(range(len(logliks))), logliks)
        plt.show()

    def viterbi(self, z):
        """Implement the Viterbi algorithm.

        Parameters
        ----------
        z : list
            A list specifying a single sequence of observations, where each o
            observation is a string (e.g., 'r')

        Returns
        -------
        states : list
            List of predicted sequence of states, each specified as (x, y) pair
        """
        # CODE GOES HERE
        # Return the list of predicted states, each specified as (x, y) pair
        for i in range(len(z)):
            if i == 0:
                pre = np.zeros((0, self.numStates)) 
                delta = self.pi*self.M[self.obs2ind(z[0]), :]
            else:
                pre = np.vstack((pre, np.argmax(self.T*delta, axis=1))) #matrix of argmaxes
                delta = self.M[self.obs2ind(z[i]), :] * np.max(self.T * delta, axis=1)
        m = np.argmax(delta).astype(int)
        lst = [divmod(m, self.envShape[1])] #use divmod to convert index back to (x,y) pair
        for i in reversed(range(len(pre))):
            m = pre[i,m].astype(int)
            lst.append(divmod(m, self.envShape[1]))
        return list(reversed(lst))

    def forward(self, z): 
        """Find all alpha_t(i).

        Parameters
        ----------
        z: list
            A list of length T specifying the sequence of observations, where each
            observation is a string (e.g., 'r')

        Returns
        -------
        alpha: numpy.ndarray
            An array of shape T x N where entry (t, i) denote the value of alpha_t(i) (as defined in lecture)
        
        """
        assert len(z) > 0
        alpha = np.zeros((0, self.numStates))
        for i in range(len(z)):
            if i == 0:
                alpha = np.vstack((alpha, self.M[self.obs2ind(z[0]), :] * self.pi))
            else:
                #find alpha for the next time step, then stack the old alphas on top of the next alpha 
                alpha = np.vstack((alpha, self.M[self.obs2ind(z[i]), :] * np.sum(self.T * alpha[-1], axis=1)))
        return alpha

    def backward(self, z):
        """Find all beta_t(i).

        Parameters
        ----------
        z: list
            A list of length T specifying the sequence of observations, where each
            observation is a string (e.g., ['r','b','r'])

        Returns
        -------
        beta: numpy.ndarray
            An array of shape T x N where entry (t, i) denote the value of beta_t(i) (as defined in lecture)
        """
        assert len(z) > 0
        beta = np.expand_dims(np.ones(self.numStates), axis=0)
        for i in range(len(z)-1, 0, -1):
            #expand dim to broadcast along columns instead of rows
            #stack newest beta on top of previously computed betas
            beta = np.vstack((np.sum(np.expand_dims(self.M[self.obs2ind(z[i]), :] * beta[0], axis=1) * self.T, axis=0) ,beta))
        return beta

    def log_obs_prob(self, alpha):
        return np.log(alpha[-1].sum())

    def computeGamma(self, alpha, beta):
        return alpha*beta #not divided by P(Z^T)

    def computeXis(self, alpha, beta, z):
        """Compute xi as an array comprised of each xi-xj pair
        Parameters
        ----------
        alpha: np.array
            An array of shape T x N calculated using the forward function above, based on z
        beta: np.array
            An array of shape T x N calculated using the backward function above, based on z
        z: np.array
            A list of length T specifying the sequence of observations, based on which alpha and beta are computed
        Returns
        -----------
        xi: np.array
            An array of shape (T-1) x N x N that contains xi_t(i,j) at entry (t,j,i) (similar to self.T: the fromState is determined by the last dimension)
        """
        xi = np.zeros((0, self.numStates, self.numStates))
        for t in range(len(z)-1):
            xi_t = alpha[t, :]* self.T *np.expand_dims(self.M[self.obs2ind(z[t+1]), :]*
            beta[t+1,:], axis=1) #expand dim to broadcast along columns
            xi = np.concatenate((xi, np.expand_dims(xi_t, axis=0)), axis=0) #expand dim to make shapes equal, except for first axis
        return xi #not divided by P(Z^T)

    def getLogStartProb(self, state):
        """Return the log (starting) probability of a particular state."""
        return np.log(self.pi[state])

    def getLogTransProb(self, fromState, toState):
        """Return the log probability associated with a state transition."""
        return np.log(self.T[toState, fromState])

    def getLogOutputProb(self, state, output):
        """Return the log probability of a state-dependent observation."""
        return np.log(self.M[output, state])

    def sub2ind(self, i, j):
        """Convert subscript (i,j) to linear index."""
        return (self.envShape[1]*i + j)

    def obs2ind(self, obs):
        """Convert observation string to linear index."""
        obsToInt = {'r': 0, 'g': 1, 'b': 2, 'y': 3}
        return obsToInt[obs]
