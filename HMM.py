########################################
# CS/CNS/EE 155 2018
# Problem Set 6
#
# Author:       Andrew Kang
# Description:  Set 6 skeleton code
########################################

# You can use this (optional) skeleton code to complete the HMM
# implementation of set 5. Once each part is implemented, you can simply
# execute the related problem scripts (e.g. run 'python 2G.py') to quickly
# see the results from your code.
#
# Some pointers to get you started:
#
#     - Choose your notation carefully and consistently! Readable
#       notation will make all the difference in the time it takes you
#       to implement this class, as well as how difficult it is to debug.
#
#     - Read the documentation in this file! Make sure you know what
#       is expected from each function and what each variable is.
#
#     - Any reference to "the (i, j)^th" element of a matrix T means that
#       you should use T[i][j].
#
#     - Note that in our solution code, no NumPy was used. That is, there
#       are no fancy tricks here, just basic coding. If you understand HMMs
#       to a thorough extent, the rest of this implementation should come
#       naturally. However, if you'd like to use NumPy, feel free to.
#
#     - Take one step at a time! Move onto the next algorithm to implement
#       only if you're absolutely sure that all previous algorithms are
#       correct. We are providing you waypoints for this reason.
#
# To get started, just fill in code where indicated. Best of luck!

import random
import numpy as np

class HiddenMarkovModel:
    '''
    Class implementation of Hidden Markov Models.
    '''

    def __init__(self, A, O):
        '''
        Initializes an HMM. Assumes the following:
            - States and observations are integers starting from 0.
            - There is a start state (see notes on A_start below). There
              is no integer associated with the start state, only
              probabilities in the vector A_start.
            - There is no end state.

        Arguments:
            A:          Transition matrix with dimensions L x L.
                        The (i, j)^th element is the probability of
                        transitioning from state i to state j. Note that
                        this does not include the starting probabilities.

            O:          Observation matrix with dimensions L x D.
                        The (i, j)^th element is the probability of
                        emitting observation j given state i.

        Parameters:
            L:          Number of states.

            D:          Number of observations.

            A:          The transition matrix.

            O:          The observation matrix.

            A_start:    Starting transition probabilities. The i^th element
                        is the probability of transitioning from the start
                        state to state i. For simplicity, we assume that
                        this distribution is uniform.
        '''

        self.L = len(A)
        self.D = len(O[0])
        self.A = A
        self.O = O
        self.A_start = [1. / self.L for _ in range(self.L)]


    def viterbi(self, x):
        '''
        Uses the Viterbi algorithm to find the max probability state
        sequence corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            max_seq:    State sequence corresponding to x with the highest
                        probability.
        '''

        M = len(x)      # Length of sequence.

        # The (i, j)^th elements of probs and seqs are the max probability
        # of the prefix of length i ending in state j and the prefix
        # that gives this probability, respectively.
        #
        # For instance, probs[1][0] is the probability of the prefix of
        # length 1 ending in state 0.
        probs = [[0. for _ in range(self.L)] for _ in range(M + 1)]
        seqs = [['' for _ in range(self.L)] for _ in range(M + 1)]

        # Problem 2A

        # transitions from starting state
        for prob in self.A_start:
            for state in range(self.L):
                probs[1][state] = prob * self.O[state][x[0]]
                seqs[1][state] = str(state)

        # fill in the rest of probs and seqs
        for i in range(1, M):
            obs = x[i]
            # solve for best length-(i+1) prefix ending in each state
            for state_prev in range(self.L):
                for state_next in range(self.L):
                    p = probs[i][state_prev] * self.A[state_prev][state_next] * self.O[state_next][obs]
                    if p > probs[i+1][state_next]:
                        probs[i+1][state_next] = p
                        seqs[i+1][state_next] = seqs[i][state_prev] + str(state_next)

        # predict best state sequence of length M
        idx = probs[M].index(max(probs[M]))
        max_seq = seqs[M][idx]

        return max_seq


    def forward(self, x, normalize=False):
        '''
        Uses the forward algorithm to calculate the alpha probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            alphas:     Vector of alphas.

                        The (i, j)^th element of alphas is alpha_j(i),
                        i.e. the probability of observing prefix x^1:i
                        and state y^i = j.

                        e.g. alphas[1][0] corresponds to the probability
                        of observing x^1:1, i.e. the first observation,
                        given that y^1 = 0, i.e. the first state is 0.
        '''

        M = len(x)      # Length of sequence.
        alphas = [[0. for _ in range(self.L)] for _ in range(M + 1)]

        # Problem 2Bi

        # first deal with starting state
        for state in range(self.L):
            alphas[0][state] = 1
            alphas[1][state] = self.O[state][x[0]] * self.A_start[state]

        # compute the remaining alphas
        for i in range(2, M+1):
            for state in range(self.L):
                t = np.array(self.A)
                alphas[i][state] = self.O[state][x[i-1]] * np.dot(alphas[i-1], t[:,state])

        # normalization
        if normalize:
            for i in range(1, len(alphas)):
                norm = sum(alphas[i])
                if norm != 0:
                    for j in range(len(alphas[i])):
                        alphas[i][j] /= norm

        return alphas


    def backward(self, x, normalize=False):
        '''
        Uses the backward algorithm to calculate the beta probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            betas:      Vector of betas.

                        The (i, j)^th element of betas is beta_j(i), i.e.
                        the probability of observing prefix x^(i+1):M and
                        state y^i = j.

                        e.g. betas[M][0] corresponds to the probability
                        of observing x^M+1:M, i.e. no observations,
                        given that y^M = 0, i.e. the last state is 0.
        '''

        M = len(x)      # Length of sequence.
        betas = [[0. for _ in range(self.L)] for _ in range(M + 1)]

        # Problem Bii

        # compute base case of recursion
        for state in range(self.L):
            betas[M][state] = 1

        # compute the remaining betas
        for i in range(M - 1, -1, -1):
            for state in range(self.L):
                s = 0
                for j in range(self.L):
                    s += betas[i+1][j]*self.A[state][j]*self.O[j][x[i]]
                betas[i][state] = s

        # normalization
        if normalize:
            for i in range(0, len(betas)):
                norm = sum(betas[i])
                if norm != 0:
                    for j in range(len(betas[i])):
                        betas[i][j] /= norm

        return betas


    def supervised_learning(self, X, Y):
        '''
        Trains the HMM using the Maximum Likelihood closed form solutions
        for the transition and observation matrices on a labeled
        datset (X, Y). Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of variable length, consisting of integers
                        ranging from 0 to D - 1. In other words, a list of
                        lists.

            Y:          A dataset consisting of state sequences in the form
                        of lists of variable length, consisting of integers
                        ranging from 0 to L - 1. In other words, a list of
                        lists.

                        Note that the elements in X line up with those in Y.
        '''

        # In supervised learning, we update A and O by counting, so the
        # random initialization does not affect the final result.
        # We initialize A and O as all zeros for convenience.
        self.A = [[0. for _ in range(self.L)] for _ in range(self.L)]
        self.O = [[0. for _ in range(self.D)] for _ in range(self.L)]

        # Calculate and store values of denominators for A in M-step formulas
        denoms_A = [0. for i in range(self.L)]
        for i in range(len(Y)):
            for j in range(len(Y[i]) - 1):
                b = Y[i][j]
                denoms_A[b] += 1

        # Calculate and store values of denominators for O
        denoms_O = [0. for i in range(self.L)]
        for i in range(len(Y)):
            for j in Y[i]:
                denoms_O[j] += 1

        # Calculate each element of A using the M-step formulas.
        for j in range(len(Y)):
            for i in range(len(Y[j]) - 1):
                a = Y[j][i+1]
                b = Y[j][i]
                self.A[b][a] += 1 / denoms_A[b]

        # Calculate each element of O using the M-step formulas.
        for j in range(len(Y)):
            for i in range(len(Y[j])):
                w = X[j][i]
                z = Y[j][i]
                self.O[z][w] += 1 / denoms_O[z]


    def unsupervised_learning(self, X, N_iters):
        '''
        Trains the HMM using the Baum-Welch algorithm on an unlabeled
        datset X. Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of length M, consisting of integers ranging
                        from 0 to D - 1. In other words, a list of lists.

            N_iters:    The number of iterations to train on.
        '''

        # Problem 2D
        for iter in range(N_iters):
            if iter % 5 == 0:
                print('Iteration: {}'.format(iter))

            # get alphas and betas for all input sequences in X
            alphas = [[] for j in range(len(X))]
            betas = [[] for j in range(len(X))]
            for j in range(len(X)):
                alphas[j] = self.forward(X[j], normalize=True)
                betas[j] = self.backward(X[j], normalize=True)

            # Update A
            A_copy = np.copy(self.A)
            for a in range(self.L):
                #denom = denoms_A[a]
                for b in range(self.L):
                    num = 0
                    for j in range(len(X)):
                        for i in range(1, len(X[j])):
                            num += self.marginal_1(a, b, i, alphas[j], betas[j], A_copy, self.O, X[j])
                    #self.A[a][b] = num / denom
                    self.A[a][b] = num

            # normalize rows
            for i in range(self.L):
                n = sum(self.A[i])
                self.A[i] /= n

            # Update O
            for w in range(self.L):
                #denom = denoms_O[w]
                for z in range(self.D):
                    num = 0
                    for j in range(len(X)):
                        for i in range(1, len(X[j]) + 1):
                            x = X[j][i - 1]
                            if x == z:
                                num += self.marginal_2(w, i, alphas[j], betas[j])
                    #self.O[w][z] = num / denom
                    self.O[w][z] = num

            # normalize rows
            for i in range(self.L):
                n = sum(self.O[i])
                self.O[i] /= n

    def marginal_1(self, a, b, i, alphas, betas, A, O, X):
        '''
        Helper function that computes P(y^i = a, y^(i+1) = b | X) using
        results of the forward-backward algorithm.

        Arguments:
            i: the element in the current sequence X
            b: the state to transition to
            a: the state to transition from
            alphas: results from forward algorithm corresponding to current
            sequence X
            betas: results from backward algorithm corresponding to current
            sequence X
            A: current transition matrix
            O: current observation matrix
            X: observed sequence
        '''

        num = alphas[i][a] * A[a][b] * O[b][X[i]] * betas[i+1][b]

        if num == 0:
            return 0

        denom = 0
        for k in range(len(alphas[0])):
            for w in range(len(betas[0])):
                denom += alphas[i][k] * A[k][w] * O[w][X[i]] * betas[i+1][w]

        return num / denom

    def marginal_2(self, z, i, alphas, betas):
        '''
        Helper function that computes P(y^i = z) using results of the
        forward-backward algorithm.

        Arguments:
            i: the element in the current sequence
            z: the state (mood that Ron is in)
            alphas: results from forward algorithm corresponding to current
            sequence
            betas: results from backward algorithm corresponding to current
            sequence
        '''

        num = alphas[i][z] * betas[i][z]

        if num == 0:
            return 0

        denom = np.dot(alphas[i], betas[i])

        return num / denom


    def generate_emission(self, M):
        '''
        Generates an emission of length M, assuming that the starting state
        is chosen uniformly at random.

        Arguments:
            M:          Length of the emission to generate.

        Returns:
            emission:   The randomly generated emission as a list.

            states:     The randomly generated states as a list.
        '''

        emission = []
        states = []

        # Problem 2F

        # randomly select initial state
        curr = random.randint(0, self.L - 1)
        states.append(curr)

        for i in range(M - 1):
            # get emission given current state
            e = np.random.choice([j for j in range(self.D)], p = self.O[curr])
            emission.append(e)

            # get next state
            next = np.random.choice([j for j in range(self.L)], p = self.A[curr])
            states.append(next)
            curr = states[-1]

        # get final emission
        e = np.random.choice([j for j in range(self.D)], p = self.O[curr])
        emission.append(e)

        return emission, states


    def probability_alphas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the forward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        # Calculate alpha vectors.
        alphas = self.forward(x)

        # alpha_j(M) gives the probability that the state sequence ends
        # in j. Summing this value over all possible states j gives the
        # total probability of x paired with any state sequence, i.e.
        # the probability of x.
        prob = sum(alphas[-1])
        return prob


    def probability_betas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the backward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        betas = self.backward(x)

        # beta_j(1) gives the probability that the state sequence starts
        # with j. Summing this, multiplied by the starting transition
        # probability and the observation probability, over all states
        # gives the total probability of x paired with any state
        # sequence, i.e. the probability of x.
        prob = sum([betas[1][j] * self.A_start[j] * self.O[j][x[0]] \
                    for j in range(self.L)])

        return prob


def supervised_HMM(X, Y):
    '''
    Helper function to train a supervised HMM. The function determines the
    number of unique states and observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for supervised learning.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers
                    ranging from 0 to D - 1. In other words, a list of lists.

        Y:          A dataset consisting of state sequences in the form
                    of lists of variable length, consisting of integers
                    ranging from 0 to L - 1. In other words, a list of lists.
                    Note that the elements in X line up with those in Y.
    '''
    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Make a set of states.
    states = set()
    for y in Y:
        states |= set(y)

    # Compute L and D.
    L = len(states)
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm

    # Randomly initialize and normalize matrix O.
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with labeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.supervised_learning(X, Y)

    return HMM

def unsupervised_HMM(X, n_states, N_iters):
    '''
    Helper function to train an unsupervised HMM. The function determines the
    number of unique observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for unsupervised learing.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers
                    ranging from 0 to D - 1. In other words, a list of lists.

        n_states:   Number of hidden states to use in training.

        N_iters:    The number of iterations to train on.
    '''

    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Compute L and D.
    L = n_states
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    # Randomly initialize and normalize matrices A and O.
    random.seed(2020)
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm

    # Randomly initialize and normalize matrix O.
    random.seed(155)
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with unlabeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.unsupervised_learning(X, N_iters)

    return HMM
