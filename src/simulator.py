import numpy as np
from copy import copy

# remove redundant info from preceeding scripts
# Fix A(N) scaling externally


class MarkovChain:
    """
    A markov chain parameterized by a pair of transition matrices
    """

    def __init__(self, ID = None, transitions = None, initial_states = None ):

        """
        initialization

        Parameters:
            ID (str): A 4-digit string representing residential ID.
            transitions (np.ndarray): Transition matrices p[a, s, s']; [2, n_state, n_state]. 
            initial_states (np.ndarray): initial distribution across the states [n_state, ]

        """

        # delete n_pop, do we need ID?

        self.ID = ID

        if transitions is None:
            raise error("empty transitions matrices")
        self.transitions = transitions

        self.n_states = self.transitions.shape[-1]


        if initial_states is None:
            # everyone starts from (2)
            initial_states = np.zeros(self.n_states)
            initial_states[1] = 1
            self.initial_states = initial_states
        else:
            self.initial_states = initial_states

        self.current_states = copy(self.initial_states)


    def step(self, a):
        """
        one step of the model with transitions[a]

        Parameters:
            a (int): either 0 or 1
        """

        if a not in [0, 1]:
            raise ValueError("Parameter 'a' must be either 0 or 1.")
    
        current_states = self.current_states
        pos_state = current_states.reshape((1,-1)) @ self.transitions[a]
        self.current_states = pos_state.reshape(-1)


    def get_current_states(self):
        return self.current_states

    def reset(self):
        self.current_states = copy(self.initial_states)

class Residents(MarkovChain):
    """
    collection of MarkovChains, represents residents
    """

    def __init__(self, transitions = None, initial_states = None, burn_in = 5):
        """
        initialization

        Parameters:
            transitions (np.ndarray): transition matrices for M neighbourhoods (zones); [M, 2, n_state, n_state]
            initial_states (dict):  corresponidng {'ID'  [initial]}
                                    'ID' (str): 4 digits string
                                    'Initial (np.ndarray)': [n_state, ]
            burn_in (int): burn-in period for MarkovChain
        """

        if transitions is None:
            raise error("not implemented")
        self.transitions = transitions
        self.n_zone, _, self.n_states = transitions.shape[:-1]

        self.zones = list()
        if initial_states is None:
            self.initial_states = [None] * self.n_zone
            for transition, i in zip(self.transitions, self.initial_states):
                self.zones.append(
                    MarkovChain(ID=i, transitions=transition, initial_states=i))
        else:
            self.initial_states = initial_states
            for transition, ID in zip(self.transitions, self.initial_states):
                self.zones.append(
                    MarkovChain(ID=ID, transitions=transition, initial_states=self.initial_states[ID]))
                

        # apply burin-in:
        self.burn_in = burn_in
        for t in np.arange(self.burn_in):
            for zone in self.zones:
                zone.step(0)

        # house keeping
        self.current_states = np.zeros((self.n_zone, self.n_states))
        for i, zone in enumerate(self.zones):
            self.current_states[i] = zone.get_current_states()


    def step(self, i, N):
        self.zones[i].step(N)
        self.current_states[i] = self.zones[i].get_current_states()

    def reset(self):
        for zone in self.zones:
            zone.reset()

        for t in np.arange(self.burn_in):
            for zone in self.zones:
                zone.step(0)

        for i, zone in enumerate(self.zones):
            self.current_states[i] = zone.get_current_states()

class ScreenSites:
    """
    A collection of screening sites, simulate TB incidence from Residents. 
    """

    def __init__(self, N = None, weights = None, residents = None):
        """
        initialization

        Parameters:
            N (np.ndarray): N[n] # potential attendee at site n; [n_sites, ]
            weights (np.ndarray): weights[n] attendance distribution at site n; [n_sites, n_zone]
            residents (class instance): An instance of Residents with n_zone # of residential area
        """

        self.N = N
        self.weights = weights
        self.residents = residents

        self.n_sites, self.n_zone  = weights.shape
        self.summary = np.zeros((self.n_sites,2)) # per arm summary
        self.status = np.zeros(self.n_sites) # van placement


    def step(self, actions):
        '''
            @param actions: one hot vector [n_sites, ] (can be multiple ones)
            @return: summary
        '''

        def stochastic_sample(temp, T):
            sampled_vals = np.zeros_like(T)
            for i, n in enumerate(temp):
                if n>0:
                    sampled_indices = np.random.choice(np.arange(T.shape[1]), size=int(n), p=T[i] / T[i].sum(), replace=True)
                    sampled_vals[i, :len(np.bincount(sampled_indices))] += np.bincount(sampled_indices)
            return sampled_vals

        self.status = actions

        # sample attendee
        self.summary = np.zeros((self.n_screen, 2))  # for bandit algo
        screen_sites = np.where(self.status == 1)[0]  # place to go this week

        # loop over each screening sites, sample attendance:
        summary = np.zeros((self.n_screen, self.n_residential))

        # determine attandee residential origin:
        for n in screen_sites:
            temp = np.random.choice(self.n_residential, int(self.N[n]) , replace=True, p=self.weights[n])
            temp_count = np.zeros(self.n_residential)

            temp_count[:len(np.bincount(temp))] += np.bincount(temp)
            summary[n] += temp_count

        # determine detection outcome
        N = summary[screen_sites, :].sum(axis = 0).astype('int') # effective attandance
        for i in np.arange(self.n_residential):
            if N >= 1:
                self.residents.step(i, 1)
            else:
                self.residents.step(i, 0)
        for n in screen_sites:
            observable = stochastic_sample(summary[n], self.residents.current_states)
            observable = observable.sum(axis=0)[2:4]
            self.summary[n] = observable

    def get_observable(self):
        return self.summary[self.status.astype(bool)].reshape((-1,2))

    def reset(self):
        self.summary = np.zeros((self.n_screen, 2))
        self.status = np.zeros(self.n_screen)
        self.residents.reset()
