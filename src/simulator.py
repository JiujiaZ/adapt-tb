import numpy as np
from copy import copy

class MarkovChain:
    """
    A markov chain parameterized by a pair of transition matrices

    Attributes:
        ID (str): A 4-digit string representing the ID of the residential area.
        transitions (np.ndarray): Transition matrices p[a, s, s'] with shape (2, n_state, n_state). 
        initial_states (np.ndarray): initial distribution across the states.
        current_states (np.ndarray): current state distribution.

    Methods:
        step(N): Simulate one step of the model, given N individuals attending.
        get_current_states(): Return the current state distribution of the population.
        get_current_summary(): Return the total population across all states (optional, depends on use case).
        reset(): Reset the model to the initial state distribution.
    """

    def __init__(self, ID = None, transitions = None, initial_states = None ):

        """
        initialization
            @param ID: 4 digits string, residential ID
            @param transitions:     transition matrix p[s, s'] (6*6)
            @param initial_states:  population
        """

        # delete n_pop, do we need ID?

        self.ID = ID

        if transitions is None:
            raise error("empty transitions matrices")
            # implement something robust here
        self.transitions = transitions

        self.n_states = self.transitions.shape[-1]


        if initial_states is None:
            # everyone starts from (1)+
            initial_states = np.zeros(self.n_states)
            initial_states[1] = 1
            self.initial_states = initial_states
        else:
            self.initial_states = initial_states

        self.current_states = copy(self.initial_states)


    def step(self, N):
        '''
        stochastically simulate the state space model for one step,
        N indivial attends
        '''

        current_states = self.current_states

        prc = min(N/self.n_population, 1) # attendance percentage

        pos_state = prc * current_states.reshape((1,-1)) @ self.transitions[1] + (1-prc) * current_states.reshape((1,-1)) @ self.transitions[0]

        # update model: + rescale first 4 states
        self.current_states = pos_state.reshape(-1)


    def get_current_states(self):
        return self.current_states

    # need?
    def get_current_summary(self):
        return self.current_states.sum(axis=0)

    def reset(self):
        self.current_states = copy(self.initial_states)