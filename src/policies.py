import numpy as np


def logsumexp_trick(x):
    """
    For exp(x_i) / sum_i exp(x_i) avoid over flow
    """

    c = x.max()
    y = c + np.log(np.sum(np.exp(x - c)))

    return np.exp(x - y)

class random_GEOTB():

    # random + historical high GEOTB
    # p = 0, fully random

    def __init__(self, rb, env, K = 1, p = 0):


        self.rb = rb
        self.K = K
        self.n_arms = rb.n_screen

        # Features from environment:
        self.env = env

        self.GEOTB_indx = np.where(self.env.feature_names == 'GEOTB_rate_total') # index from feature
        self.p = p # probability of relying on GEOTB

        # output:
        self.cum_total_screened = 0
        self.cum_total_diagnosed = 0



    def step(self):

        # update environment
        self.env.update(self.rb)
        X = self.env.get_context()

        GEOTB_policy = np.random.binomial(n = 1, p = self.p)

        if GEOTB_policy == 1:
            GEOTB_feature = X[:,self.GEOTB_indx].reshape(-1) + 1e-5 #[n_arm]

            p = GEOTB_feature/GEOTB_feature.sum()
            indx = np.random.choice(self.n_arms, size=self.K, replace=False, p=p)
        else:
            indx = np.random.choice(self.n_arms, size=self.K, replace=False)

        actions = np.zeros(self.n_arms, dtype=int)
        actions[indx] = 1

        self.rb.step(actions=actions)

        observation = self.rb.get_observable()  # [n_observe, 2]
        self.cum_total_screened += observation.sum()
        self.cum_total_diagnosed += observation[:, 0].sum()

    def reset(self):
        self.env.reset()
        self.rb.reset()

        self.cum_total_screened = 0
        self.cum_total_diagnosed = 0



class customize():


    def __init__(self, rb, env):


        self.rb = rb
        self.n_arms = rb.n_screen

        # Features from environment:
        self.env = env

        # output:
        self.cum_total_screened = 0
        self.cum_total_diagnosed = 0

    def step(self, schedule):

        # schedule, one-hot like: send num bans to location

        # update environment
        self.env.update(self.rb)
        # based on schedualling
        actions = schedule > 0

        self.rb.step(actions=actions)

        observation = self.rb.get_observable()  # [n_observe, 2]
        # adjust for multiple vans
        observation = observation * schedule[actions].reshape(-1, 1)
        self.cum_total_screened += observation.sum()
        self.cum_total_diagnosed += observation[:, 0].sum()

    def reset(self):
        self.env.reset()
        self.rb.reset()

        self.cum_total_screened = 0
        self.cum_total_diagnosed = 0

        # for check sub optimality
        self.temp_cum_total_screened = 0
        self.temp_cum_total_diagnosed = 0
        self.temp_pulls = list()

class exp3():

    def __init__(self, rb, env, T, K = 1, eta = 0.01):

        # decay historical reward
        self.rb = rb
        self.env = env
        self.K = K
        self.n_arms = rb.n_screen
        self.p = np.ones( self.n_arms ) / self.n_arms

        # this is for internal update not the actual reward
        self.cum_reward = np.zeros(self.n_arms)
        self.his_reward = np.zeros((self.n_arms, T))

        # theoretical lr: L_infty norm
        self.L = 100
        self.eta = np.sqrt(np.log(self.n_arms) / (T * self.n_arms * self.L**2))


        # for output
        self.cum_total_screened = 0
        self.cum_total_diagnosed = 0

    def step(self):
        self.env.update(self.rb)

        indx = np.random.choice(self.n_arms, size=self.K, replace=False, p=self.p)
        actions = np.zeros(self.n_arms, dtype=int)
        actions[indx] = 1

        self.rb.step(actions=actions)


        observation = self.rb.get_observable()
        # for output
        self.cum_total_screened += observation.sum()
        self.cum_total_diagnosed += observation[:, 0].sum()

        # for model updateL
        indx = observation[:, 0] == 0
        self.reward = np.zeros(len(observation.sum(axis=1)))
        self.reward[indx] = - observation.sum(axis=1)[indx] / (0.1 + observation[:, 0][indx])
        self.reward[~indx] = - observation.sum(axis=1)[~indx] / observation[:, 0][~indx]


        # update: (avoiding messing up order)
        # get pulled arm
        indx = np.where(self.rb.status)[0]
        # adjusted reward
        R_hat = np.zeros(self.n_arms)
        # observed_states = self.rb.current_states[indx].argmax(axis=-1)
        R_hat[indx] = self.reward / self.p[indx]

        # exp3 update:
        self.cum_reward += R_hat
        # avoid over flow use log-sum-exp trick
        self.p = logsumexp_trick(self.eta * self.cum_reward)
        # mix with exploration
        self.p = (1 - self.eta) * self.p + self.eta / self.n_arms


    def reset(self,reset_rb = True):
        if reset_rb :
            self.rb.reset()

        self.p = np.ones(self.n_arms) / self.n_arms
        self.cum_reward = np.zeros(self.n_arms)
        self.his_reward = np.zeros_like(self.his_reward)

        self.cum_total_screened = 0
        self.cum_total_diagnosed = 0

class LinUCB():

    def __init__(self, rb, env, T,  alpha = 1, lam = 1, K = 1, reg = None):

        """
        :param rb:
        :param env: Environment
        :param n_dims:
        :param K:
        """


        self.rb = rb
        self.K = K
        self.n_arms = rb.n_screen
        # alpha: around high probability delta = 0.1
        self.alpha = alpha
        self.lam = lam

        self.T = T

        # Features from environment:
        self.env = env
        self.n_dims = env.n_dims


        # estimation parameters N[a, d, d]
        # self.A = np.tile(np.zeros((self.n_dims, self.n_dims)), (self.n_arms, 1, 1))
        self.A = np.tile(np.eye(self.n_dims) * self.lam, (self.n_arms, 1, 1))
        self.b = np.zeros((self.n_arms, self.n_dims, 1))
        self.theta = None

        self.t = 1
        self.reward = 0
        # regularization center
        if reg is None:
            self.theta_star = np.zeros((self.n_arms, self.n_dims, 1))
        else:
            self.theta_star = reg


        self.cum_total_screened = 0
        self.cum_total_diagnosed = 0


    def step(self):
        self.env.update(self.rb)
        X = self.env.get_context()

        theta = np.zeros((self.n_arms, self.n_dims, 1))
        p = np.zeros(self.n_arms)
        for n in range(self.n_arms):
            # A_inv = np.linalg.pinv(self.A[n] + self.lam/self.t * np.eye(self.n_dims))
            A_inv = np.linalg.pinv(self.A[n])

            decay = (self.T - self.t) / self.T
            # decay = 1/np.sqrt(self.t)
            # decay = 1
            theta[n] = A_inv @ (self.b[n] + self.lam * decay * self.theta_star[n])
            p[n] = (theta[n].T @ X[n] + self.alpha * np.sqrt(X[n].T @ A_inv @ X[n])).item()

            self.theta = theta
            # select arms
            indx = np.argsort(p)[-self.K:]

        # set action
        actions = np.zeros(self.n_arms, dtype=int)
        actions[indx] = 1

        self.rb.step(actions=actions)
        observation = self.rb.get_observable()  # [n_observe, 2]
        # for output
        self.cum_total_screened += observation.sum()
        self.cum_total_diagnosed += observation[:, 0].sum()

        indx = observation[:, 0] == 0
        self.reward = np.zeros(len(observation.sum(axis=1)))
        self.reward[indx] = - observation.sum(axis=1)[indx] / (0.1 + observation[:, 0][indx])
        self.reward[~indx] = - observation.sum(axis=1)[~indx] / observation[:, 0][~indx]


        # get pulled arm
        indx = np.where(self.rb.status)[0]
        # update: over each selected arm:
        for n, i in enumerate(indx):
            observe_feature = X[i].reshape((-1,1))

            # no decay update
            self.A[i] += observe_feature @ observe_feature.T
            self.b[i] += self.reward[n] * observe_feature

        self.t += 1


    def reset(self):

        self.env.reset()
        self.rb.reset()

        # self.A = np.tile(np.zeros((self.n_dims, self.n_dims)), (self.n_arms, 1, 1))
        self.A = np.tile(np.eye(self.n_dims) * self.lam, (self.n_arms, 1, 1))
        self.b = np.zeros((self.n_arms, self.n_dims, 1))
        self.theta = None

        self.reward = 0
        self.t = 1

        self.cum_total_screened = 0
        self.cum_total_diagnosed = 0

