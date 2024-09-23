import numpy as np

class Envrionment():

    def __init__(self, feature_names, context_static):
        """

        @param feature_names: [n_dims, ]
        @param context_static: [n_arms, n_dims]

        """

        self.feature_names = feature_names
        self.context_static = context_static
        self.context_dynamic = None
        self.n_arms = context_static.shape[0]
        self.observable_arm = np.zeros(self.n_arms)  # one hot pulled arm
        self.historical_pulled = list()
        self.t = 0 # current time

        # set two dynamical feature [lag, previous yield]
        self.n_dims = context_static.shape[1] + 2

        self.latest_visit = np.zeros(self.n_arms)  # [n_arms,] 0 means never actively visited
        self.prev_yields = np.zeros(self.n_arms)  # [n_arms,] pos / total

    def reset(self):
        self.t = 0
        self.latest_visit = np.zeros(self.n_arms)
        self.prev_yields = np.zeros(self.n_arms)
        self.historical_pulled = list()



    def compute_dynamic_feature(self):
        """

        @:param tau:   current time - last time being pulled

        """
        tau = self.t - self.latest_visit

        return np.vstack((tau, self.prev_yields)).T


    def update(self, sim):
        """

        @param n_arm:      observable arm
        @param context_dynamic: [n_dims] only pulled arm
        @return:
        """
        if self.t > 0:
            self.observable_arm = sim.status
            arm_indx = self.observable_arm.astype(bool)

            self.latest_visit[arm_indx] = self.t
            self.historical_pulled.append(np.where(arm_indx)[0])
            observable_summary = sim.get_observable() # [n_arm, 2]

            # replace None:
            self.prev_yields[arm_indx] = observable_summary[:, 0]/observable_summary.sum(axis = 1) #[n_arms, ]
            self.prev_yields = np.nan_to_num(self.prev_yields, copy=True, nan=0.0, posinf=None, neginf=None)

        self.context_dynamic = self.compute_dynamic_feature()
        self.t += 1

    def get_context(self, intercept = True):
        """
        full feature regardless pull or not
        :return:
        """

        context = np.hstack([self.context_static, self.context_dynamic])

        if not intercept:
            indx = np.where(self.feature_names == 'intercept')[0]
            context = np.delete(context, indx, axis = 1)

        return context


