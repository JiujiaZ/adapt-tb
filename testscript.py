import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from src.simulator import *
from src.policies import *
from src.Environment import *
from src.tools import *

def run_policy(K = 4, T = 102, n_trials = 10, method = 'random', site_reg = None, prior = None,
                   transitions=None, initial_states=None, N=None, weights=None, features=None, features_name=None):

    """
    Run specified policies

    @params K: number of vans between 1 to 95
    @params T: total number of weeks for simulation
    @params n_trails: number of repeats

    Return:
        [n_trials, T, 2], last din: total screen, total diag.

    """
    # valid method check
    if method not in ['random','historic TB rates','exp3', 'LinUCB', 'TS']:
        raise ValueError("method must be one of: 'random','historic TB rates','exp3', 'LinUCB', 'TS")

    # initialize bandit:
    residents = Residents(transitions=transitions, initial_states=initial_states)
    screening_sites = ScreenSites(N=N, weights=weights, residents=residents)
    environment = Envrionment(features_name, features)

    if method == 'exp3':
        model = exp3(screening_sites, environment, T, K=K)
    elif method == 'LinUCB':
        model = LinUCB(screening_sites, environment, T, alpha=1, lam=1, K=K, reg=site_reg)
    elif method == 'random':
        model = random_GEOTB(screening_sites, environment, K=K, p=0)
    elif method == 'historic TB rates':
        model = random_GEOTB(screening_sites, environment, K=K, p=1)
    elif method == 'TS':
        model = ThompsonSampling(screening_sites, environment, T, alpha=1, beta=1, K=K, prior = prior)

    # run specified policy:
    rewards = []
    for i in range(n_trials):
        print(f'Trial {i + 1}/{n_trials}')
        screening_sites.reset()
        environment.reset()
        model.reset()

        reward = list()
        for t in range(T):
            model.step()
            reward.append(np.array([model.cum_total_screened, model.cum_total_diagnosed]))
        rewards.append(np.array(reward))
    return np.array(rewards)

# run all multi-policies with same K, T, model params
def run_simulation(K = 4, T = 102, n_trials = 10, methods = ['random'], site_reg = None, prior = None,
                   transitions=None, initial_states=None, N=None, weights=None, features=None, features_name=None):

    simulated_data = dict()
    for method in methods:
        print(f'Running {method}')
        data = run_policy(K=K, T=T, n_trials=n_trials, method=method, site_reg=site_reg, prior = prior,
                          transitions=transitions, initial_states=initial_states, N=N, weights=weights, features=features, features_name=features_name)
        simulated_data[method] = data

    return simulated_data

# for a single K, params value, overlay all methods
def plot_simulation(ax, K, r, d, simulated_data):

    labels = simulated_data.keys()

    y_min = 0
    y_max = 0

    for n, label in enumerate(labels):

        data = simulated_data[label] # [n_repeats, time, 2]

        running = data[:,:,0] / data[:,:,1] # [n_repeats, time]
        running_mean = running.mean(axis = 0)
        running_std = running.std(axis = 0)


        # plot mean:
        ax.plot(np.arange(len(running_mean)), running_mean, label = label)
        # add std
        ax.fill_between(np.arange(len(running_mean)),
                         running_mean-running_std, running_mean+running_std, alpha=0.2)

        if n == 0:
            y_min = running_mean.min()
            indx = (~np.isnan(running_mean)) & (~np.isinf(running_mean))
            y_max = running_mean[indx].max()

        if y_min > running_mean.min():
            y_min = running_mean.min()

        indx = (~np.isnan(running_mean)) & (~np.isinf(running_mean))
        if y_max < running_mean[indx].max():
            y_max = running_mean[indx].max()

    y_min = max(0, y_min-10)
    y_max += 10
    ax.set_ylim([y_min, y_max])
    ax.set_ylabel('scan / +', fontsize=12)

    x_min = 0
    x_max = len(running_mean)-1
    ax.set_xticks(range(x_min, x_max, 20))
    ax.set_xticklabels([str(i + 1) for i in range(x_min, x_max, 20)])
    ax.set_xlabel('Weeks', fontsize=12)

    ax.legend()
    ax.set_title( f'Model Parameters: r={r}, d={d}, # Units: {K}', fontsize=12)



K = 2
r = 2
d = 0.43

# Read parameters from saved pickle files
params_file = f'data/parameters/parameters_r{r}_d{int(d * 100)}.pkl'
with open(params_file, 'rb') as f:
    parameters = pickle.load(f)

transitions = parameters['transitions']
initial_states = parameters['initial_states']
A = parameters['A_potential']
pi = parameters['travel_pi']

# Load features
features = pd.read_csv('data/input/features.csv').iloc[:, 1:].to_numpy()
features_name = pd.read_csv('data/input/features.csv').columns[1:]

# Load site_reg
params_file = f'data/input/site_reg.pkl'
with open(params_file, 'rb') as f:
    site_reg = pickle.load(f)

# load prior
params_file = f'data/input/prior.pkl'
with open(params_file, 'rb') as f:
    prior = pickle.load(f)


# specify methods and run simulation
methods = ['random','historic TB rates','exp3', 'LinUCB', 'TS']
simulated_data = run_simulation(K = K, T = 156, n_trials = 1000, methods = methods, site_reg = site_reg, prior = prior,
                                transitions=transitions, initial_states=initial_states, N=A, weights=pi, features=features, features_name=features_name)



# Plot results
fig, ax = plt.subplots(figsize=(6, 4))
plot_simulation(ax, K, r, d, simulated_data)
# Save the figure
figure_dir = 'results/simulation/'
ensure_dir_exists(figure_dir)
plt.savefig(f'{figure_dir}simulation_K{K}_r{r}_d{int(d * 100)}.png', bbox_inches='tight')
plt.close()
