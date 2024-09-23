import numpy as np
import torch
import torch.nn as nn
from src.tools import load_json_with_arrays, ensure_dir_exists
import pickle

def reparameterize(Ts, r=1, option=1):

    """
    reparameterize Ts into transition matrix

    Parameter:
        Ts (torch.tensor): Ts[0] for trainable gamma, Ts[1] for fixed d
        r (int): ratio of s+ to s-, {1,2,3,4,5}
        option (int): same as a_t, {0,1}

    Return:
        transition (torch.tensor): T_{a_t} ;[5,5]
    """
    scale2 = 0.01/4
    scale1 = scale2 * r

    gamma = (1 - torch.exp(-Ts[0] * Ts[0])) * (1-scale2)
    d = Ts[1]

    zero = torch.zeros_like(gamma)
    one = torch.ones_like(gamma)

    if option == 1:
        s1 = scale1 * one
        s2 = scale2 * one
        row0 = torch.stack([(1 - s1) * (1-d), zero, s1, zero, d * (1-s1)])
    else:
        s1 = scale1 * zero
        s2 = scale2 * zero
        row0 = torch.stack([1 - s1 - d, zero, s1, zero, d])

    row1 = torch.stack([gamma, 1 - gamma - s2, zero, s2, zero])
    row2 = torch.stack([zero, zero, zero, zero, one])
    row3 = torch.stack([zero, one, zero, zero, zero])
    row4 = torch.stack([zero, zero, zero, zero, one])
    transition = torch.stack([row0, row1, row2, row3, row4]).squeeze(-1)

    return transition

def get_loss(gap, delta, r, Ts):
    """
    compute loss

    Parameters:
        gap (np.ndarray): screening time differences. t_k - t_{k-1}; [# observation -1, ]
        delta (np.ndarray): prc. diagnosed, prc not diagnosed; [# observation -1, 2]
        r (int): ratio of s+ to s-, {1,2,3,4,5}
        Ts (torch.tensor): Ts[0] for trainable gamma, Ts[1] for fixed d; Before Reprameterization

    Return:
        L: loss function with traceable gradient wrt Ts
    """


    loss = nn.MSELoss(reduction='mean')
    transitions = torch.stack([reparameterize(Ts, r, option=0), reparameterize(Ts, r, option=1)])
    L = 0

    for i, (k, p0) in enumerate(zip(gap, delta[:-1])):
        p1 = delta[i + 1]

        p1 = torch.FloatTensor(p1).reshape((1, -1))
        p0 = torch.FloatTensor(p0).reshape((1, -1))

        if i == 0:
            prev = torch.zeros((1, 5))

            # adjust initial start based on r
            prev[0, :2] = p0
            prev[0, 1] = prev[0,1] * r
            prev = torch.matmul(prev, transitions[1])

        if k > 1:
            current = torch.matmul(torch.matmul(prev, torch.matrix_power(transitions[0], k - 1)), transitions[1])
        else:
            current = torch.matmul(prev, transitions[1])

        # observable prediction:
        summary = current[0, 2:4] / current[0, 2:4].sum()
        L += loss(summary, p1.reshape(-1))
        prev = current

    L = L / (i + 1)
    return L

def fit_transition(time, gap, delta, r, Ts, eps=1e-5):
    """
    Fit Markov Model through adam

    Parameters:
        time (int): maximum iteration
        gap (np.ndarray): screening time differences. t_k - t_{k-1}; [# observation -1, ]
        delta (np.ndarray): prc. diagnosed, prc not diagnosed; [# observation -1, 2]
        r (int): ratio of s+ to s-, {1,2,3,4,5}
        Ts (torch.tensor): Ts[0] for trainable gamma, Ts[1] for fixed d; Before Reprameterization
        eps (np.float): gradient norm threshold for early termination

    Return:
        transitions (np.ndarray): estimated transition matrix; [2,5,5]
        flag (int): exit condition. 1: max iter hit. 0: early termination evoked
        grads_norm (np.float): gradient norm when exit
    """

    loss = []
    flag = 1  # max iter hit
    optim = torch.optim.Adam(Ts, lr=1e-3)

    for t in range(time):
        # print('t: ', t)
        optim.zero_grad()

        L = get_loss(gap, delta, r, Ts)
        L.backward()
        loss.append(L.item())

        optim.step()

        grads = []
        grads_norm = 0
        for T in Ts:
            g = T.grad
            if g is not None:
                grads.append(g)
                grads_norm += torch.norm(g)
                # print(g.item())

        if grads_norm <= eps:
            flag = 0
            print('early stopping due to small gradient at iteration', t + 1)
            break


    transitions = np.zeros((2, 5, 5))
    transition = reparameterize(Ts, r, option=0).detach().numpy()
    transitions[0] = transition / transition.sum(axis=1, keepdims=True)
    transition = reparameterize(Ts, r, option=1).detach().numpy()
    transitions[1] = transition / transition.sum(axis=1, keepdims=True)

    return transitions, flag, grads_norm.item()

def execute_model_fitting(r = 2, d = 0.43, time = 1000):

    """
    execute model fitting by fixing predefined paraters s+, d and input data in site_info

    Parameters:
        r (int): ratio of s+ to s-, {1,2,3,4,5}
        d (float): {0.33, 0.43}
        time (int): max iter time

    Return:
        transitions (np.ndarray): [# sites, 2, n_state, n_state]
        records (dict): sites exits due to max iter hits, and grad norm; {'site': grad_norm}
        initial_states (dict): {site ID, initial distribution}
    """

    # site_info (dict): {'residential ID', 'gap', 'delta', 'pos' },
    site_info = load_json_with_arrays("data/input/site_info.json")

    sites = list(site_info.keys())
    transitions = np.array([len(sites),2,5,5])
    records = dict()
    initial_states = dict()
    eps = (600 / 100000) / r

    for i, n in enumerate(sites):

        print('eval site: ', n)

        gap = site_info[n]['gap']
        delta = site_info[n]['delta']

        # check special condition:
        if site_info[n]['pos'].sum():  # no TB detected
            scale2 = 0.01/4
            scale1 = scale2 * r
            est_T = np.array([
                [
                    [.0, 1.0, .0, .0, .0],
                    [.0, 1.0, .0, .0, .0],
                    [.0, 1.0, .0, .0, .0],
                    [.0, 1.0, .0, .0, .0],
                    [.0, 1.0, .0, .0, .0]
                ],
                [
                    [.0, 1.0 - scale1, scale1, .0, .0],
                    [.0, 1.0 - scale2, .0, scale2, .0],
                    [.0, 1.0, .0, .0, .0],
                    [.0, 1.0, .0, .0, .0],
                    [.0, 1.0, .0, .0, .0],
                ]
            ])
            print('no TB occured')
            flag = 0
        else:

            # initialize T: (before reparameterization)
            gamma = torch.tensor([0.04], dtype=torch.float32, requires_grad = True)
            d = torch.tensor([d], dtype=torch.float32, requires_grad = False)

            est_T, flag, grad_norm = fit_transition(time, gap, delta, r,[gamma, d])

        if flag == 1:
             print(f"Warning: site {n} max iter hit")
             records[n] = grad_norm.item()

        transitions[i] = est_T
        initial_states[n] = np.array([eps,1-eps,0,0,0])

    return transitions, records, initial_states

def get_model_parameter(r = 2, d = 0.43):

    print(f"Model fitting: r = {r}, d = {d}")
    transitions, _, initial_states = execute_model_fitting(r = r, d = d)
    A = load_json_with_arrays("data/input/A_potential.json")
    pi = load_json_with_arrays("data/input/travel_pi.json")

    parameters = {'transitions': transitions,
                  'initial_states': initial_states,
                  'A_potential': A,
                  'travel_pi': pi
                  }

    # save for later use
    save_name = 'parameters' + '_r' + str(r) + '_d' + str(int(d*100)) + '.pkl'
    save_path = 'data/parameters/'

    ensure_dir_exists(save_path)

    with open(save_path+save_name, 'wb') as f:
        pickle.dump(parameters, f)

    print(f"Saved Model Parameters: r = {r}, d = {d}")


