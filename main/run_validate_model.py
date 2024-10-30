import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from src.simulator import *
from src.policies import *
from src.Environment import *
from src.tools import *

# Function to run the model simulation (customized)
def run_validation(schedules, n_trials=10,
                   transitions=None, initial_states=None, N=None, weights=None, features=None, features_name=None):
    """
    Run the simulation for the given parameters.
    """
    # Initialize environment components
    residents = Residents(transitions=transitions, initial_states=initial_states)
    screening_sites = ScreenSites(N=N, weights=weights, residents=residents)
    environment = Envrionment(features_name, features)
    model = customize(screening_sites, environment)

    # Run customized policy
    rewards = []
    for i in range(n_trials):
        print(f'Trial {i + 1}/{n_trials}')
        screening_sites.reset()
        environment.reset()
        model.reset()

        reward = list()
        for action in schedules:
            model.step(action)
            reward.append(np.array([model.cum_total_screened, model.cum_total_diagnosed]))
        rewards.append(np.array(reward))
    return np.array(rewards)

# Function to calculate coverage statistics
def coverage_check(simulated_data, observed_data):

    simulated_diff = np.diff(simulated_data[:, :, 1]) / np.diff(simulated_data[:, :, 0])
    observed_diff = (np.diff(observed_data[:, :, 1]) / np.diff(observed_data[:, :, 0])).reshape(-1)

    simulated_mean = simulated_diff.mean(axis=0)
    simulated_std = simulated_diff.std(axis=0)
    coverage_1std = (simulated_mean - simulated_std <= observed_diff) & (simulated_mean + simulated_std >= observed_diff)
    coverage_2std = (simulated_mean - 2 * simulated_std <= observed_diff) & (simulated_mean + 2 * simulated_std >= observed_diff)

    return simulated_mean, simulated_std, coverage_1std.mean(), coverage_2std.mean()

# Plotting function for validation
def plot_validation(ax, simulated_data, observed_data, x_lim, y_lim, colors, r, d):
    simulated_mean, simulated_std, per_1std, per_2std = coverage_check(simulated_data, observed_data)

    # Plot simulated raw data
    for running in np.diff(simulated_data[:, :, 1]) / np.diff(simulated_data[:, :, 0]):
        ax.scatter(np.arange(len(running)), running, label='simulated (raw)', color=colors[0])

    # Plot observed data
    observed_diff = (np.diff(observed_data[:, :, 1]) / np.diff(observed_data[:, :, 0])).reshape(-1)
    ax.scatter(np.arange(len(observed_diff)), observed_diff, label='observed', color=colors[1])

    # Plot mean of simulated data with error bars (1 standard deviation)
    ax.errorbar(np.arange(len(simulated_mean)), simulated_mean, yerr=simulated_std, fmt='none', color='k', ms=2)
    ax.scatter(np.arange(len(simulated_mean)), simulated_mean, s=20, marker="D", color="k", label='simulated (mean)')

    # Customize plot appearance
    ax.set_ylim(y_lim)
    ax.set_xlim(x_lim)
    ax.set_xlabel('Weeks', fontsize=12)
    ax.set_xticks(range(x_lim[0], x_lim[1], 10))
    ax.set_xticklabels([str(i + 1) for i in range(x_lim[0], x_lim[1], 10)])
    ax.set_title(f'Model Parameters: r={r}, d={d}, Coverage: {per_1std*100:.1f}% (1std) / {per_2std*100:.1f}% (2std)', fontsize=10)

# Main function to run the simulation and create plots
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run model validation with optional r and d parameters.")
    parser.add_argument('--r', type=int, help='The value of r (default uses SGE_TASK_ID)', required=False)
    parser.add_argument('--d', type=float, help='The value of d (default uses SGE_TASK_ID)', required=False)
    args = parser.parse_args()

    # Get SGE_TASK_ID or fallback to test value
    task_id = int(os.getenv('SGE_TASK_ID', 1))

    # Define r and d values based on task_id if not provided as arguments
    r_values = [1, 2, 3, 4, 5]
    d_values = [0.23, 0.33, 0.43]

    if args.r is None or args.d is None:
        r_index = (task_id - 1) % len(r_values)
        d_index = (task_id - 1) // len(r_values)

        r = r_values[r_index]
        d = d_values[d_index]
    else:
        r = args.r
        d = args.d

    print(f'Using r={r} and d={d}')

    # Read parameters from saved pickle files
    params_file = f'data/parameters/parameters_r{r}_d{int(d*100)}.pkl'
    with open(params_file, 'rb') as f:
        parameters = pickle.load(f)

    transitions = parameters['transitions']
    initial_states = parameters['initial_states']
    A = parameters['A_potential']
    pi = parameters['travel_pi']

    # Load features
    features = pd.read_csv('data/input/features.csv').iloc[:, 1:].to_numpy()
    features_name = pd.read_csv('data/input/features.csv').columns[1:]

    # Load observed data
    observed_data = load_json_with_arrays('data/input/observations.json')

    # Load schedules
    with open('data/input/schedules.pkl', 'rb') as f:
        schedules = pickle.load(f)

    # Run validation
    simulated_data = run_validation(schedules, n_trials=1000, transitions=transitions, initial_states=initial_states,
                                    N=A, weights=pi, features=features, features_name=features_name)


    # Save
    output_dir = 'data/output/validation/'
    ensure_dir_exists(output_dir)
    np.savez_compressed(f'{output_dir}simulated_data_r{r}_d{int(d*100)}.npz', matrix = simulated_data)

    # Plot results
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_validation(ax, simulated_data, observed_data, x_lim=[0, 54], y_lim=[0, 0.02], colors=['#1f77b4', 'red'], r=r, d=d)

    # Save the figure
    figure_dir = 'results/validation/'
    ensure_dir_exists(figure_dir)
    plt.savefig(f'{figure_dir}validation_r{r}_d{int(d*100)}.png', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()
