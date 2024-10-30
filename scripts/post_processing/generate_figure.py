import numpy as np
import matplotlib.pyplot as plt

def make_subplot(ax, data, title, x_lim, colors, x_label = True):

    truncate = 0
    labels = data.keys()

    for n, label in enumerate(labels):

        x = data[label] # [n_repeats, time, 2]
        running = x[:,:,0] / x[:,:,1] # [n_repeats, time]
        running_mean = running.mean(axis = 0)
        running_std = running.std(axis = 0)

        ax.plot(np.arange(len(running_mean))[truncate:], running_mean[truncate:], label = label, c = colors[n])
        ax.fill_between(np.arange(len(running_mean))[truncate:],
                         running_mean[truncate:]-running_std[truncate:],
                         running_mean[truncate:]+running_std[truncate:], alpha=0.2, color = colors[n])


        if n == 0:
            y_min = running_mean.min()
            y_max = running_mean[~np.isnan(running_mean)].max()

        if y_min > running_mean.min():
            y_min = running_mean.min()

        if y_max < running_mean[~np.isnan(running_mean)].max():
            y_max = running_mean[~np.isnan(running_mean)].max()


    y_min = 40
    y_max = 200

    ax.set_ylim([y_min, y_max])
    ax.set_xlim(x_lim)
    if x_label:
        ax.set_xlabel('Weeks')
    ax.set_title(title)

def main(r = 2, d = 0.43):

    Ks = [1, 2, 3, 4]
    nrows, ncols = 2, 2
    fig, axs = plt.subplots(nrows, ncols, figsize=(10, 8))
    x_lim = [0, 160]

    nums = ['A. ', 'B. ', 'C. ', 'D. ']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 'k']

    for ax_idx in range(nrows * ncols):
        i = ax_idx // ncols  # Integer division to find the row index
        j = ax_idx % ncols  # Modulo operation to find the column index

        K = Ks[ax_idx]

        read_dir = 'data/output/simulation/'
        data = np.load(f'{read_dir}simulated_data_K{K}_r{r}_d{int(d * 100)}.npz')

        title = nums[ax_idx] + str(K)
        if K > 1:
            title += ' mobile units'
        else:
            title += ' mobile unit'

        if nrows * ncols == 1:

            make_subplot(axs, data, title, x_lim, colors)
            axs.set_title('')
            axs.set_xlabel('Weeks', fontsize=12)
            handles, _ = axs.get_legend_handles_labels()

        else:
            if i == 1:
                make_subplot(axs[i, j], data, title, x_lim, colors)
            else:
                make_subplot(axs[i, j], data, title, x_lim, colors, x_label=False)

            handles, labels = [], []
            for ax in axs.flat:
                for handle, label in zip(*ax.get_legend_handles_labels()):
                    if label not in labels:
                        handles.append(handle)
                        labels.append(label)

    labels = ['random', 'historic rates', 'exp3', 'LinUCB']

    fig.legend(handles, labels, loc='lower center', ncol=len(labels))
    fig.text(0.0, 0.5, '# needed to screen to identify one person with TB', va='center', rotation='vertical',
             fontsize=16)

    # Adjust layout
    plt.subplots_adjust(left=0.07, right=0.95, top=0.95, bottom=0.1)

    save_name = 'main_result.png'
    save_dir = 'scripts/post_processing/'

    plt.savefig(save_dir + save_name, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()

