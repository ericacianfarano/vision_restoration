import matplotlib.pyplot as plt

from imports import *
from helpers import *

def avg_onoff_response_groups(object):

    groups = np.unique([animal[3:-3] for animal in object.dat.keys()])
    screens = ['big', 'small']
    colours = ['black', 'blue', 'red']

    min_timepoints = np.array(
        list(chain.from_iterable([object.dat[animal][day][sub_file]['zscored_matrix_baseline'].shape[-1]
                                  for sub_file in object.dat[animal][day].keys() if ('chirp' in sub_file)]
                                 for animal in object.dat.keys()
                                 for day in object.dat[animal].keys()))).min()

    fig, ax = plt.subplots(len(screens), len(groups), figsize=(12, 8), sharex = True, sharey=True)

    for i, screen in enumerate(screens):
        # average response across recordings, repeats, across cells

        for i_group, group_name in enumerate(groups):
            group_dat = np.array(list(chain.from_iterable(
                [object.dat[animal][day][sub_file]['zscored_matrix_baseline'][1:,
                 object.dat[animal][day][sub_file]['thresholded_cells'] == 1, :min_timepoints].mean(axis=(0, 1))
                 for sub_file in object.dat[animal][day].keys() if (('chirp' in sub_file) and (screen in sub_file))]
                for animal in object.dat.keys() if (group_name in animal)
                for day in object.dat[animal].keys())))

            group_cells_responsive = np.array(list(chain.from_iterable(
                [np.round(100 * object.dat[animal][day][sub_file]['thresholded_cells'].sum() / len(
                    object.dat[animal][day][sub_file]['thresholded_cells']), 2)
                 for sub_file in object.dat[animal][day].keys() if (('chirp' in sub_file) and (screen in sub_file))]
                for animal in object.dat.keys() if (group_name in animal)
                for day in object.dat[animal].keys()))).mean()

            ax[i, i_group].plot(np.arange(min_timepoints),group_dat.T, alpha = 0.4, color=colours[i_group])
            ax[i, i_group].plot(np.arange(min_timepoints), np.nanmean(group_dat, axis=0), linewidth = 2, color='black')
            screen_label = 'Ultrabright' if screen == 'small' else 'Regular'
            ax[i, i_group].set_title (f'{group_name}, {screen_label} monitor ({np.round(group_cells_responsive, 2)} % responsive)', fontsize = 10)

            ax[i, i_group].axvline(object.fps * 0, c='grey', alpha=0.5)
            ax[i, i_group].axvline(object.fps * 4, c='black', alpha=0.5)
            ax[i, i_group].axvline(object.fps * 8, c='black', alpha=1, linestyle='--')
            ax[i, i_group].axvline(object.fps * 12, c='black', alpha=0.5)
            ax[i, i_group].axvline(object.fps * 16, c='grey', alpha=0.5)
            ax[i, i_group].set_xlabel('Time since stimulus onset (s)')
            ax[i, i_group].set_ylabel('Z-scored response')
            ax[i, i_group].set_xticks(np.arange(min_timepoints)[::4 * object.fps],
                             (np.arange(min_timepoints) // object.fps)[::4 * object.fps])

    plt.suptitle('Full-Field Flash response')
    plt.tight_layout()
    save_fig(object, 'Full-Field Flash', 'average on-off responses groups')
    plt.show()



def avg_onoff_response(object):
    '''
    Using z-scored responses, calculate each group's average response to the Full-Field-Flash (FFF)
    '''

    groups = np.unique([animal[3:-3] for animal in object.dat.keys()])
    colours = ['black', 'blue', 'red']

    min_timepoints = np.array(list(chain.from_iterable([object.dat[animal][day][sub_file]['zscored_matrix_baseline'].shape[-1]
                                                        for sub_file in object.dat[animal][day].keys() if ('chirp' in sub_file)]
                                                        for animal in object.dat.keys()
                                                        for day in object.dat[animal].keys()))).min()

    screens = ['big', 'small']
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey = True)

    for i, screen in enumerate(screens):
        # average response across recordings, repeats, across cells

        for i_group, group_name in enumerate(groups):
            group_dat = np.array(list(chain.from_iterable(
                [object.dat[animal][day][sub_file]['zscored_matrix_baseline'][1:, object.dat[animal][day][sub_file]['thresholded_cells'] == 1, :min_timepoints].mean(axis=(0, 1))
                 for sub_file in object.dat[animal][day].keys() if (('chirp' in sub_file) and (screen in sub_file))]
                for animal in object.dat.keys() if (group_name in animal)
                for day in object.dat[animal].keys())))

            group_cells_responsive = np.array(list(chain.from_iterable(
                [np.round(100 * object.dat[animal][day][sub_file]['thresholded_cells'].sum() / len(
                    object.dat[animal][day][sub_file]['thresholded_cells']), 2)
                 for sub_file in object.dat[animal][day].keys() if (('chirp' in sub_file) and (screen in sub_file))]
                for animal in object.dat.keys() if (group_name in animal)
                for day in object.dat[animal].keys()))).mean()

            ax[i].plot(np.arange(min_timepoints), np.nanmean(group_dat, axis=0), color=colours[i_group], alpha = 0.7, label=f'{group_name} ({np.round(group_cells_responsive, 2)} % responsive)')
            ax[i].fill_between(np.arange(min_timepoints), np.nanmean(group_dat, axis=0) - stats.sem(group_dat, axis=0, nan_policy = 'omit'),np.nanmean(group_dat, axis=0) + stats.sem(group_dat, axis=0, nan_policy = 'omit'), color = colours[i_group], alpha = 0.4)

        ax[i].axvline(object.fps * 0, c='grey', alpha = 0.5)
        ax[i].axvline(object.fps * 4, c='black', alpha = 0.5)
        ax[i].axvline(object.fps * 8, c='black', alpha = 1, linestyle = '--')
        ax[i].axvline(object.fps * 12, c='black', alpha = 0.5)
        ax[i].axvline(object.fps * 16, c='grey', alpha = 0.5)
        ax[i].set_title('Ultrabright Monitor' if screen == 'small' else 'Regular Monitor')
        ax[i].set_xlabel('Time since stimulus onset (s)')
        ax[i].set_ylabel('Z-scored response')
        ax[i].set_xticks(np.arange(min_timepoints)[::4 * object.fps],
                         (np.arange(min_timepoints) // object.fps)[::4 * object.fps])
        ax[i].legend(prop={'size': 7})

    plt.suptitle('Full-Field Flash response')
    save_fig(object, 'Full-Field Flash', 'average on-off responses')
    plt.show()

def plot_on_off_index(object, thresholded_cells = True):
    groups = np.unique([animal[3:-3] for animal in object.dat.keys()])
    colours = ['black', 'blue', 'red']
    screens = ['big', 'small']
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey = True)
    for i, screen in enumerate(screens):
        for i_group, group_name in enumerate(groups):
            if thresholded_cells:
                group_dat = np.hstack(list(chain.from_iterable(
                    [object.dat[animal][day][sub_file]['on_off_index'][object.dat[animal][day][sub_file]['thresholded_cells']==1]
                     for sub_file in object.dat[animal][day].keys() if (('chirp' in sub_file) and (screen in sub_file))]
                    for animal in object.dat.keys() if (group_name in animal)
                    for day in object.dat[animal].keys())))
            else:
                group_dat = np.hstack(list(chain.from_iterable(
                    [object.dat[animal][day][sub_file]['on_off_index']
                     for sub_file in object.dat[animal][day].keys() if (('chirp' in sub_file) and (screen in sub_file))]
                    for animal in object.dat.keys() if (group_name in animal)
                    for day in object.dat[animal].keys())))
            ax[i].hist(group_dat, density = True,histtype='step', bins = np.linspace(-1, 1, 20), linewidth=3, edgecolor = colours[i_group], alpha = 0.8, label = group_name)
            ax[i].set_title('Ultrabright Monitor' if screen == 'small' else 'Regular Monitor')
            ax[i].set_xlabel('ON/OFF ratio')
            ax[i].set_ylabel('Proportion of cells')
    plt.suptitle('ON/OFF ratio (1 = ON, -1 = OFF)')
    plt.legend()
    save_fig(object, 'Full-Field Flash', 'ON_OFF ratio')
    plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for i, screen in enumerate(screens):
        for i_group, group_name in enumerate(groups):
            if thresholded_cells:
                group_dat = np.array(list(chain.from_iterable(
                    [np.nanmean(object.dat[animal][day][sub_file]['on_off_index'][
                         object.dat[animal][day][sub_file]['thresholded_cells'] == 1])
                     for sub_file in object.dat[animal][day].keys() if (('chirp' in sub_file) and (screen in sub_file))]
                    for animal in object.dat.keys() if (group_name in animal)
                    for day in object.dat[animal].keys())))
            else:
                group_dat = np.array(list(chain.from_iterable(
                    [np.nanmean(object.dat[animal][day][sub_file]['on_off_index'])
                     for sub_file in object.dat[animal][day].keys() if (('chirp' in sub_file) and (screen in sub_file))]
                    for animal in object.dat.keys() if (group_name in animal)
                    for day in object.dat[animal].keys())))

            [ax[i].scatter(i_group, group_dat[i_sample], color=colours[i_group], s=25, alpha=0.3) for i_sample in range(len(group_dat))]
            ax[i].scatter(i_group, np.nanmean(group_dat), s=70, color=colours[i_group], alpha=1, label=group_name)
            ax[i].set_title('Ultrabright Monitor' if screen == 'small' else 'Regular Monitor')
            ax[i].set_xticks([])
            ax[i].set_ylabel('Average On/OFF index')
    plt.suptitle('ON/OFF ratio (1 = ON, -1 = OFF)')
    plt.legend()
    save_fig(object, 'Full-Field Flash', 'ON_OFF ratio scatter')
    plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for i, screen in enumerate(screens):
        for i_group, group_name in enumerate(groups):
            if thresholded_cells:
                group_off = np.array(list(chain.from_iterable(
                            [100*(object.dat[animal][day][sub_file]['on_off_index'][object.dat[animal][day][sub_file]['thresholded_cells'] == 1] < -0.2).sum()/object.dat[animal][day][sub_file]['thresholded_cells'].sum()
                             for sub_file in object.dat[animal][day].keys() if (('chirp' in sub_file) and (screen in sub_file))]
                            for animal in object.dat.keys() if (group_name in animal)
                            for day in object.dat[animal].keys())))
                group_on = np.array(list(chain.from_iterable(
                            [100*(object.dat[animal][day][sub_file]['on_off_index'][object.dat[animal][day][sub_file]['thresholded_cells'] == 1] > 0.2).sum()/object.dat[animal][day][sub_file]['thresholded_cells'].sum()
                             for sub_file in object.dat[animal][day].keys() if (('chirp' in sub_file) and (screen in sub_file))]
                            for animal in object.dat.keys() if (group_name in animal)
                            for day in object.dat[animal].keys())))
            else:
                group_off = np.array(list(chain.from_iterable(
                            [100*(object.dat[animal][day][sub_file]['on_off_index'] < -0.2).sum()/object.dat[animal][day][sub_file]['thresholded_cells'].sum()
                             for sub_file in object.dat[animal][day].keys() if (('chirp' in sub_file) and (screen in sub_file))]
                            for animal in object.dat.keys() if (group_name in animal)
                            for day in object.dat[animal].keys())))
                group_on = np.array(list(chain.from_iterable(
                            [100*(object.dat[animal][day][sub_file]['on_off_index'] > 0.2).sum()/object.dat[animal][day][sub_file]['thresholded_cells'].sum()
                             for sub_file in object.dat[animal][day].keys() if (('chirp' in sub_file) and (screen in sub_file))]
                            for animal in object.dat.keys() if (group_name in animal)
                            for day in object.dat[animal].keys())))

            [ax[i].scatter(i_group*len(groups), group_off[i_sample], color=colours[i_group], marker = '^', s=25, alpha=0.3) for i_sample in range(len(group_off))]
            ax[i].scatter(i_group*len(groups), np.nanmean(group_off), s=70, color=colours[i_group],marker = '^', alpha=1, label=group_name)

            [ax[i].scatter(i_group*len(groups) + 1, group_on[i_sample], color=colours[i_group], s=25, alpha=0.3) for i_sample in range(len(group_on))]
            ax[i].scatter(i_group*len(groups) +1, np.nanmean(group_on), s=70, color=colours[i_group], alpha=1, label=group_name)

            ax[i].set_title('Ultrabright Monitor' if screen == 'small' else 'Regular Monitor')
            ax[i].set_xticks(np.array([[i*len(groups), i*len(groups)+1] for i in range(3)]).flatten(), ['OFF', 'ON']*len(groups))
            ax[i].set_ylabel('Proportion of ON vs. OFF cells')
    plt.suptitle('ON vs OFF proportion')
    plt.legend()
    save_fig(object, 'Full-Field Flash', 'ON_OFF prop scatter')
    plt.show()


def plot_response_speed(object, thresholded_cells = True):
    groups = np.unique([animal[3:-3] for animal in object.dat.keys()])
    colours = ['black', 'blue', 'red']
    screens = ['big', 'small']
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey = True)
    for i, screen in enumerate(screens):
        for i_group, group_name in enumerate(groups):
            if thresholded_cells:
                group_dat = np.hstack(list(chain.from_iterable(
                    [object.dat[animal][day][sub_file]['on_slope'][(object.dat[animal][day][sub_file]['on_slope']>0) & (object.dat[animal][day][sub_file]['thresholded_cells']==1)]
                     for sub_file in object.dat[animal][day].keys() if (('chirp' in sub_file) and (screen in sub_file))]
                    for animal in object.dat.keys() if (group_name in animal)
                    for day in object.dat[animal].keys())))
            else:
                group_dat = np.hstack(list(chain.from_iterable(
                    [object.dat[animal][day][sub_file]['on_slope'][(object.dat[animal][day][sub_file]['on_slope']>0)]
                     for sub_file in object.dat[animal][day].keys() if (('chirp' in sub_file) and (screen in sub_file))]
                    for animal in object.dat.keys() if (group_name in animal)
                    for day in object.dat[animal].keys())))
            #ax[i].hist(group_dat, density=True, color=colours[i_group], alpha=0.5,label=group_name)
            ax[i].hist(group_dat, density = True,histtype='step',bins = np.linspace(0, 5, 10),linewidth=3, edgecolor = colours[i_group], alpha = 0.8, label = group_name)
            ax[i].set_title('Ultrabright Monitor' if screen == 'small' else 'Regular Monitor')
            ax[i].set_xlabel('Response slope')
            ax[i].set_ylabel('Proportion of cells')
    plt.suptitle('Response speed')
    plt.legend()
    save_fig(object, 'Full-Field Flash', 'Response speed')
    #plt.close('all')

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey = True)
    for i, screen in enumerate(screens):
        for i_group, group_name in enumerate(groups):
            if thresholded_cells:
                # only take where slope > 0, because 0 means no increase, -1 means decrease
                group_dat = np.array(list(chain.from_iterable(
                    [object.dat[animal][day][sub_file]['on_slope'][(object.dat[animal][day][sub_file]['on_slope']>0) & (object.dat[animal][day][sub_file]['thresholded_cells']==1)].mean()
                     for sub_file in object.dat[animal][day].keys() if (('chirp' in sub_file) and (screen in sub_file))]
                    for animal in object.dat.keys() if (group_name in animal)
                    for day in object.dat[animal].keys())))
            else:
                group_dat = np.array(list(chain.from_iterable(
                    [object.dat[animal][day][sub_file]['on_slope'][(object.dat[animal][day][sub_file]['on_slope']>0)].mean()
                     for sub_file in object.dat[animal][day].keys() if (('chirp' in sub_file) and (screen in sub_file))]
                    for animal in object.dat.keys() if (group_name in animal)
                    for day in object.dat[animal].keys())))
            [ax[i].scatter(i_group, group_dat[i_sample], color = colours[i_group], s = 25, alpha = 0.3) for i_sample in range(len(group_dat))]
            ax[i].scatter(i_group, np.nanmean(group_dat), s = 70, color=colours[i_group], alpha=1, label = group_name)
            ax[i].set_title('Ultrabright Monitor' if screen == 'small' else 'Regular Monitor')
            ax[i].set_ylabel('Response speed')
            #ax[i].set_xlabel('Group')
            ax[i].set_xticks([])
    plt.suptitle('Response speed')
    plt.legend()
    save_fig(object, 'Full-Field Flash', 'Response speed scatter')
    plt.show()


def plot_response_amplitude(object, thresholded_cells = True):
    groups = np.unique([animal[3:-3] for animal in object.dat.keys()])
    colours = ['black', 'blue', 'red']
    screens = ['big', 'small']
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey = True)
    for i, screen in enumerate(screens):
        for i_group, group_name in enumerate(groups):
            if thresholded_cells:
                group_dat = np.hstack(list(chain.from_iterable(
                    [object.dat[animal][day][sub_file]['response_amplitude'][object.dat[animal][day][sub_file]['thresholded_cells']==1]
                     for sub_file in object.dat[animal][day].keys() if (('chirp' in sub_file) and (screen in sub_file))]
                    for animal in object.dat.keys() if (group_name in animal)
                    for day in object.dat[animal].keys())))
            else:
                group_dat = np.hstack(list(chain.from_iterable(
                    [object.dat[animal][day][sub_file]['response_amplitude']
                     for sub_file in object.dat[animal][day].keys() if (('chirp' in sub_file) and (screen in sub_file))]
                    for animal in object.dat.keys() if (group_name in animal)
                    for day in object.dat[animal].keys())))
            #ax[i].hist(group_dat, density=True, color=colours[i_group], alpha=0.5,label=group_name)
            ax[i].hist(group_dat, density = True,histtype='step',bins = np.linspace(0, 5, 10), linewidth=3,edgecolor = colours[i_group], alpha = 0.8, label = group_name)
            ax[i].set_title('Ultrabright Monitor' if screen == 'small' else 'Regular Monitor')
            ax[i].set_xlabel('Response amplitude')
            ax[i].set_ylabel('Proportion of cells')
    plt.suptitle('Response amplitude')
    plt.legend()
    save_fig(object, 'Full-Field Flash', 'Response amplitude')
    #plt.close('all')

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey = True)
    for i, screen in enumerate(screens):
        for i_group, group_name in enumerate(groups):
            if thresholded_cells:
                group_dat = np.array(list(chain.from_iterable(
                    [object.dat[animal][day][sub_file]['response_amplitude'][object.dat[animal][day][sub_file]['thresholded_cells']==1].mean()
                     for sub_file in object.dat[animal][day].keys() if (('chirp' in sub_file) and (screen in sub_file))]
                    for animal in object.dat.keys() if (group_name in animal)
                    for day in object.dat[animal].keys())))
            else:
                group_dat = np.array(list(chain.from_iterable(
                    [object.dat[animal][day][sub_file]['response_amplitude'].mean()
                     for sub_file in object.dat[animal][day].keys() if (('chirp' in sub_file) and (screen in sub_file))]
                    for animal in object.dat.keys() if (group_name in animal)
                    for day in object.dat[animal].keys())))

            [ax[i].scatter(i_group, group_dat[i_sample], color = colours[i_group], s = 25, alpha = 0.3) for i_sample in range(len(group_dat))]
            ax[i].scatter(i_group, np.nanmean(group_dat), s = 70, color=colours[i_group], alpha=1, label = group_name)
            ax[i].set_title('Ultrabright Monitor' if screen == 'small' else 'Regular Monitor')
            ax[i].set_ylabel('Response amplitude')
            #ax[i].set_xlabel('Group')
            ax[i].set_xticks([])
    plt.suptitle('Response amplitude')
    plt.legend()
    save_fig(object, 'Full-Field Flash', 'Response amplitude scatter')
    plt.show()

if __name__ == '__main__':

    # now cluster different cell types
    neural_dat = \
    data_object.dat['EC_GCaMP6s_09']['20241122']['big100_chirps_000_007']['zscored_matrix_baseline'].mean(axis=0)[
        data_object.dat['EC_GCaMP6s_09']['20241122']['big100_chirps_000_007']['thresholded_cells'] == 1]
    neural_dat = \
    data_object.dat['EC_GCaMP6s_06']['20241121']['big100_chirps_000_000']['zscored_matrix_baseline'].mean(axis=0)[
        data_object.dat['EC_GCaMP6s_06']['20241121']['big100_chirps_000_000']['thresholded_cells'] == 1]

    neural_dat = \
    data_object.dat['EC_RD1_09']['20250110']['big100_chirps_000_001']['zscored_matrix_baseline'].mean(axis=0)[
        data_object.dat['EC_RD1_09']['20250110']['big100_chirps_000_001']['thresholded_cells'] == 1]
    neural_dat = \
    data_object.dat['EC_RD1_09']['20250110']['big100_chirps_000_001']['zscored_matrix_baseline'].mean(axis=0)[
        data_object.dat['EC_RD1_09']['20250110']['big100_chirps_000_001']['thresholded_cells'] == 1]

    n_clusters = 5

    classifications = {}  # {group: {} for group in groups}
    response_type = {}
    for animal in data_object.dat.keys():
        classifications[animal] = {}
        response_type[animal] = {}
        for day in data_object.dat[animal].keys():
            classifications[animal][day] = {}
            response_type[animal][day] = {}
            for sub_file in [subfile for subfile in data_object.dat[animal][day].keys() if
                             ('big' in subfile and 'chirps' in subfile)]:
                classifications[animal][day][sub_file] = {}
                print(animal)
                #response_type[animal][day][sub_file] = {}
                neural_dat = \
                    data_object.dat[animal][day][sub_file]['zscored_matrix_baseline'].mean(axis=0)[
                        data_object.dat[animal][day][sub_file]['thresholded_cells'] == 1]

                pca = PCA(n_components=2)
                neural_embedding = pca.fit_transform(neural_dat)  # Shape: (n_cells, 2)
                gmm = GaussianMixture(n_components=n_clusters, random_state=42)
                cluster_labels = gmm.fit_predict(neural_embedding)

                fig, ax = plt.subplots(2, 3, figsize=(15, 8))
                ax = ax.ravel()
                plasma_trunc = mcolors.LinearSegmentedColormap.from_list(
                    "plasma_trunc", plt.cm.plasma(np.linspace(0, 0.85, 256)))
                sns.scatterplot(x=neural_embedding[:, 0], y=neural_embedding[:, 1], hue=cluster_labels,
                                palette=plasma_trunc, s=50, alpha=0.8, ax=ax[0])
                ax[0].legend()
                ax[0].set_title('GMM neural clustering')
                cluster_classifications = []
                for i in range(n_clusters):
                    cluster_response = neural_dat[np.argwhere(cluster_labels == i)[:, 0], :]
                    ax[i + 1].plot(cluster_response.T, c='grey', alpha=0.5)
                    # ax[i+1].plot(smoothed_trace, c='blue', alpha=0.8, linewidth = 4)
                    ax[i + 1].plot(cluster_response.mean(axis=0), c='black', alpha=1, linewidth=2)
                    ax[i + 1].axvline(data_object.fps * 4, c='grey')
                    ax[i + 1].axvline(data_object.fps * 8, c='grey')
                    ax[i + 1].axvline(data_object.fps * 12, c='grey')
                    ax[i + 1].axvline(data_object.fps * 16, c='grey')
                    ax[i + 1].set_ylim([-2, 12])

                    # smooth_window = 10
                    # smoothed_trace = np.convolve(cluster_response.mean(axis=0), np.ones(smooth_window) / smooth_window, mode="same")
                    # if i ==2:
                    features = compute_cluster_features(cluster_response)
                    classification = classify_clusters(features)
                    cluster_classifications.append(classification)
                    ax[i + 1].set_title(
                        f'cluster {i} ({classification}), n = {len(np.argwhere(cluster_labels == i)[:, 0])}')

                    percent_cells_classified = (cluster_response.shape[0] / neural_dat.shape[0]) * 100
                    # plt.show()
                    if classification in classifications[animal][day][sub_file]:  # if key in dict
                        classifications[animal][day][sub_file][classification] += percent_cells_classified
                    else:  # if not in dict, make new list
                        classifications[animal][day][sub_file][classification] = percent_cells_classified
                plt.suptitle(f'{animal}, {day}, {sub_file[:3]}')
                plt.show()

                cluster_labels_classification = []
                for i in cluster_labels:
                    cluster_labels_classification.append(cluster_classifications[i])
                # dictionary storing response for each type
                response_type[animal][day][sub_file] = {type: np.array([neural_dat[i,:] for i, val in enumerate(cluster_labels_classification) if val == type]) for type in cluster_classifications}

    #PLOTTING response slope for the different groups
    groups = np.unique([animal.split('_')[1] for animal in data_object.dat.keys()])
    group_dat = {g: {'ON': [], 'OFF': []} for g in groups}
    fig, ax = plt.subplots(1, len(group_classifications.keys()), figsize=(10, 4), sharey=True)
    for animal in response_type.keys():
        for day in response_type[animal].keys():
            for sub_file in response_type[animal][day].keys():
                if "MWopto" in animal:
                    group = "MWopto"
                    group_n = 2
                    colour = 'blue'
                elif "RD1" in animal:
                    group = "RD1"
                    group_n = 1
                    colour = 'red'
                elif "GCaMP6s" in animal:
                    group = "GCaMP6s"
                    group_n = 0
                    colour = 'green'

                d = response_type[animal][day][sub_file]
                if 'ON' in d:
                    on_slope = slope(d['ON'][:, data_object.fps * 8:data_object.fps * 12])
                else:
                    on_slope = [np.nan]
                # if 'Promiscuous' in d:
                #     prom_on_slope = slope(d['Promiscuous'][:, data_object.fps * 8:data_object.fps * 12])
                #     #prom_off_slope = slope(d['Promiscuous'][:, data_object.fps * 8:data_object.fps * 12])
                # else:
                #     prom_on_slope = [np.nan]
                # on_slope = np.hstack((on_slope, prom_on_slope))
                if 'OFF' in d:
                    off_slope = np.hstack((slope(d['OFF'][:, data_object.fps * 4:data_object.fps * 8]),
                                           slope(d['OFF'][:, data_object.fps * 12:data_object.fps * 16])))
                else:
                    off_slope = [np.nan]

                group_dat[group]['ON'].append(np.nanmean(on_slope))
                group_dat[group]['OFF'].append(np.nanmean(off_slope))
                ax[group_n].scatter([0] * len(on_slope), on_slope, c=colour, s=40, alpha=0.5)
                ax[group_n].scatter([1] * len(off_slope), off_slope, c=colour, s=40, alpha=0.5)
                ax[group_n].set_xticks(np.arange(2))
                ax[group_n].set_xticklabels(['ON', 'OFF'])
                ax[group_n].set_title(group)
                ax[group_n].set_ylabel('response slope')
                ax[group_n].set_xlabel('cell type')
                # ax[group_n].set_ylim(-5, 100)
    for group in group_dat.keys():
        if "MWopto" in group:
            group_n = 2
            colour = 'blue'
        elif "RD1" in group:
            group_n = 1
            colour = 'red'
        elif "GCaMP6s" in group:
            group_n = 0
            colour = 'green'
        ax[group_n].bar([0], np.nanmean(group_dat[group]['ON']), color=colour, alpha=0.5)
        ax[group_n].bar([1], np.nanmean(group_dat[group]['OFF']), color=colour, alpha=0.5)
    plt.tight_layout()
    plt.show()


    # plotting the number of ON/OFF/WEAK cell types for each group
    groups = np.unique([animal.split('_')[1] for animal in data_object.dat.keys()])
    group_classifications = {key: {} for key in groups}
    fig, ax = plt.subplots(1, len(group_classifications.keys()), figsize=(10, 4))
    for animal in classifications.keys():
        for day in classifications[animal].keys():
            for sub_file in classifications[animal][day].keys():
                if "MWopto" in animal:
                    group = "MW"
                    group_n = 2
                    colour = 'blue'
                elif "RD1" in animal:
                    group = "RD1"
                    group_n = 1
                    colour = 'red'
                elif "GCaMP6s" in animal:
                    group = "GCaMP6s"
                    group_n = 0
                    colour = 'green'

                d = classifications[animal][day][sub_file]
                weak = d['Weak'] if 'Weak' in d else 0
                prom = d['Promiscuous'] if 'Promiscuous' in d else 0
                on = d['ON'] if 'ON' in d else 0
                off = d['OFF'] if 'OFF' in d else 0

                print(animal, on, off)

                y = np.array([on, off, weak])#np.array([on, off, prom, weak])
                ax[group_n].scatter(np.arange(len(y)), y, c=colour, s=40, alpha=0.7)
                ax[group_n].plot(y, c=colour, alpha=0.4)
                ax[group_n].set_xticks(np.arange(3))
                ax[group_n].set_xticklabels(['ON', 'OFF', 'Weak']) # ['ON', 'OFF', 'Prom', 'Weak'])
                ax[group_n].set_title(group)
                ax[group_n].set_ylabel('% responsive cells')
                ax[group_n].set_xlabel('cell type')
                ax[group_n].set_ylim(-5, 100)
    plt.tight_layout()
    plt.show()

    def slope (arr):
        '''
        :param arr: array of shape n_cells, n_timepoints
        :return:
        '''
        R_max = arr.max(axis=1)  # max response for each cell
        threshold_10, threshold_90 = 0.1 * R_max, 0.9 * R_max  # 10% and 90% thresholds
        time_axis = np.arange(arr.shape[1])  # time axis  (n_timepoints,)

        # Find crossing times for each cell
        t_10 = np.array([find_threshold_crossing(arr[i, :], threshold_10[i], time_axis) for i in range(arr.shape[0])])
        t_90 = np.array([find_threshold_crossing(arr[i, :], threshold_90[i], time_axis) for i in range(arr.shape[0])])

        # Compute slope (change in response over time)
        response_speed = (threshold_90 - threshold_10) / (t_90 - t_10)
        response_speed[np.isinf(response_speed)] = -1

        return response_speed

    def compute_slope(cluster_responses_arr, short_window=15, long_window=30, smooth_window=10):
        """
        Computes a robust estimate of the slope near the peak response.
        - cluster_responses_arr: 2D array (response of each neuron in a cluster)
        - smooth_window: window size for moving average smoothing
        """
        # if there are only 3 cells, we have stronger smoothing
        if cluster_responses_arr.shape[0] < 3:
            smooth_window = 50

        response_trace = cluster_responses_arr.mean(axis=0)
        # Smooth signal (moving average)
        smoothed_trace = np.convolve(response_trace, np.ones(smooth_window) / smooth_window, mode="same")

        # Fit a linear regression to estimate slope
        short_timepoints, long_timepoints = np.arange(short_window), np.arange(long_window)
        short_slope, short_intercept, _, _, _ = linregress(short_timepoints, smoothed_trace[:short_window])
        long_slope, long_intercept, _, _, _ = linregress(long_timepoints, smoothed_trace[:long_window])

        timepoints = short_timepoints if abs(short_slope) > abs(long_slope) else long_timepoints
        best_slope = short_slope if abs(short_slope) > abs(long_slope) else long_slope
        best_intercept = short_intercept if abs(short_slope) > abs(long_slope) else long_intercept

        # plt.figure()
        # plt.plot(smoothed_trace)
        # plt.plot(timepoints, best_slope * timepoints + best_intercept, color="green", linewidth=2, label="Linear Fit (Slope)")
        # plt.axvline(timepoints[0])
        # plt.axvline(timepoints[-1])
        # plt.title(best_slope)
        # plt.show()
        return best_slope


    epochs = {"grey1": (data_object.fps * 0, data_object.fps * 4), "black1": (data_object.fps * 4, data_object.fps * 8),
              "white": (data_object.fps * 8, data_object.fps * 12),
              "black2": (data_object.fps * 12, data_object.fps * 16),
              "grey2": (data_object.fps * 16, data_object.fps * 20)}


    def compute_cluster_features(cluster_responses_arr):
        cluster_responses = cluster_responses_arr.mean(axis=0)
        features = {}
        for name, (start, end) in epochs.items():
            features[name + "_mean"] = cluster_responses[start:end].mean()
            features[name + "_f0"] = cluster_responses[start] # initial fluorescence value
            features[name + "_peak"] = cluster_responses[start:end].max()
            features[name + "_slope"] = compute_slope(cluster_responses_arr[:, start:end])
        return features


    def classify_clusters(features):

        avg_black_mean = np.array(features["black1_mean"], features["black2_mean"]).mean()
        max_black_peak = max(features["black1_peak"], features["black2_peak"])

        # white_responsive = features['white_peak'] > 1 and features['white_slope'] > 0
        # black_responsive = max_black_peak > 1 and np.array((features["black1_slope"], features["black2_slope"])).mean() > 0
        # grey_responsive = features['grey2_slope'] > 0
        # black_nonresponsive = avg_black_mean < 0.2 and np.array((features["black1_slope"], features["black2_slope"])).mean() < 0
        peak_multiplicative = 2.5
        white_responsive = (features['white_slope'] > 0.025) and (features['white_peak'] > peak_multiplicative*features['white_f0'])
        black_responsive = ((features["black1_slope"] > 0.015) or (features["black2_slope"] > 0.015)) and ((features['black1_peak'] > peak_multiplicative*features['black1_f0']) or (features['black2_peak'] > peak_multiplicative*features['black2_f0']))  # np.array((features["black1_slope"], features["black2_slope"])).mean() > 0.005
        grey_responsive = features['grey2_slope'] > 0.015 and (features['grey2_peak'] > peak_multiplicative*features['grey2_f0'])
        black_nonresponsive = np.array((features["black1_slope"], features["black2_slope"])).mean() < 0.05

        print(
            f'white_responsive {white_responsive} black_responsive {black_responsive} black_nonresponsive {black_nonresponsive} grey_responsive {grey_responsive}')
        if (white_responsive or grey_responsive) and black_responsive:
            classification = "Promiscuous"
        elif white_responsive:
            classification = "ON"
        elif black_responsive:
            classification = "OFF"
        else:
            classification = "Weak"
        return classification

############## the above analysis, but looking at individual cells instead

    def slope (arr):
        '''
        :param arr: array of shape n_cells, n_timepoints
        :return:
        '''
        R_max = arr.max(axis=1)  # max response for each cell
        threshold_10, threshold_90 = 0.1 * R_max, 0.9 * R_max  # 10% and 90% thresholds
        time_axis = np.arange(arr.shape[1])  # time axis  (n_timepoints,)

        # Find crossing times for each cell
        t_10 = np.array([find_threshold_crossing(arr[i, :], threshold_10[i], time_axis) for i in range(arr.shape[0])])
        t_90 = np.array([find_threshold_crossing(arr[i, :], threshold_90[i], time_axis) for i in range(arr.shape[0])])

        # Compute slope (change in response over time)
        response_speed = (threshold_90 - threshold_10) / (t_90 - t_10)
        response_speed[np.isinf(response_speed)] = -1

        return response_speed

    def slope (arr):
        '''
        :param arr: array of shape n_cells, n_timepoints
        :return:
        '''
        R_max = arr.max()  # max response for each cell
        threshold_10, threshold_90 = 0.1 * R_max, 0.9 * R_max  # 10% and 90% thresholds
        time_axis = np.arange(len(arr))  # time axis  (n_timepoints,)

        # Find crossing times for each cell
        t_10 = find_threshold_crossing(arr, threshold_10, time_axis)
        t_90 = find_threshold_crossing(arr, threshold_90, time_axis)

        # Compute slope (change in response over time)
        response_speed = (threshold_90 - threshold_10) / (t_90 - t_10)
        #response_speed[np.isinf(response_speed)] = -1
        response_speed = np.nan if np.isinf(response_speed) else response_speed

        return response_speed

    # def compute_slope (cell_response_trace, smooth_window = 10):
    #     '''
    #     :param arr: cell response, shape n_timepoints
    #     :return:
    #     '''
    #     # convolve the trace
    #     smoothed_trace = np.convolve(cell_response_trace, np.ones(smooth_window) / smooth_window, mode="same")
    #
    #     # find min and max values to use
    #     R_max = smoothed_trace.max()  # max response for each cell
    #     threshold_10, threshold_90 = 0.1 * R_max, 0.9 * R_max  # 10% and 90% thresholds
    #
    #     # Find crossing times for these min/max values
    #     t_10 = find_threshold_crossing(smoothed_trace, threshold_10, np.arange(len(smoothed_trace)))
    #     t_90 = find_threshold_crossing(smoothed_trace, threshold_90, np.arange(len(smoothed_trace)))
    #
    #     slope, intercept, _, _, _ = linregress(np.arange(t_10, t_90), smoothed_trace[t_10:t_90])
    #
    #     # # Compute slope (change in response over time)
    #     # slope = (threshold_90 - threshold_10) / (t_90 - t_10)
    #     # slope[np.isinf(slope)] = -1
    #
    #     plt.figure()
    #     plt.plot(smoothed_trace)
    #     plt.scatter(t_10, threshold_10, c = 'red', s = 20)
    #     plt.scatter(t_90, threshold_90, c='blue', s=20)
    #     plt.plot(np.arange(t_10, t_90), slope * np.arange(t_10, t_90) + intercept, color="green", linewidth=2, label="Linear Fit (Slope)")
    #     plt.title(slope)
    #     plt.show()
    #
    #     return slope

    def compute_slope(cell_response_trace, short_window=int(data_object.fps*(3/4)), long_window=int(data_object.fps*(3/2)),slow_window = int(data_object.fps*(3)), smooth_window=int(data_object.fps/2)):
        """
        Computes a robust estimate of the slope near the peak response.
        - cluster_responses_arr: 1D array (response of neuron)
        - smooth_window: window size for moving average smoothing
        """
        # Smooth signal (moving average)
        smoothed_trace = np.convolve(cell_response_trace, np.ones(smooth_window) / smooth_window, mode="same")

        # Fit a linear regression to estimate slope
        short_timepoints, long_timepoints, slow_timepoints = np.arange(short_window), np.arange(long_window), np.arange(slow_window)
        short_slope, short_intercept, _, _, _ = linregress(short_timepoints, smoothed_trace[:short_window])
        long_slope, long_intercept, _, _, _ = linregress(long_timepoints, smoothed_trace[:long_window])
        slow_slope, slow_intercept, _, _, _ = linregress(slow_timepoints, smoothed_trace[:slow_window])

        timepoints = short_timepoints if abs(short_slope) > abs(long_slope) else long_timepoints
        best_slope = short_slope if abs(short_slope) > abs(long_slope) else long_slope
        best_intercept = short_intercept if abs(short_slope) > abs(long_slope) else long_intercept

        # plt.figure()
        # plt.plot(smoothed_trace)
        # plt.plot(timepoints, best_slope * timepoints + best_intercept, color="green", linewidth=2, label="Linear Fit (Slope)")
        # plt.axvline(timepoints[0])
        # plt.axvline(timepoints[-1])
        # plt.title(best_slope)
        # plt.show()
        return best_slope, slow_slope


    epochs = {"grey1": (data_object.fps * 0, data_object.fps * 4), "black1": (data_object.fps * 4, data_object.fps * 8),
              "white": (data_object.fps * 8, data_object.fps * 12),
              "black2": (data_object.fps * 12, data_object.fps * 16),
              "grey2": (data_object.fps * 16, data_object.fps * 20)}


    # def compute_features(cell_response):
    #     # returns these features for a specific cell
    #     features = {}
    #     for name, (start, end) in epochs.items():
    #         features[name + "_mean"] = cell_response[start:end].mean()
    #         features[name + "_f0"] = cell_response[start] # initial fluorescence value
    #         features[name + "_peak"] = cell_response[start:end].max()
    #         features[name + "_slope"], features[name + "_slow_slope"] = compute_slope(cell_response[start:end])
    #     return features
    #
    #
    # def classify_cell(features):
    #
    #     avg_black_mean = np.array(features["black1_mean"], features["black2_mean"]).mean()
    #     max_black_peak = max(features["black1_peak"], features["black2_peak"])
    #
    #     # white_responsive = features['white_peak'] > 1 and features['white_slope'] > 0
    #     # black_responsive = max_black_peak > 1 and np.array((features["black1_slope"], features["black2_slope"])).mean() > 0
    #     # grey_responsive = features['grey2_slope'] > 0
    #     # black_nonresponsive = avg_black_mean < 0.2 and np.array((features["black1_slope"], features["black2_slope"])).mean() < 0
    #     peak_multiplicative = 2.5
    #     white_responsive = (features['white_slope'] > 0.025) and (features['white_peak'] > peak_multiplicative*features['white_f0'])
    #     black_responsive = ((features["black1_slope"] > 0.015) or (features["black2_slope"] > 0.015)) and ((features['black1_peak'] > peak_multiplicative*features['black1_f0']) or (features['black2_peak'] > peak_multiplicative*features['black2_f0']))  # np.array((features["black1_slope"], features["black2_slope"])).mean() > 0.005
    #     grey_responsive = features['grey2_slope'] > 0.015 and (features['grey2_peak'] > peak_multiplicative*features['grey2_f0'])
    #     black_nonresponsive = np.array((features["black1_slope"], features["black2_slope"])).mean() < 0.05
    #
    #     # SLOW REPONSIVE
    #     peak_multiplicative = 2.5
    #     white_slow_responsive = (features['white_slow_slope'] > 0.025) and (features['white_peak'] > peak_multiplicative*features['white_f0'])
    #     black_slow_responsive = ((features["black1_slow_slope"] > 0.015) or (features["black2_slow_slope"] > 0.015)) and ((features['black1_peak'] > peak_multiplicative*features['black1_f0']) or (features['black2_peak'] > peak_multiplicative*features['black2_f0']))  # np.array((features["black1_slope"], features["black2_slope"])).mean() > 0.005
    #     grey_slow_responsive = features['grey2_slow_slope'] > 0.015 and (features['grey2_peak'] > peak_multiplicative*features['grey2_f0'])
    #
    #
    #     #print(f'white_responsive {white_responsive} black_responsive {black_responsive} black_nonresponsive {black_nonresponsive} grey_responsive {grey_responsive}')
    #     if (white_responsive or grey_responsive) and black_responsive:
    #         classification = "ON-OFF"
    #     elif white_responsive:
    #         classification = "ON"
    #     elif black_responsive:
    #         classification = "OFF"
    #     elif white_slow_responsive:
    #         classification = "ON"
    #     elif black_slow_responsive:
    #         classification = "OFF"
    #     else:
    #         classification = "Weak"
    #     return classification


    def compute_features(cell_response):
        '''
        :param cell_response: array shape(n_repeats, n_timepoints) > for each cell
        :param threshold:
        :return: features with arrays (1d)
        '''
        # returns these features for a specific cell
        features = {}
        for name, (start, end) in epochs.items():
            if name == 'grey1':
                features[name + "_mean"] = cell_response[:, start:end].mean(axis = 0)
                features[name + "_median"] = np.median(cell_response[:, start:end], axis = 0) # initial fluorescence value
            else: # we can look a few frames back
                baseline = cell_response[:, start-10:start].mean(axis = 1) # mean over timepoints

                # subtract baseline, mean/median over repeats
                features[name + "_mean"] = (cell_response[:, start:end] - baseline[:,None]).mean(axis = 0)
                features[name + "_median"] = np.median(cell_response[:, start:end] - baseline[:,None], axis = 0) # initial fluorescence value

        return features


    def classify_cell(features, threshold):


        #print((features['white_mean'] > threshold).shape)
        #print(features['white_median'].shape)
        #
        # cell_exceeds_threshold = (z_scored_response_mean > cell_threshold[:, None]).any(axis=1) & (
        #             z_scored_response_median > cell_threshold[:, None] * 0.9).any(axis=1)

        std_mult = 1.6
        white_responsive = (features['white_mean'] > threshold).any() & (features['white_median'] > threshold*std_mult).any()
        black1_responsive = (features['black1_mean'] > threshold).any() & (features['black1_median'] > threshold * std_mult).any()
        black2_responsive = (features['black2_mean'] > threshold).any() & (features['black2_median'] > threshold *std_mult).any()
        black_responsive = black1_responsive or black2_responsive
        grey_responsive = (features['grey2_mean'] > threshold).any() & (features['grey2_median'] > threshold * std_mult).any()

        #print(f'white_responsive {white_responsive} black_responsive {black_responsive} black_nonresponsive {black_nonresponsive} grey_responsive {grey_responsive}')
        if (white_responsive or grey_responsive) and black_responsive:
            classification = "ON-OFF"
        elif white_responsive:
            classification = "ON"
        elif black_responsive:
            classification = "OFF"
        else:
            classification = "Weak"
        return classification

    max_response = max(
        np.max(arr)
        for group in response_groups.values()
        for day in group.values()
        for subfile in day.values()
        for arr in subfile.values()
        if len(arr) > 0  # Skip empty arrays
    )
    min_response = min(
        np.min(arr)
        for group in response_groups.values()
        for day in group.values()
        for subfile in day.values()
        for arr in subfile.values()
        if len(arr) > 0  # Skip empty arrays
    )

    # classifications has the amount of each cell type (each animal is an entry)
    classifications = {group: {'ON': [],'OFF': [],'Weak': [],'ON-OFF': [] } for group in ['GCaMP6s', 'RD1', 'MWopto']}
    response_groups = {}  # {group: {} for group in groups}
    for animal in data_object.dat.keys():

        #if 'MWopto' in animal:
            # if 'MWopto_04' not in animal:
            #     continue
            # print(animal)
        group = animal.split('_')[1]
        response_groups[animal] = {}
        for day in data_object.dat[animal].keys():
            response_groups[animal][day] = {}
            for sub_file in [subfile for subfile in data_object.dat[animal][day].keys() if
                             ('big' in subfile and 'chirps' in subfile)]:
                response_groups[animal][day][sub_file] = {'ON': [], 'OFF': [], 'Weak': [], 'ON-OFF': []}

                # shape n_repeats, n_cells, n_timepoints
                z_scored_response = data_object.dat[animal][day][sub_file]['zscored_matrix_baseline'][:,data_object.dat[animal][day][sub_file]['thresholded_cells'] == 1]

                # cell-by-cell threshold : first second of the response (baseline period) > std over repeats and timepoints > multiply each cell by 1.5
                std_threshold = 1
                cell_threshold = z_scored_response[..., :data_object.fps].std(axis=(0, -1)) * std_threshold

                # now iterate through each cell to classify them
                # for each cell, compute its classification and store it
                for cell in range(z_scored_response.shape[1]):
                    zscore_arr = z_scored_response[:, cell, :]  # shape n_repeats, n_timepoints
                    features = compute_features(zscore_arr)
                    classification = classify_cell(features, cell_threshold[cell])
                    response_groups[animal][day][sub_file][classification].append(np.convolve(zscore_arr.mean(axis=0), np.ones(5) / 5, mode="same"))


                # # for each cell, compute its classification and store it
                # for cell in range(neural_dat.shape[0]):
                #     features = compute_features(neural_dat[cell])
                #     classification = classify_cell(features)
                #     response_groups[animal][day][sub_file][classification].append(neural_dat[cell])

                # turn values into arrays
                for cell_type in response_groups[animal][day][sub_file].keys():
                    response_groups[animal][day][sub_file][cell_type] = np.array(response_groups[animal][day][sub_file][cell_type])

                fig, ax = plt.subplots (2,2, figsize = (5,5), sharey = True)
                ax = ax.ravel()
                plasma_trunc = mcolors.LinearSegmentedColormap.from_list(
                    "plasma_trunc", plt.cm.plasma(np.linspace(0, 0.85, 256)))

                if "MWopto" == group:
                    colour = 'blue'
                elif "RD1" == group:
                    colour = 'firebrick'
                elif "GCaMP6s" == group:
                    colour = 'black'

                for i, cell_type in enumerate(response_groups[animal][day][sub_file].keys()):
                    response = response_groups[animal][day][sub_file][cell_type]

                    #ax[i].plot(response.T, c='grey', alpha=0.5)
                    ax[i].plot(response.mean(axis=0), c=colour, alpha=1, linewidth=2)
                    # ax[i].axvline(data_object.fps * 4, c='grey')
                    # ax[i].axvline(data_object.fps * 8, c='grey')
                    # ax[i].axvline(data_object.fps * 12, c='grey')
                    # ax[i].axvline(data_object.fps * 16, c='grey')

                    ax[i].axvspan(data_object.fps * 0, data_object.fps * 4, color='grey', alpha=0.2)
                    ax[i].axvspan(data_object.fps * 4, data_object.fps * 8, color='black', alpha=0.4)
                    ax[i].axvspan(data_object.fps * 12, data_object.fps * 16, color='black', alpha=0.4)
                    ax[i].axvspan(data_object.fps * 16, data_object.fps * 20, color='grey', alpha=0.2)
                    ax[i].set_title(f'{cell_type}, n = {response.shape[0]}')
                    ax[i].set_ylim([min_response,4])

                plt.suptitle(animal)
                plt.show()

                for cell_type in response_groups[animal][day][sub_file].keys():
                    percent_cells_classified = 100*(response_groups[animal][day][sub_file][cell_type].shape[0]/z_scored_response.shape[1])
                    classifications[group][cell_type].append(percent_cells_classified)

    # turn list into array
    for group in classifications:
        for cell_type in classifications[group]:
            classifications[group][cell_type] = np.array(classifications[group][cell_type])

    # plotting average response for all animals
    group_data = {
        'MWopto': {'ON': [], 'OFF': [], 'Weak': [], 'ON-OFF': []},
        'GCaMP6s': {'ON': [], 'OFF': [], 'Weak': [], 'ON-OFF': []},
        'RD1': {'ON': [], 'OFF': [], 'Weak': [], 'ON-OFF': []},
    }
    for animal in response_groups:
        for day in response_groups[animal]:
            for sess in response_groups[animal][day]:
                for cell_type in ['ON', 'OFF', 'Weak', 'ON-OFF']:
                    data = response_groups[animal][day][sess][cell_type]
                    if data.size > 0:
                        group_data[animal.split('_')[1]][cell_type].append(data)

    # turn list into array
    for group in group_data:
        for cell_type in classifications[group]:
            list = group_data[group][cell_type]
            if len(list) > 0:
                group_data[group][cell_type] = np.vstack([arr[:,:395] for arr in group_data[group][cell_type]])


    fig, ax = plt.subplots(2, 2, figsize=(7, 5), sharey=True)
    ax = ax.ravel()
    for group in group_data:

        if "MWopto" == group:
            colour = 'blue'
        elif "RD1" == group:
            colour = 'firebrick'
        elif "GCaMP6s" == group:
            colour = 'black'

        for i, cell_type in enumerate(group_data[group]):
            response = group_data[group][cell_type]

            if len(response) > 0:

                mean_response = response.mean(axis=0)
                sem_response = sem(response, axis=0)

                #ax[i].plot(response.T, c='grey', alpha=0.5)
                ax[i].plot(mean_response, c=colour, alpha=0.8, linewidth=2)
                ax[i].fill_between(np.arange(response.shape[1]), mean_response - sem_response, mean_response + sem_response,color=colour, alpha=0.3)
                ax[i].axvspan(data_object.fps * 0, data_object.fps * 4, color='grey', alpha=0.10)
                ax[i].axvspan(data_object.fps * 4, data_object.fps * 8, color='black', alpha=0.15)
                ax[i].axvspan(data_object.fps * 12, data_object.fps * 16, color='black', alpha=0.15)
                ax[i].axvspan(data_object.fps * 16, data_object.fps * 20, color='grey', alpha=0.10)
                ax[i].set_title(f'{cell_type}, n = {response.shape[0]}')
                ax[i].set_ylim([-0.5,2.])

        #plt.suptitle(group)
        plt.show()





    # plotting the number of ON/OFF/WEAK cell types
    groups = np.unique([animal.split('_')[1] for animal in data_object.dat.keys() if 'GNAT' not in animal])
    group_classifications = {key: {'ON':0,'OFF':0,'ON-OFF':0,'Weak':0} for key in groups}
    for animal in response_groups:
        for day in response_groups[animal]:
            for sub_file in response_groups[animal][day]:
                for cell_type in ['ON','OFF', 'ON-OFF','Weak']:
                    group_classifications [animal.split('_')[1]][cell_type] += response_groups[animal][day][sub_file][cell_type].shape[0]


    # plotting the number of ON/OFF/WEAK cell types for each group
    groups = np.unique([animal.split('_')[1] for animal in data_object.dat.keys() if 'GNAT' not in animal])
    group_classifications = {key: {} for key in groups}
    fig, ax = plt.subplots(1, len(group_classifications.keys()), figsize=(8,4))
    for group_n, group in enumerate(classifications):

        if "MWopto" == group:
            colour = 'blue'
        elif "RD1" == group:
            colour = 'firebrick'
        elif "GCaMP6s" == group:
            colour = 'black'

        d = classifications[group]
        weak = d['Weak'] if 'Weak' in d else 0
        prom = d['ON-OFF'] if 'ON-OFF' in d else 0
        on = d['ON'] if 'ON' in d else 0
        off = d['OFF'] if 'OFF' in d else 0

        ax[group_n].scatter([0]* len(on), on, c=colour, s=35, alpha=0.6)
        ax[group_n].scatter([1] * len(off), off, c=colour, s=35, alpha=0.6)
        ax[group_n].scatter([2] * len(weak), weak, c=colour, s=35, alpha=0.6)
        ax[group_n].scatter([3] * len(prom), prom, c=colour, s=35, alpha=0.6)

        ax[group_n].bar([0], on.mean(), color=colour,alpha=0.3)
        ax[group_n].bar([1], off.mean(), color=colour,alpha=0.3)
        ax[group_n].bar([2], weak.mean(), color=colour, alpha=0.3)
        ax[group_n].bar([3], prom.mean(), color=colour, alpha=0.3)

        y = np.array([on, off, weak, prom])  # np.array([on, off, prom, weak])
        ax[group_n].plot(y, c=colour, alpha=0.3)

        ax[group_n].set_xticks(np.arange(4))
        ax[group_n].set_xticklabels(['ON', 'OFF', 'Weak', 'ON-OFF']) # ['ON', 'OFF', 'Prom', 'Weak'])
        ax[group_n].set_title(group)
        ax[group_n].set_ylabel('% responsive cells')
        ax[group_n].set_xlabel('cell type')
        ax[group_n].set_ylim(-5, 100)
    plt.tight_layout()
    plt.show()

    #PLOTTING response slope for the different groups
    groups = np.unique([animal.split('_')[1] for animal in data_object.dat.keys()])
    group_dat = {g: {'ON': [], 'OFF': []} for g in groups}
    fig, ax = plt.subplots(1, len(group_classifications.keys()), figsize=(4, 3), sharey=True)
    for animal in response_groups.keys():
        for day in response_groups[animal].keys():
            for sub_file in response_groups[animal][day].keys():
                if "MWopto" in animal:
                    group = "MWopto"
                    group_n = 2
                    colour = 'blue'
                elif "RD1" in animal:
                    group = "RD1"
                    group_n = 1
                    colour = 'firebrick'
                elif "GCaMP6s" in animal:
                    group = "GCaMP6s"
                    group_n = 0
                    colour = 'black'

                d = response_groups[animal][day][sub_file]
                if len(d['ON']) > 0:
                    on_slope = slope(d['ON'].mean(axis = 0)[data_object.fps * 8:data_object.fps * 12])
                else:
                    on_slope = np.nan
                if len(d['OFF']) > 0:
                    off_slope = np.hstack((slope(d['OFF'].mean(axis = 0)[data_object.fps * 4:data_object.fps * 8]),
                                           slope(d['OFF'].mean(axis = 0)[data_object.fps * 12:data_object.fps * 16])))
                else:
                    off_slope = np.nan

                group_dat[group]['ON'].append(on_slope)
                group_dat[group]['OFF'].append(np.nanmean(off_slope))
                # ax[group_n].scatter([0] * len(on_slope), on_slope, c=colour, s=40, alpha=0.5)
                # ax[group_n].scatter([1] * len(off_slope), off_slope, c=colour, s=40, alpha=0.5)
                ax[group_n].scatter(0, on_slope, c=colour, s=35, alpha=0.5)
                ax[group_n].scatter(1 , np.nanmean(off_slope), c=colour, s=35, alpha=0.5)
                ax[group_n].set_xticks(np.arange(2))
                ax[group_n].set_xticklabels(['ON', 'OFF'])
                ax[group_n].set_title(group)
                ax[group_n].set_ylabel('response slope')
                ax[group_n].set_xlabel('cell type')
                # ax[group_n].set_ylim(-5, 100)
    for group in group_dat.keys():
        if "MWopto" in group:
            group_n = 2
            colour = 'blue'
        elif "RD1" in group:
            group_n = 1
            colour = 'firebrick'
        elif "GCaMP6s" in group:
            group_n = 0
            colour = 'black'
        ax[group_n].bar([0], np.nanmean(group_dat[group]['ON']), color=colour, alpha=0.3)
        ax[group_n].bar([1], np.nanmean(group_dat[group]['OFF']), color=colour, alpha=0.3)
        ax[group_n].errorbar([0], np.nanmean(group_dat[group]['ON']), yerr=sem(group_dat[group]['ON'], nan_policy='omit'), fmt='none', ecolor=colour, alpha = 0.5, capsize=5)
        ax[group_n].errorbar([1], np.nanmean(group_dat[group]['OFF']), yerr=sem(group_dat[group]['OFF'], nan_policy='omit'), fmt='none', ecolor=colour, alpha = 0.5, capsize=5)

    plt.tight_layout()
    plt.show()

    #PLOTTING response AMPLITUDE for the different groups
    groups = np.unique([animal.split('_')[1] for animal in data_object.dat.keys()])
    group_dat = {g: {'ON': [], 'OFF': []} for g in groups}
    fig, ax = plt.subplots(1, len(group_classifications.keys()), figsize=(4, 3), sharey=True)
    for animal in response_groups.keys():
        for day in response_groups[animal].keys():
            for sub_file in response_groups[animal][day].keys():
                if "MWopto" in animal:
                    group = "MWopto"
                    group_n = 2
                    colour = 'blue'
                elif "RD1" in animal:
                    group = "RD1"
                    group_n = 1
                    colour = 'firebrick'
                elif "GCaMP6s" in animal:
                    group = "GCaMP6s"
                    group_n = 0
                    colour = 'black'

                d = response_groups[animal][day][sub_file]
                if len(d['ON']) > 0:
                    on_amp = d['ON'].max(axis = 1).mean()
                else:
                    on_amp = np.nan
                if len(d['OFF']) > 0:
                    off_amp = np.hstack((d['OFF'].max(axis = 1).mean(),
                                           d['OFF'].max(axis = 1).mean()))
                else:
                    off_amp = np.nan

                group_dat[group]['ON'].append(on_amp)
                group_dat[group]['OFF'].append(np.nanmean(off_amp))
                # ax[group_n].scatter([0] * len(on_slope), on_slope, c=colour, s=40, alpha=0.5)
                # ax[group_n].scatter([1] * len(off_slope), off_slope, c=colour, s=40, alpha=0.5)
                ax[group_n].scatter(0, on_amp, c=colour, s=35, alpha=0.5)
                ax[group_n].scatter(1 , np.nanmean(off_amp), c=colour, s=35, alpha=0.5)
                ax[group_n].set_xticks(np.arange(2))
                ax[group_n].set_xticklabels(['ON', 'OFF'])
                ax[group_n].set_title(group)
                ax[group_n].set_ylabel('response amplitude')
                ax[group_n].set_xlabel('cell type')
                # ax[group_n].set_ylim(-5, 100)
    for group in group_dat.keys():
        if "MWopto" in group:
            group_n = 2
            colour = 'blue'
        elif "RD1" in group:
            group_n = 1
            colour = 'firebrick'
        elif "GCaMP6s" in group:
            group_n = 0
            colour = 'black'
        ax[group_n].bar([0], np.nanmean(group_dat[group]['ON']), color=colour, alpha=0.3)
        ax[group_n].bar([1], np.nanmean(group_dat[group]['OFF']), color=colour, alpha=0.3)
        ax[group_n].errorbar([0], np.nanmean(group_dat[group]['ON']), yerr=sem(group_dat[group]['ON'], nan_policy='omit'), fmt='none', ecolor=colour, alpha = 0.5, capsize=5)
        ax[group_n].errorbar([1], np.nanmean(group_dat[group]['OFF']), yerr=sem(group_dat[group]['OFF'], nan_policy='omit'), fmt='none', ecolor=colour, alpha = 0.5, capsize=5)

    plt.tight_layout()
    plt.show()


