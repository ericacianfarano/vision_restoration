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
            ax[i].hist(group_dat, density = True,bins = np.linspace(-1, 1, 20), color = colours[i_group], alpha = 0.5, label = group_name)
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
                    [object.dat[animal][day][sub_file]['peak_response_idx'][object.dat[animal][day][sub_file]['thresholded_cells']==1]
                     for sub_file in object.dat[animal][day].keys() if (('chirp' in sub_file) and (screen in sub_file))]
                    for animal in object.dat.keys() if (group_name in animal)
                    for day in object.dat[animal].keys())))
            else:
                group_dat = np.hstack(list(chain.from_iterable(
                    [object.dat[animal][day][sub_file]['peak_response_idx']
                     for sub_file in object.dat[animal][day].keys() if (('chirp' in sub_file) and (screen in sub_file))]
                    for animal in object.dat.keys() if (group_name in animal)
                    for day in object.dat[animal].keys())))
            #ax[i].hist(group_dat, density=True, color=colours[i_group], alpha=0.5,label=group_name)
            ax[i].hist(group_dat, density = True,bins = np.linspace(0, object.fps, 10), color = colours[i_group], alpha = 0.5, label = group_name)
            ax[i].set_title('Ultrabright Monitor' if screen == 'small' else 'Regular Monitor')
            ax[i].set_xlabel('Index of Peak Response')
            ax[i].set_ylabel('Proportion of cells')
    plt.suptitle('Response speed')
    plt.legend()
    save_fig(object, 'Full-Field Flash', 'Response speed')
    #plt.close('all')

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey = True)
    for i, screen in enumerate(screens):
        for i_group, group_name in enumerate(groups):
            if thresholded_cells:
                group_dat = np.array(list(chain.from_iterable(
                    [object.dat[animal][day][sub_file]['peak_response_idx'][object.dat[animal][day][sub_file]['thresholded_cells']==1].mean()
                     for sub_file in object.dat[animal][day].keys() if (('chirp' in sub_file) and (screen in sub_file))]
                    for animal in object.dat.keys() if (group_name in animal)
                    for day in object.dat[animal].keys())))
            else:
                group_dat = np.array(list(chain.from_iterable(
                    [object.dat[animal][day][sub_file]['peak_response_idx'].mean()
                     for sub_file in object.dat[animal][day].keys() if (('chirp' in sub_file) and (screen in sub_file))]
                    for animal in object.dat.keys() if (group_name in animal)
                    for day in object.dat[animal].keys())))
            [ax[i].scatter(i_group, group_dat[i_sample], color = colours[i_group], s = 25, alpha = 0.3) for i_sample in range(len(group_dat))]
            ax[i].scatter(i_group, np.nanmean(group_dat), s = 70, color=colours[i_group], alpha=1, label = group_name)
            ax[i].set_title('Ultrabright Monitor' if screen == 'small' else 'Regular Monitor')
            ax[i].set_ylabel('Index of Peak Response')
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
            ax[i].hist(group_dat, density = True,bins = np.linspace(0, 5, 10), color = colours[i_group], alpha = 0.5, label = group_name)
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
