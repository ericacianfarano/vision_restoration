
from classes import *
from full_field_plots import *
animals_days = {'EC_MWoptoasleep_08': ['20250318'],
                'EC_MWoptoasleep_10': ['20250318'],
                'EC_MWoptoawake_08': ['20250318'],
                'EC_MWoptoawake_10': ['20250318'],
                'EC_GCaMP6sasleep_06': ['20250318'],
                'EC_GCaMP6sasleep_09': ['20250318'],
                'EC_GCaMP6sawake_06': ['20250318'],
                'EC_GCaMP6sawake_09': ['20250318'],
                'EC_RD1asleep_06': ['20250315'],
                'EC_RD1asleep_08': ['20250315'],
                'EC_RD1awake_06': ['20250314'],
                'EC_RD1awake_08': ['20250314']
                }
data_object = DataAnalysis (['E:\\vision_restored', 'I:\\vision_restored'], dict_animals_days = animals_days, response_type = 'fluorescence', dlc = False, show_plots = False)




def responsive_cells_hist (object):

    conditions = ['awake', 'asleep']
    stim = 'chirps'
    screens = ['big', 'small']
    fig, ax = plt.subplots(2, 2, sharey = True, figsize=(5, 8))

    for i_cond, cond in enumerate(conditions):
        for i, screen in enumerate(screens):
            # average response across recordings, repeats, across cells
            def get_dat (group, condition, screen):
                d = np.array(list(chain.from_iterable(
                    [100 * object.dat[animal][day][sub_file]['thresholded_cells'].sum() / len(
                        object.dat[animal][day][sub_file]['thresholded_cells'])
                     for sub_file in object.dat[animal][day].keys() if ((stim in sub_file) and (screen in sub_file))]
                    for animal in object.dat.keys() if ((group in animal) and (condition in animal))
                    for day in object.dat[animal].keys() if (day in object.dict_animals_days[animal]))))
                return d

            opto = get_dat ('MWopto', cond, screen)
            rd1 = get_dat('RD1', cond, screen)
            control = get_dat('GCaMP6s', cond, screen)

            print(stim, screen)
            print('opto', opto.mean(),'rd1',rd1.mean(), 'control', control.mean())

            # if stim == 'SFxO':
            #     gnat = np.array(list(chain.from_iterable(
            #         [100*object.dat[animal][day][sub_file]['thresholded_cells'].sum()/len(object.dat[animal][day][sub_file]['thresholded_cells'])
            #          for sub_file in object.dat[animal][day].keys() if ((screen in sub_file) and ('l4' not in sub_file))]
            #         for animal in object.dat.keys() if ('GNAT' in animal)
            #         for day in object.dat[animal].keys() if (day in object.dict_animals_days[animal]))))
            #
            #     print(gnat)
            # is there a statistically sig diff in the distributions of the three groups?
            stat, p = kruskal(control, rd1, opto)
            print(f'KW H-statistic: {stat:.3f}, p-value: {p:.3f}')

            if p < 0.05: # follow up with testing pairwise comparisons (with correction for multiple comparisons)
                # Do pairwise comparisons manually:
                print('mannwhitney two-sided test, with bonferroni multiple comparison correction')
                pvals = [
                    mannwhitneyu(control, rd1, alternative='two-sided').pvalue,
                    mannwhitneyu(control, opto, alternative='two-sided').pvalue,
                    mannwhitneyu(rd1, opto, alternative='two-sided').pvalue
                ]

                # Apply Bonferroni correction manually (3 comparisons):
                _, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='bonferroni')

                print(pvals_corrected)

                # Define comparisons in same order as pvals
                comparisons = [
                    ('GCaMP6s', 'rd1'),
                    ('GCaMP6s', 'opto'),
                    ('rd1', 'opto')
                ]

                # Coordinates for group positions on x-axis
                group_coords = {'GCaMP6s': 0, 'rd1': 1, 'opto': 2}
                max_val = max(np.max(control), np.max(rd1), np.max(opto)) * 1.05
                step = max_val * 0.05  # vertical spacing between significance bars
                h = max_val

                # Loop through each comparison and plot if significant
                for (group1, group2), p_val in zip(comparisons, pvals_corrected):
                    if p_val < 0.05:
                        x1, x2 = group_coords[group1], group_coords[group2]
                        y = h
                        h += step  # update height for next line if needed

                        # Decide number of stars
                        if p_val < 0.001:
                            stars = '***'
                        elif p_val < 0.01:
                            stars = '**'
                        else:
                            stars = '*'

                        # Draw the line and stars
                        ax[i_cond, i].plot([x1, x1, x2, x2], [y, y + step / 2, y + step / 2, y], lw=1.5, c='k')
                        ax[i_cond, i].text((x1 + x2) * 0.5, y + step * 0.6, stars,
                                           ha='center', va='bottom', fontsize=14)

            ax[i_cond, i].scatter(np.array([0] * len(control)), control, s = 35, color = 'black', alpha = 0.45, label = 'control')
            ax[i_cond, i].scatter(np.array([1] * len(rd1)), rd1, s = 35, color = 'firebrick', alpha = 0.45, label = 'RD1')
            ax[i_cond, i].scatter(np.array([2] * len(opto)), opto,s = 35, color = 'blue', alpha = 0.45, label = 'opto')
            # if stim == 'SFxO':
            #     ax[i_stim, i].scatter(np.array([3] * len(gnat)), gnat, s=65, color='pink', alpha=0.6, label='GNAT')

            ax[i_cond, i].bar(np.array([0]), control.mean(), color = 'black', alpha = 0.3, label = 'control')
            ax[i_cond, i].bar(np.array([1]), rd1.mean(), color = 'firebrick', alpha = 0.3, label = 'RD1')
            ax[i_cond, i].bar(np.array([2]), opto.mean(), color = 'blue', alpha = 0.3, label = 'opto')

            ax[i_cond, i].errorbar([0], control.mean(),yerr=sem(control, nan_policy='omit'), fmt='none', ecolor='black', alpha=0.5, capsize=5)
            ax[i_cond, i].errorbar([1], rd1.mean(), yerr=sem(rd1, nan_policy='omit'), fmt='none', ecolor='firebrick',alpha=0.5, capsize=5)
            ax[i_cond, i].errorbar([2], opto.mean(), yerr=sem(opto, nan_policy='omit'), fmt='none', ecolor='blue',alpha=0.5, capsize=5)

            # if stim == 'SFxO':
            #     ax[i_stim, i].bar(np.array([3]), gnat.mean(), color='pink', alpha=0.3, label='GNAT')

            #screen_label = 'regular monitor' if 'big' in screen else 'ultrabright monitor'
            ax[i_cond, i].set_title(f'FFF {cond} ({"regular monitor" if "big" in screen else "ultrabright monitor"})')

            # if i == 0:
            #     ax[i_stim, i].set_ylabel('Proportion of responsive cells (%)')
            # if stim == 'SFxO':
            #     ax[i_stim, i].set_xticks(np.array([0,1,2,3]))
            #     ax[i_stim, i].set_xticklabels(['Control', 'RD1', 'Opto', 'GNAT'])
            # else:
            ax[i_cond, i].set_xticks(np.array([0, 1, 2]))
            ax[i_cond, i].set_xticklabels(['Control', 'RD1', 'Opto'])

    fig.text(0.04, 0.5, 'Proportion of responsive cells (%)', va='center', rotation='vertical', fontsize=12)
    plt.savefig(os.path.join(object.save_path, '% responsive cells'))
    plt.show()


epochs = {"grey1": (data_object.fps * 0, data_object.fps * 4), "black1": (data_object.fps * 4, data_object.fps * 8),
          "white": (data_object.fps * 8, data_object.fps * 12),
          "black2": (data_object.fps * 12, data_object.fps * 16),
          "grey2": (data_object.fps * 16, data_object.fps * 20)}


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
            features[name + "_mean"] = cell_response[:, start:end].mean(axis=0)
            features[name + "_median"] = np.median(cell_response[:, start:end], axis=0)  # initial fluorescence value
        else:  # we can look a few frames back
            baseline = cell_response[:, start - 10:start].mean(axis=1)  # mean over timepoints

            # subtract baseline, mean/median over repeats
            features[name + "_mean"] = (cell_response[:, start:end] - baseline[:, None]).mean(axis=0)
            features[name + "_median"] = np.median(cell_response[:, start:end] - baseline[:, None],
                                                   axis=0)  # initial fluorescence value

    return features


def classify_cell(features, threshold):
    # print((features['white_mean'] > threshold).shape)
    # print(features['white_median'].shape)
    #
    # cell_exceeds_threshold = (z_scored_response_mean > cell_threshold[:, None]).any(axis=1) & (
    #             z_scored_response_median > cell_threshold[:, None] * 0.9).any(axis=1)

    std_mult = 1.6
    white_responsive = (features['white_mean'] > threshold).any() & (
                features['white_median'] > threshold * std_mult).any()
    black1_responsive = (features['black1_mean'] > threshold).any() & (
                features['black1_median'] > threshold * std_mult).any()
    black2_responsive = (features['black2_mean'] > threshold).any() & (
                features['black2_median'] > threshold * std_mult).any()
    black_responsive = black1_responsive or black2_responsive
    grey_responsive = (features['grey2_mean'] > threshold).any() & (
                features['grey2_median'] > threshold * std_mult).any()

    # print(f'white_responsive {white_responsive} black_responsive {black_responsive} black_nonresponsive {black_nonresponsive} grey_responsive {grey_responsive}')
    if (white_responsive or grey_responsive) and black_responsive:
        classification = "ON-OFF"
    elif white_responsive:
        classification = "ON"
    elif black_responsive:
        classification = "OFF"
    else:
        classification = "Weak"
    return classification

def responsive_cells_hist (object):

    condition = ['awake', 'asleep']
    screens = ['big', 'small']
    fig, ax = plt.subplots(2, 2, sharey = True, figsize=(5, 8))

    for i_cond, cond in enumerate(condition):
        for i, screen in enumerate(screens):

            def get_dat (group):
                d = np.array(list(chain.from_iterable(
                    [100*object.dat[animal][day][sub_file]['thresholded_cells'].sum()/len(object.dat[animal][day][sub_file]['thresholded_cells'])
                     for sub_file in object.dat[animal][day].keys() if (('chirp' in sub_file) and (screen in sub_file) and ('l4' not in sub_file))]
                    for animal in object.dat.keys() if (group in animal)
                    for day in object.dat[animal].keys() if (day in object.dict_animals_days[animal]))))
                return d

            groups = np.unique([animal.split('_')[1] for animal in object.dat.keys() if cond in animal])
            for g in groups:    # ['GCaMP6sawake', 'MWoptoawake', 'RD1awake']
                d = get_dat(g)
                print(g, d)

                if 'GCaMP6s' in g:
                    colour, label, coord = 'black', 'Sighted', 0
                elif 'RD1' in g:
                    colour, label, coord = 'firebrick', 'rd1', 1
                if 'MWopto' in g:
                    colour, label, coord = 'blue', 'MWopto', 2

                ax[i_cond, i].scatter(np.array([coord] * len(d)), d, s=35, color=colour, alpha=0.45)
                ax[i_cond, i].bar(np.array([coord]), d.mean(), color = colour, alpha = 0.3, label = label)
                ax[i_cond, i].errorbar([coord], d.mean(), yerr=sem(d, nan_policy='omit'), fmt='none',ecolor=colour, alpha=0.5, capsize=5)

                ax[i_cond, i].set_title(f'{cond} ({"reg. monitor" if "big" in screen else "bright monitor"})')
                ax[i_cond, i].set_xticks(np.array([0, 1, 2]))
                ax[i_cond, i].set_xticklabels(['Control', 'RD1', 'Opto'])

    fig.text(0.04, 0.5, 'Proportion of responsive cells (%)', va='center', rotation='vertical', fontsize=12)
    #plt.savefig(os.path.join(object.save_path, '% responsive cells'))
    plt.show()

responsive_cells_hist(data_object)

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
classifications = {group: {'ON': [], 'OFF': [], 'Weak': [], 'ON-OFF': []} for group in ['GCaMP6sasleep','GCaMP6sawake', 'RD1asleep', 'RD1awake', 'MWoptoasleep','MWoptoawake']}
response_groups = {}  # {group: {} for group in groups}
for animal in data_object.dat.keys():
    if animal == 'MWoptoasleep':
        print(animal)
    group = animal.split('_')[1]
    response_groups[animal] = {}
    for day in data_object.dat[animal].keys():
        response_groups[animal][day] = {}
        for sub_file in [subfile for subfile in data_object.dat[animal][day].keys() if
                         ('big' in subfile and 'chirps' in subfile)]:
            response_groups[animal][day][sub_file] = {'ON': [], 'OFF': [], 'Weak': [], 'ON-OFF': []}

            # shape n_repeats, n_cells, n_timepoints
            z_scored_response = data_object.dat[animal][day][sub_file]['zscored_matrix_baseline'][:,
                                data_object.dat[animal][day][sub_file]['thresholded_cells'] == 1]

            # cell-by-cell threshold : first second of the response (baseline period) > std over repeats and timepoints > multiply each cell by 1.5
            std_threshold = 0.8#1
            cell_threshold = z_scored_response[..., :data_object.fps].std(axis=(0, -1)) * std_threshold

            # now iterate through each cell to classify them
            # for each cell, compute its classification and store it
            for cell in range(z_scored_response.shape[1]):
                zscore_arr = z_scored_response[:, cell, :] # shape n_repeats, n_timepoints
                features = compute_features(zscore_arr) # input is shape n_repeats, n_
                classification = classify_cell(features, cell_threshold[cell])
                response_groups[animal][day][sub_file][classification].append(np.convolve(zscore_arr.mean(axis=0), np.ones(5) / 5, mode="same"))

            # # for each cell, compute its classification and store it
            # for cell in range(neural_dat.shape[0]):
            #     features = compute_features(neural_dat[cell])
            #     classification = classify_cell(features)
            #     response_groups[animal][day][sub_file][classification].append(neural_dat[cell])

            # turn values into arrays
            for cell_type in response_groups[animal][day][sub_file].keys():
                response_groups[animal][day][sub_file][cell_type] = np.array(
                    response_groups[animal][day][sub_file][cell_type])

            fig, ax = plt.subplots(2, 2, figsize=(5, 5), sharey=True)
            ax = ax.ravel()
            plasma_trunc = mcolors.LinearSegmentedColormap.from_list(
                "plasma_trunc", plt.cm.plasma(np.linspace(0, 0.85, 256)))

            if "MWopto" in group:
                colour = 'blue'
            elif "RD1" in group:
                colour = 'firebrick'
            elif "GCaMP6s" in group:
                colour = 'black'

            for i, cell_type in enumerate(response_groups[animal][day][sub_file].keys()):
                response = response_groups[animal][day][sub_file][cell_type]

                # ax[i].plot(response.T, c='grey', alpha=0.5)
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
                #ax[i].set_ylim([min_response, 4])

            plt.suptitle(animal)
            plt.show()

            for cell_type in response_groups[animal][day][sub_file].keys():
                if z_scored_response.shape[1] > 0:
                    percent_cells_classified = 100 * (
                                response_groups[animal][day][sub_file][cell_type].shape[0] / z_scored_response.shape[1])
                else:
                    percent_cells_classified = 0
                classifications[group][cell_type].append(percent_cells_classified)

# turn list into array
for group in classifications:
    for cell_type in classifications[group]:
        classifications[group][cell_type] = np.array(classifications[group][cell_type])

# plotting average response for all animals
group_data = {
    'MWoptoawake': {'ON': [], 'OFF': [], 'Weak': [], 'ON-OFF': []},
    'GCaMP6sawake': {'ON': [], 'OFF': [], 'Weak': [], 'ON-OFF': []},
    'RD1awake': {'ON': [], 'OFF': [], 'Weak': [], 'ON-OFF': []},
    'MWoptoasleep': {'ON': [], 'OFF': [], 'Weak': [], 'ON-OFF': []},
    'GCaMP6sasleep': {'ON': [], 'OFF': [], 'Weak': [], 'ON-OFF': []},
    'RD1asleep': {'ON': [], 'OFF': [], 'Weak': [], 'ON-OFF': []}
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
            #print(group_data[group][cell_type].shape)
            group_data[group][cell_type] = np.vstack([arr[:, :395] for arr in group_data[group][cell_type]])

fig, ax = plt.subplots(2, 2, figsize=(10, 8), sharey=True)
ax = ax.ravel()
for group in group_data:

    if "MWopto" in group:
        if 'asleep' in group:
            colour = 'cornflowerblue'
        elif 'awake' in group:
            colour = 'darkblue'
    elif "RD1" in group:
        if 'asleep' in group:
            colour = 'salmon'
        elif 'awake' in group:
            colour = 'firebrick'
    elif "GCaMP6s" in group:
        if 'asleep' in group:
            colour = 'grey'
        elif 'awake' in group:
            colour = 'black'

    for i, cell_type in enumerate(group_data[group]):
        response = group_data[group][cell_type]

        if len(response) > 0:
            mean_response = response.mean(axis=0)
            sem_response = sem(response, axis=0)

            # ax[i].plot(response.T, c='grey', alpha=0.5)
            ax[i].plot(mean_response, c=colour, alpha=0.8, linewidth=2, label = group)
            ax[i].fill_between(np.arange(response.shape[1]), mean_response - sem_response, mean_response + sem_response,color=colour, alpha=0.3)
            ax[i].axvspan(data_object.fps * 0, data_object.fps * 4, color='grey', alpha=0.10)
            ax[i].axvspan(data_object.fps * 4, data_object.fps * 8, color='black', alpha=0.15)
            ax[i].axvspan(data_object.fps * 12, data_object.fps * 16, color='black', alpha=0.15)
            ax[i].axvspan(data_object.fps * 16, data_object.fps * 20, color='grey', alpha=0.10)
            ax[i].set_title(f'{cell_type}, n = {response.shape[0]}')
            ax[i].set_ylim([-1, 4.3])
            ax[i].legend()

    plt.suptitle(group)
    #plt.legend()
    plt.show()

# plotting the number of ON/OFF/WEAK cell types
groups = np.unique([animal.split('_')[1] for animal in data_object.dat.keys() if 'GNAT' not in animal])
group_classifications = {key: {'ON': 0, 'OFF': 0, 'ON-OFF': 0, 'Weak': 0} for key in groups}
for animal in response_groups:
    for day in response_groups[animal]:
        for sub_file in response_groups[animal][day]:
            for cell_type in ['ON', 'OFF', 'ON-OFF', 'Weak']:
                group_classifications[animal.split('_')[1]][cell_type] += \
                response_groups[animal][day][sub_file][cell_type].shape[0]

# plotting the number of ON/OFF/WEAK cell types for each group
groups = np.unique([animal.split('_')[1] for animal in data_object.dat.keys() if 'GNAT' not in animal])
group_classifications = {key: {} for key in groups}
fig, ax = plt.subplots(3, 2, figsize=(4.5, 7))
ax = ax.ravel()
for group_n, group in enumerate(classifications):

    if "MWopto" in group:
        if 'asleep' in group:
            colour = 'cornflowerblue'
        elif 'awake' in group:
            colour = 'darkblue'
    elif "RD1" in group:
        if 'asleep' in group:
            colour = 'salmon'
        elif 'awake' in group:
            colour = 'firebrick'
    elif "GCaMP6s" in group:
        if 'asleep' in group:
            colour = 'grey'
        elif 'awake' in group:
            colour = 'black'


    d = classifications[group]
    weak = d['Weak'] if 'Weak' in d else 0
    prom = d['ON-OFF'] if 'ON-OFF' in d else 0
    on = d['ON'] if 'ON' in d else 0
    off = d['OFF'] if 'OFF' in d else 0

    ax[group_n].scatter([0] * len(on), on, c=colour, s=35, alpha=0.6)
    ax[group_n].scatter([1] * len(off), off, c=colour, s=35, alpha=0.6)
    ax[group_n].scatter([2] * len(weak), weak, c=colour, s=35, alpha=0.6)
    ax[group_n].scatter([3] * len(prom), prom, c=colour, s=35, alpha=0.6)

    ax[group_n].bar([0], np.mean(on), color=colour, alpha=0.3)
    ax[group_n].bar([1], np.mean(off), color=colour, alpha=0.3)
    ax[group_n].bar([2], np.mean(weak), color=colour, alpha=0.3)
    ax[group_n].bar([3], np.mean(prom), color=colour, alpha=0.3)

    y = np.array([on, off, weak, prom])  # np.array([on, off, prom, weak])
    ax[group_n].plot(y, c=colour, alpha=0.3)

    ax[group_n].set_xticks(np.arange(4))
    ax[group_n].set_xticklabels(['ON', 'OFF', 'Weak', 'ON-OFF'])  # ['ON', 'OFF', 'Prom', 'Weak'])
    ax[group_n].set_title(group)
    if group_n == 0 :
        ax[group_n].set_ylabel('% responsive cells')
    ax[group_n].set_xlabel('cell type')
    ax[group_n].set_ylim(-5, 100)
plt.tight_layout()
plt.show()



def avg_onoff_response(object):
    '''
    Using z-scored responses, calculate each group's average response to the Full-Field-Flash (FFF)
    '''

    groups = np.unique([animal.split('_')[1] for animal in object.dat.keys()])

    min_timepoints = np.array(list(chain.from_iterable([object.dat[animal][day][sub_file]['zscored_matrix_baseline'].shape[-1]
                                                        for sub_file in object.dat[animal][day].keys() if ('chirp' in sub_file)]
                                                        for animal in object.dat.keys()
                                                        for day in object.dat[animal].keys()))).min()

    screens = ['big', 'small']
    fig, ax = plt.subplots(2, 2, figsize=(10, 4), sharey = True)

    for i, screen in enumerate(screens):
        # average response across recordings, repeats, across cells

        for i_group, group_name in enumerate(groups):

            if "MWopto" in group_name:
                if 'asleep' in group_name:
                    colour = 'cornflowerblue'
                    col = 1
                elif 'awake' in group_name:
                    colour = 'darkblue'
                    col = 0
            elif "RD1" in group_name:
                if 'asleep' in group_name:
                    colour = 'salmon'
                    col = 1
                elif 'awake' in group_name:
                    colour = 'firebrick'
                    col = 0
            elif "GCaMP6s" in group_name:
                if 'asleep' in group_name:
                    colour = 'grey'
                    col = 1
                elif 'awake' in group_name:
                    colour = 'black'
                    col = 0

            group_dat = np.array(list(chain.from_iterable(
                [object.dat[animal][day][sub_file]['zscored_matrix_baseline'][1:, object.dat[animal][day][sub_file]['thresholded_cells'] == 1, :min_timepoints].mean(axis=(0, 1))
                 for sub_file in object.dat[animal][day].keys() if (('chirp' in sub_file) and (screen in sub_file))]
                for animal in object.dat.keys() if (group_name in animal)
                for day in object.dat[animal].keys())))
            #
            # group_cells_responsive = np.array(list(chain.from_iterable(
            #     [np.round(100 * object.dat[animal][day][sub_file]['thresholded_cells'].sum() / len(
            #         object.dat[animal][day][sub_file]['thresholded_cells']), 2)
            #      for sub_file in object.dat[animal][day].keys() if (('chirp' in sub_file) and (screen in sub_file))]
            #     for animal in object.dat.keys() if (group_name in animal)
            #     for day in object.dat[animal].keys()))).mean()
            group_cells_responsive = np.array(list(chain.from_iterable(
                [object.dat[animal][day][sub_file]['thresholded_cells'].sum()
                 for sub_file in object.dat[animal][day].keys() if (('chirp' in sub_file) and (screen in sub_file))]
                for animal in object.dat.keys() if (group_name in animal)
                for day in object.dat[animal].keys()))).mean()

            ax[i, col].plot(np.arange(min_timepoints), np.nanmean(group_dat, axis=0), color=colour, alpha = 0.7, label=f'{group_name} ({np.round(group_cells_responsive, 2)} % responsive)')
            ax[i, col].fill_between(np.arange(min_timepoints), np.nanmean(group_dat, axis=0) - stats.sem(group_dat, axis=0, nan_policy = 'omit'),np.nanmean(group_dat, axis=0) + stats.sem(group_dat, axis=0, nan_policy = 'omit'), color = colour, alpha = 0.4)

            ax[i, col].axvline(object.fps * 0, c='grey', alpha = 0.5)
            ax[i, col].axvline(object.fps * 4, c='black', alpha = 0.5)
            ax[i, col].axvline(object.fps * 8, c='black', alpha = 1, linestyle = '--')
            ax[i, col].axvline(object.fps * 12, c='black', alpha = 0.5)
            ax[i, col].axvline(object.fps * 16, c='grey', alpha = 0.5)
            ax[i, col].set_title('Ultrabright Monitor' if screen == 'small' else 'Regular Monitor')
            ax[i, col].set_xlabel('Time since stimulus onset (s)')
            ax[i, col].set_ylabel('Z-scored response')
            ax[i, col].set_xticks(np.arange(min_timepoints)[::4 * object.fps],
                             (np.arange(min_timepoints) // object.fps)[::4 * object.fps])
            ax[i, col].legend(prop={'size': 7})
            ax[i, col].axvspan(data_object.fps * 0, data_object.fps * 4, color='grey', alpha=0.10)
            ax[i, col].axvspan(data_object.fps * 4, data_object.fps * 8, color='black', alpha=0.15)
            ax[i, col].axvspan(data_object.fps * 12, data_object.fps * 16, color='black', alpha=0.15)
            ax[i, col].axvspan(data_object.fps * 16, data_object.fps * 20, color='grey', alpha=0.10)

    plt.suptitle('Full-Field Flash response')
    save_fig(object, 'Full-Field Flash', 'average on-off responses')
    plt.show()

avg_onoff_response(data_object)
