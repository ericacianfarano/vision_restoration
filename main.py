import matplotlib.pyplot as plt

from classes import *
#from classes_olfaction import *

animals_days = {'EC_GCaMP6s_07': ['20240927'], 'EC_GCaMP6s_08': ['20240926']}           #simple  8 orientation x 6 repeats
animals_days = {'EC_GECO920_17': ['20241004']} #                 control geco phpeb animal (seeing) imaged at 920           'EC_GCaMP6s_11': ['20241009', '20241113'],

# To-do:
#- raster plots for SFxO
# organize SFs according

animals_days = {'EC_MWopto_04': ['20241111', '20241112', '20241113','20241120', '20241129'],
                'EC_MWopto_02': ['20241120'],
                'EC_MWopto_05': ['20241127'],
                'EC_MWopto_03': ['20241128'],
                'EC_MWopto_08': ['20250306'],
                'EC_MWopto_10': ['20250306', '20250311'],
                'EC_GCaMP6s_05': ['20240925'],
                'EC_GCaMP6s_06': ['20240925', '20241113', '20241121'],
                'EC_GCaMP6s_08': ['20240927'], #0927
                'EC_GCaMP6s_09': ['20241113', '20241122'],
                'EC_RD1_05': ['20241122'],  # '20240927', '20241116',
                'EC_RD1_06': ['20250108'], #['20241002', '20241116']
                #'EC_RD1_07': ['20241002','20241003','20241116'],
                'EC_RD1_08': ['20250108'], #['20241003','20241116'],
                'EC_RD1_09': ['20250110'], #['20241007','20241113'],
                'EC_RD1_10': ['20250110'], #['20241007','20241113'],
                }

#
# animals_days = {'EC_GNAT_03': ['20240924'],
#                 'EC_GNAT_05': ['20240924'],
#                 'EC_GNAT_06': ['20240924'],
#                 'EC_GNAT_04': ['20241004'],
#                 #'EC_GCaMP6s_11': ['20241113'],
#                 'EC_GCaMP6s_12': ['20250123'],
#                 'EC_GCaMP6s_13': ['20250123'],
#                 }

plt.figure()
thresh = data_object.dat['EC_GCaMP6s_06']['20241121']['big100_chirps_000_000']['thresholded_cells']
arr = data_object.dat['EC_GCaMP6s_06']['20241121']['big100_chirps_000_000']['zscored_responses_ttls'][:,thresh==1,:]
onoff = data_object.dat['EC_GCaMP6s_06']['20241121']['big100_chirps_000_000']['on_off_index'][thresh==1]
off_idx, on_idx, zero_idx = np.argwhere(onoff == onoff.min())[0][0], np.argwhere(onoff == onoff.max())[0][0], np.argwhere(onoff == -np.abs(onoff).min())[0][0]

plt.plot (arr[:, off_idx, :].mean(axis = 0), c = 'red', label = 'idx ~= -1 (off)')
plt.plot (arr[:, on_idx, :].mean(axis = 0), c = 'blue', label = 'idx ~= 1 (on)')
plt.plot (arr[:, zero_idx, :].mean(axis = 0), c = 'green', label = 'idx ~= 0')
plt.axvline(data_object.fps*4)
plt.axvline(data_object.fps*8)
plt.axvline(data_object.fps*12)
plt.axvline(data_object.fps*16)
plt.legend()
plt.show()


#animals_days = {'EC_GCaMP6s_06': ['20241121']}
data_object = DataAnalysis (['H:\\vision_restored', 'G:\\vision_restored'], dict_animals_days = animals_days, response_type = 'fluorescence', dlc = False, show_plots = False)

# full field flash response
plot_on_off_index(data_object, thresholded_cells = True)
avg_onoff_response(data_object)
avg_onoff_response_groups(data_object)
plot_response_speed(data_object, thresholded_cells = True)
plot_response_amplitude(data_object, thresholded_cells = True)

# olfaction stuff
animals_days = {'EC_RD1_06': ['20250223'], 'EC_RD1_08': ['20250223']}           #simple  8 orientation x 6 repeats

data_object = OlfactionAnalysis ('I:\\olfaction', dict_animals_days = animals_days, zscore_threshold=3, response_type = 'fluorescence', dlc = False, show_plots = False)

arr = data_object.dat['EC_RD1_06']['20250223']['olf_v1_000_000']['responses']
ttl = data_object.dat['EC_RD1_06']['20250223']['olf_v1_000_000']['ttl_data']

ttl_responses = np.array([arr [:, t:t+data_object.fps*4] for t in ttl])

############### ENSEMBLES

animals_days = {'EC_RD1opto_04': ['20241113'],
                'EC_RD1opto_02': ['20241120'],
                'EC_RD1opto_05': ['20241127'],
                'EC_RD1opto_03': ['20241128'],
                'EC_GCaMP6s_06': ['20240925'],# '20241113', '20241121'],
                'EC_GCaMP6s_08': ['20240927'], #0927
                'EC_GCaMP6s_09': ['20241113', '20241122'],
                'EC_RD1_05': ['20241122'],  # '20240927', '20241116',
                'EC_RD1_06': ['20250108'], #['20241002', '20241116']
                #'EC_RD1_07': ['20241002','20241003','20241116'],
                'EC_RD1_08': ['20250108'], #['20241003','20241116'],
                'EC_RD1_09': ['20250110'], #['20241007','20241113'],
                'EC_RD1_10': ['20250110'], #['20241007','20241113'],
                }

# if not os.path.exists(os.path.join(data_object.save_path, 'embedded population vectors')):
#     os.makedirs(os.path.join(data_object.save_path, 'embedded population vectors'))

with PdfPages(os.path.join(data_object.save_path, 'population vectors.pdf')) as pdf:

    for animal in animals_days.keys():

        fig, ax = plt.subplots(2, 2, figsize=(11, 9), sharex = True, sharey = True)
        days_recordings = [(day, subfile) for day in animals_days[animal] for subfile in data_object.dat[animal][day].keys() if 'SFxTF' in subfile]

        # for each recording, stack the first (2 mins) and last (15 mins) wait periods for the big & small screens
        # result is list with n_recordings elements, each with shape n_cells, n_timepoints
        wait_data = [np.hstack([np.squeeze(data_object.dat[animal][day][subfile]['zscored_responses_ttls'][stim]) for stim in
                       data_object.dat[animal][day][subfile]['zscored_responses_ttls'].keys() if 'Wait' in stim]) for (day, subfile) in days_recordings]

        grating_data = [np.hstack([data_object.dat[animal][day][subfile]['zscored_responses_ttls'][stim].reshape(-1, data_object.dat[
            animal][day][subfile]['zscored_responses_ttls'][stim].shape[0] * data_object.dat[animal][day][subfile]['zscored_responses_ttls'][stim].shape[-1])
                        for stim in data_object.dat[animal][day][subfile]['zscored_responses_ttls'].keys() if 'Grating' in stim]) for (day, subfile) in days_recordings]

        for i_stim, stim in enumerate(['wait', 'grating']):

            if stim == 'wait':
                data = wait_data
            elif stim == 'grating':
                data = grating_data

            # enumerating through big/small screen
            for i, (day, subfile) in enumerate(days_recordings):

                colors = plt.cm.plasma(np.linspace(0, 0.9, data[i].shape[-1]))

                # Binarize data > 1 where activity is above the 95th percentile
                threshold = np.percentile(data[i], 95)  # Example threshold
                binary_data = (data[i] > threshold).astype(int)

                # A population vector at time t is a binary vector (shape n_neurons) representing the activity of all neurons at that time.
                # dimension = # neurons
                population_vectors = binary_data.T  # n_timepoints x n_neurons

                # 3) Identify significant vectors > test significance of each population vector against a null hypothesis (random activity).

                # Randomly shuffle the activity across neurons while preserving the temporal structure.
                shuffled_data = np.random.permutation(binary_data.flatten()).reshape(binary_data.shape)

                # Calculate the probability of observing the number of active neurons in the real data compared to shuffled data.
                active_counts_real = np.sum(binary_data, axis=0)  # sum across neurons (for each timepoint)
                active_counts_shuffled = np.sum(shuffled_data, axis=0)  # sum across neurons (for each timepoint)
                p_values = np.mean(active_counts_real[:, None] <= active_counts_shuffled, axis=1)

                # Retain only those population vectors where p <0.05 --> these are significantly different from random activity
                significant_vectors = population_vectors[p_values < 0.05]
                colors = colors[p_values < 0.05]

                #colors = plt.cm.plasma(np.linspace(0, 0.9, significant_vectors.shape[0]))

                # Step 4: Dimensionality reduction with PCA
                # Project the high-dimensional population vectors into a lower-dimensional space.
                pca = PCA(n_components=2)
                reduced_vectors = pca.fit_transform(significant_vectors)

                # Each dot in the reduced space corresponds to a population vector, with clusters representing similar patterns of neuronal activity.
                # each dot in the reduced dimensional space represents a population vector and clusters of vectors therefore define a given neuronal ensemble.
                ax[i_stim,i].scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c=colors, alpha=0.6)
                ax[i_stim,i].set_title((stim, subfile))
                ax[i_stim,i].set_xlabel('PC1')
                ax[i_stim,i].set_ylabel('PC2')
                ax[i_stim, i].set_xticks([])
                ax[i_stim, i].set_yticks([])

        plt.suptitle(f'Neuronal Ensembles ({animal})')
        #plt.show()
        pdf.savefig(fig)
        plt.close(fig)

with PdfPages(os.path.join(data_object.save_path, 'pca_projection.pdf')) as pdf:

    for i_animal, animal in enumerate(animals_days.keys()):

        # list of tuples, with each entry being (day, subfile)
        days_recordings = [(day, subfile) for day in animals_days[animal] for subfile in data_object.dat[animal][day].keys()if 'SFxO' in subfile][:2]

        for i, (day, subfile) in enumerate(days_recordings):

            # param_matrix = data_object.dat[animal][day][subfile]['param_matrix_whole_zscore'] # shape n_repeats, n_orientations, n_sfs, n_cells, n_timepoints
            # n_repeats, n_orientations, n_sfs, n_cells, n_timepoints = param_matrix.shape
            # data_trials = param_matrix.reshape(n_repeats * n_orientations * n_sfs, n_cells * n_timepoints) # shape (n_trials, n_features)

            # shape n_repeats, n_orientations, n_cells, n_timepoints
            param_matrix = data_object.dat[animal][day][subfile]['param_matrix_whole_zscore'][:,:,3]
            n_repeats, n_orientations, n_cells, n_timepoints = param_matrix.shape

            # shape (n_repeats x n_orientations, n_cells x n_timepoints) > (n_trials, n_features)
            data_trials = param_matrix.reshape(n_repeats * n_orientations, n_cells * n_timepoints)

            # Normalize each trial's activity pattern (vector) to unit length (L2 norm)
            # each trial (each row) is normalized to unit length (its L2 norm is 1)
            norms = np.linalg.norm(data_trials, axis=1, keepdims=True)
            data_trials_normalized = data_trials / norms

            # performing PCA & projection
            pca, principal_components, explained_variance_ratio = perform_pca (data_trials_normalized, n_components = 10)

            # calculating participation ratio (dimensionality)
            pr = participation_ratio(pca)

            screen = 'reg' if 'big' in subfile else 'ultrabright'
            plot_pca_projection(animal + ', ' + screen, principal_components, n_repeats, n_orientations)
            pdf.savefig(fig)
            plt.close(fig)

            # fig2 = plot_variance_explained(animal + ', ' + screen, explained_variance_ratio)
            # pdf.savefig(fig2)
            # plt.close(fig2)

            #plot_pca_projection_ax(ax, animal, principal_components, n_repeats, n_orientations)
            #plot_variance_explained(animal, explained_variance_ratio)

def filter_active_epochs (data_trials_array, response_threshold = 3, cell_threshold = 4):
    '''
    Remove epochs with no activity or only 1-2 cells active
    :param data_trials_array: numpy array of shape (n_cells, n_chunks)
    :param threshold: z-score threshold that cells need to be to be considered active
    :return:
    '''
    # check whether each cell goes above response threshold in each epoch
    # then count number of cells (per epoch) that exceed response threshold (shape n_epochs)
    cells_above_threshold = np.any(data_trials_array > response_threshold, axis=-1).sum(axis = -1)

    # valid epochs are only those with > 2 cells co-active (shape n_epochs)
    valid_epochs = cells_above_threshold >= cell_threshold

    return valid_epochs



chunk_size = 10 # n frames

#with PdfPages(os.path.join(data_object.save_path, 'pca_projection_spon.pdf')) as pdf:
var_explained = {'Control': [], 'RD1':[]}

for i_animal, animal in enumerate(animals_days.keys()):

    group = 'Control' if 'GCaMP6s' in animal else 'RD1'

    # list of tuples, with each entry being (day, subfile)
    days_recordings = [(day, subfile) for day in animals_days[animal] for subfile in data_object.dat[animal][day].keys() if ('SFxTF' in subfile) and ('big' in subfile)][:2]

    for i, (day, subfile) in enumerate(days_recordings):
        print(animal, day, subfile)

        # shape n_Cells, n_timepoints
        spon_arr = np.squeeze(data_object.dat[animal][day][subfile]['zscored_responses_ttls']['Wait_1'])
        n_full_chunks = spon_arr.shape[1] // chunk_size

        # shape > (n_trials, n_features, n_timepoints_per_epoch) (n_epochs, n_cells, n_timepoints)
        #data_trials = np.array([spon_arr[:,i:i + chunk_size] for i in range(0, n_full_chunks * chunk_size, chunk_size)]).reshape (n_full_chunks, -1)
        data_trials_all = np.array([spon_arr[:, i:i + chunk_size] for i in range(0, n_full_chunks * chunk_size, chunk_size)])

        # only take epochs that have at least 'cell_threshold' co-active cells that each go above 'response_threshold' > (n_epochs, n_cells, n_timepoints)

        valid_epochs = filter_active_epochs(data_trials_all[:400], response_threshold=3, cell_threshold = 4)

        plot_raster(data_trials_all, valid_epochs, n_trials_to_plot=60, n_cells_to_plot=400, threshold=4)

        #####################

        # then average over each epoch (time) to get average response> (n_epochs, n_cells)
        data_trials = data_trials.mean (axis = -1)

        print(f'{animal}, {day}, {data_trials.shape[0]/data_trials_all.shape[0]*100}% of trials have >4 coactive cells')

        # Normalize each trial's activity pattern (vector) to unit length (L2 norm)
        # each trial (each row) is normalized to unit length (its L2 norm is 1)
        norms = np.linalg.norm(data_trials, axis=1, keepdims=True)
        data_trials_normalized = data_trials / norms

        # performing PCA & projection
        pca, principal_components, explained_variance_ratio = perform_pca (data_trials_normalized, n_components = 10)

        # calculating participation ratio (dimensionality)
        #pr = participation_ratio(pca)

        var_explained[group].append(explained_variance_ratio)

        # screen = 'reg' if 'big' in subfile else 'ultrabright'
        # n_repeats = None
        # n_orientations = None
        # # fig = plot_pca_projection(animal + ', ' + screen, principal_components, n_repeats, n_orientations, trial_type = 'spontaneous')
        # # pdf.savefig(fig)
        # # plt.close(fig)
        #
        # #plot_variance_explained(animal, explained_variance_ratio)
        #
        # if 'opto' not in animal:
        #     color = 'red' if 'RD1' in animal else 'blue'
        #     plt.plot(np.arange(1, len(explained_variance_ratio) + 1), explained_variance_ratio, c = color,alpha = 0.6, marker='o')

variance_explained (var_explained)

n_trials_plot = 15
n_cells_plot = 100
# take the first n_trials_plot trials
data_trials = data_trials_all[20:20+n_trials_plot, :n_cells_plot]
valid_epochs_short = valid_epochs[20:20+n_trials_plot]

stacked_trials = data_trials.reshape((data_trials.shape[0]*data_trials.shape[-1], -1)).shape

def binarize_array(data, threshold=4):
    return (data >= threshold).astype(int)

def plot_raster(data_trials_array, valid_epochs_arr, n_trials_to_plot = 20, n_cells_to_plot = 100, threshold=4):

    # take the first n_trials_plot trials > shape n_epochs, n_cells, n_timepoints
    dat = data_trials_array[:n_trials_to_plot, :n_cells_to_plot]
    valid_epochs_short = valid_epochs_arr[:n_trials_to_plot]

    # shape n_cells, n_epochs x n_timepoints
    stacked_trials = np.hstack([dat[epoch] for epoch in range(dat.shape[0])])#dat.reshape((dat.shape[0]*dat.shape[-1], -1)).T
    bin_stacked = binarize_array(stacked_trials, threshold=threshold)

    plt.figure(figsize = (10,5))
    plt.imshow(bin_stacked, cmap = 'Greys')
    for epoch in range(dat.shape[0]):
        if valid_epochs_short[epoch]:
            plt.axvspan(epoch * dat.shape[-1], (epoch + 1) * dat.shape[-1], color='grey', alpha=0.2)
            #print(epoch * dat.shape[-1], (epoch + 1) * dat.shape[-1])
    plt.xticks(np.arange(0,stacked_trials.shape[-1],data_object.fps*5), np.arange(0,stacked_trials.shape[-1],data_object.fps*5)/data_object.fps)
    plt.xlabel('Time (s)')
    plt.ylabel('Cell Number')
    plt.show()

plot_raster(data_trials_all, valid_epochs, n_trials_to_plot = 20, n_cells_to_plot = 100, threshold=4)

def plot_raster_with_valid_times(data_trials, valid_epochs, threshold=4):
    """
    Plots a raster with spikes where the response exceeds the threshold.
    Grey vertical bars highlight timepoints where the epoch is valid.

    :param data_trials: numpy array (n_epochs, n_cells, n_timepoints)
    :param valid_epochs: boolean array (n_epochs,) indicating valid epochs
    :param threshold: response threshold for detecting activity
    """
    n_epochs, n_cells, n_timepoints = data_trials.shape

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    for epoch in range(n_epochs):
        # Identify active cells and their time points
        active_cells, active_times = np.where(data_trials[epoch] > threshold)

        # Offset y-values by epoch to separate trials
        ax.scatter(active_times, active_cells + epoch * n_cells, s=1, color='black')

        # Highlight valid time points with grey bars
        if valid_epochs[epoch]:
            ax.axvspan(epoch * n_timepoints, (epoch + 1) * n_timepoints, color='grey', alpha=0.2)
        print(epoch * n_timepoints, (epoch + 1) * n_timepoints)

    ax.set_xlabel("Time (frames)")
    ax.set_ylabel("Cells (grouped by epoch)")
    ax.set_title("Raster Plot with Valid Epochs Highlighted")
    plt.show()


plot_raster_with_valid_times(data_trials, valid_epochs_short, threshold=4)


plt.figure()
plt.imshow(arr[:100], cmap = 'Greys')
[plt.axvline(line, c = 'red', alpha = 0.5) for line in np.arange(0,arr.shape[-1], chunk_size)]
plt.show()


plt.figure(figsize=(12, 8))
for epoch_idx, (epoch, is_valid) in enumerate(zip(data_trials, valid_epochs)):
    color = 'red' if is_valid else 'black'

    cells, timepoints = np.where(epoch > 4)         # Find spike locations (nonzero values)
    timepoints += epoch_idx * epoch.shape[1] # Adjust timepoints to stack epochs horizontally
    plt.scatter(timepoints, cells, color=color, s=1) # plot spikes

plt.xlabel('Time (stacked epochs)', fontsize=12)
plt.ylabel('Cells', fontsize=12)
plt.title('Raster Plot: Valid (Red) vs Invalid (Black)', fontsize=14)
plt.show()

# Create raster plot
fig, ax = plt.subplots(figsize=(15, 8))

# Get spike times (list of time indices for each cell)
spike_times = [np.where(data_trials[cell])[0] for cell in range(n_cells)]

# Plot raster (each row represents a different cell)
ax.eventplot(spike_times, linelengths=0.8, linewidths=0.5, color="black")

# Add vertical lines at epoch boundaries
for i in range(1, n_epochs):
    ax.axvline(i * epoch_length, color="red", linestyle="--", linewidth=1.5)

# Labels and formatting
ax.set_xlabel("Time (bins)")
ax.set_ylabel("Cell index")
ax.set_title("Raster Plot with 5 Epochs Stacked Horizontally")
ax.invert_yaxis()  # Ensure cell 0 is at the top

plt.tight_layout()
plt.show()


def variance_explained (var_explained_dictionary):
    '''
    Plot variance explained of the different groups in dictionary 'var_explained_dictionary' (Control & RD1)

    Calculate the T-test for the means of two independent samples of scores (welche's t test)
    to see if the first PCs in each group are statistically significant

    '''

    plt.figure(figsize=(8, 5))
    control, rd1 = np.array(var_explained_dictionary['Control']),  np.array(var_explained_dictionary['RD1'])
    control_mean, rd1_mean = control.mean(axis = 0), rd1.mean(axis = 0)
    control_sem, rd1_sem = sem(control, axis = 0), sem(rd1, axis = 0)

    plt.plot(control.T, alpha = 0.3, c = 'blue')
    plt.plot(control_mean, alpha = 0.8, c = 'blue', linewidth = 2.5, label = 'Control')
    plt.plot(rd1.T, alpha = 0.3, c = 'red')
    plt.plot(rd1_mean, alpha = 0.8, c = 'red', linewidth = 2.5, label = 'RD1')
    plt.fill_between(np.arange(control.shape[1]), control_mean - control_sem, control_mean + control_sem, color='blue', alpha=0.2)
    plt.fill_between(np.arange(rd1.shape[1]), rd1_mean - rd1_sem, rd1_mean + rd1_sem, color='red', alpha=0.2)
    plt.xticks([])
    plt.yticks([])

    plt.ylim([min(control_mean.min(), rd1_mean.min() - 0.005), max(control_mean.max(), rd1_mean.max()) + 0.02])

    # welch's t tests on first PC
    control_pc1 = np.array([arr[0] for arr in var_explained_dictionary['Control']])
    rd1_pc1 = np.array([arr[0] for arr in var_explained_dictionary['RD1']])
    t_stat, p_value = ttest_ind(control_pc1, rd1_pc1, equal_var=False)  # Welch's t-test (unequal variance assumption)
    print(f"T-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")
    if p_value < 0.05:
        plt.text(0, max(control_mean[0], rd1_mean[0]) + 0.01, '*', fontsize=20, ha='center')

    plt.legend(fontsize = 14)
    plt.xlabel('PC index')
    plt.ylabel('Variance explained')
    plt.title(f'Variance explained by PCs')
    plt.show()



def plot_pca_projection(animal, principal_components, n_repeats, n_orientations, trial_type = 'evoked'):
    """
    Plots the projection of trials onto the first two principal components,
    with color indicating stimulus orientation.

    Args:
        animal (str): Animal identifier.
        principal_components (np.ndarray): Principal components of shape (n_trials, n_components).
        n_repeats (int): Number of stimulus repeats.
        n_orientations (int): Number of stimulus orientations.
        trial_type (str): either 'evoked' or 'spontaneous'
    """

    if trial_type == 'evoked':
        # Assign colors according to orientation
        orientations = np.linspace(0, 315, n_orientations)  # 0, 45, 90, ..., 315
        orientation_ids = np.repeat(np.arange(n_orientations), n_repeats)
        colors = orientations[orientation_ids]

        fig = plt.figure(figsize=(5, 4))
        #ax = fig.add_subplot(111, projection = '3d')
        ax = fig.add_subplot(111)

        scatter = ax.scatter(
            principal_components[:, 0],
            principal_components[:, 1],
            #principal_components[:, 2],
            c=colors,
            cmap='plasma',
            alpha=0.8
        )

        # color bar showing orientation
        cbar = plt.colorbar(scatter, ax=ax, ticks=orientations)
        cbar.set_label('Orientation (degrees)')
        cbar.set_ticks(orientations)
        cbar.set_ticklabels([f'{int(o)}°' for o in orientations])


    elif trial_type == 'spontaneous':

        fig = plt.figure(figsize=(5, 4))
        #ax = fig.add_subplot(111, projection='3d')
        ax = fig.add_subplot(111)

        scatter = ax.scatter(
            principal_components[:, 0],
            principal_components[:, 1],
            #principal_components[:, 2],
            c=np.arange(principal_components.shape[0]),
            cmap='plasma',
            alpha=0.8
        )

    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_title(f'{animal} - PCA Projection \n of single-trial grating-evoked activity')

    plt.show()
    return fig

# ------------------------
# Variance Explained Curve (for plot b)
# ------------------------
def plot_variance_explained(animal, explained_variance_ratio):
    plt.figure(figsize=(5, 3))
    plt.plot(np.arange(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o')
    plt.xlabel('PC index')
    plt.ylabel('Variance explained')
    plt.title(f'Variance explained by PCs ({animal})')
    return fig


        # for each recording, stack the first (2 mins) and last (15 mins) wait periods for the big & small screens
        # result is list with n_recordings elements, each with shape n_cells, n_timepoints
        # wait_data = [np.hstack([np.squeeze(data_object.dat[animal][day][subfile]['zscored_responses_ttls'][stim]) for stim in
        #                data_object.dat[animal][day][subfile]['zscored_responses_ttls'].keys() if 'Wait' in stim]) for (day, subfile) in days_recordings]

        grating_data = [np.hstack([data_object.dat[animal][day][subfile]['zscored_responses_ttls'][stim].reshape(-1, data_object.dat[
            animal][day][subfile]['zscored_responses_ttls'][stim].shape[0] * data_object.dat[animal][day][subfile]['zscored_responses_ttls'][stim].shape[-1])
                        for stim in data_object.dat[animal][day][subfile]['zscored_responses_ttls'].keys() if 'Grating' in stim]) for (day, subfile) in days_recordings]

        #for i_stim, stim in enumerate(['wait', 'grating']):
        for i_stim, stim in enumerate([ 'grating']):

            if stim == 'wait':
                data = wait_data
            elif stim == 'grating':
                data = grating_data

            # enumerating through big/small screen
            for i, (day, subfile) in enumerate(days_recordings):

                colors = plt.cm.plasma(np.linspace(0, 0.9, data[i].shape[-1]))

                print(data[i].shape)

                # Binarize data > 1 where activity is above the 95th percentile
                threshold = np.percentile(data[i], 95)  # Example threshold
                binary_data = (data[i] > threshold).astype(int)

                # A population vector at time t is a binary vector (shape n_neurons) representing the activity of all neurons at that time.
                # dimension = # neurons
                population_vectors = binary_data.T  # n_timepoints x n_neurons

                # 3) Identify significant vectors > test significance of each population vector against a null hypothesis (random activity).

                # Randomly shuffle the activity across neurons while preserving the temporal structure.
                shuffled_data = np.random.permutation(binary_data.flatten()).reshape(binary_data.shape)

                # Calculate the probability of observing the number of active neurons in the real data compared to shuffled data.
                active_counts_real = np.sum(binary_data, axis=0)  # sum across neurons (for each timepoint)
                active_counts_shuffled = np.sum(shuffled_data, axis=0)  # sum across neurons (for each timepoint)
                p_values = np.mean(active_counts_real[:, None] <= active_counts_shuffled, axis=1)

                # Retain only those population vectors where p <0.05 --> these are significantly different from random activity
                significant_vectors = population_vectors[p_values < 0.05]
                colors = colors[p_values < 0.05]

                #colors = plt.cm.plasma(np.linspace(0, 0.9, significant_vectors.shape[0]))

                # Step 4: Dimensionality reduction with PCA
                # Project the high-dimensional population vectors into a lower-dimensional space.
                pca = PCA(n_components=2)
                reduced_vectors = pca.fit_transform(significant_vectors)

                # Each dot in the reduced space corresponds to a population vector, with clusters representing similar patterns of neuronal activity.
                # each dot in the reduced dimensional space represents a population vector and clusters of vectors therefore define a given neuronal ensemble.
                ax[i_stim,i].scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c=colors, alpha=0.6)
                ax[i_stim,i].set_title((stim, subfile))
                ax[i_stim,i].set_xlabel('PC1')
                ax[i_stim,i].set_ylabel('PC2')
                ax[i_stim, i].set_xticks([])
                ax[i_stim, i].set_yticks([])

        plt.suptitle(f'Embeddings ({animal})')
        #plt.show()
        pdf.savefig(fig)
        plt.close(fig)


data = np.squeeze(data_object.dat['EC_RD1_10']['20250110']['big100_SFxO_000_007']['zscored_responses_ttls']['Wait_0'])
data = np.squeeze(data_object.dat['EC_GCaMP6s_09']['20241122']['small_SFxO_000_012']['zscored_responses_ttls']['Wait_0'])
data = np.squeeze(data_object.dat['EC_RD1opto_03']['20241128']['big100_SFxO_000_005']['zscored_responses_ttls']['Wait_0'])

# Binarize data
threshold = np.percentile(data, 95)  # Example threshold
binary_data = (data > threshold).astype(int)

# A population vector at time t is a binary vector of size n neurons representing the activity of all neurons at that time.
population_vectors = binary_data.T  # n_timepoints x n_neurons

# 3) Identify significant vectors > test significance of each population vector against a null hypothesis (random activity).

# Randomly shuffle the activity across neurons while preserving the temporal structure.
shuffled_data = np.random.permutation(binary_data.flatten()).reshape(binary_data.shape)

# Calculate the probability of observing the number of active neurons in the real data compared to shuffled data.
active_counts_real = np.sum(binary_data, axis=0)        # sum across neurons (for each timepoint)
active_counts_shuffled = np.sum(shuffled_data, axis=0)  # sum across neurons (for each timepoint)
p_values = np.mean(active_counts_real[:, None] <= active_counts_shuffled, axis=1)

# Retain only those population vectors where p <0.01.
significant_vectors = population_vectors[p_values < 0.05]

# Step 4: Dimensionality reduction with PCA
# Project the high-dimensional population vectors into a lower-dimensional space.
pca = PCA(n_components=3)
reduced_vectors = pca.fit_transform(significant_vectors)

# Each dot in the reduced space corresponds to a population vector, with clusters representing similar patterns of neuronal activity.
# each dot in the reduced dimensional space represents a population vector and clusters of vectors therefore define a given neuronal ensemble.
plt.figure(figsize=(8, 8))
plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c='b', alpha=0.6)
plt.title('Neuronal Ensembles in Reduced Dimensional Space')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()




stim = 'SFxTF'
max_n_neurons = np.array([data_object.dat[animal][day][sub_file]['zscored_responses'].shape[0]
                            for animal in data_object.dat
                            for day in data_object.dat[animal]
                            for sub_file in data_object.dat[animal][day]]).min()# if stim in sub_file]).min()

specs = np.array([(animal,day,sub_file)
                            for animal in data_object.dat
                            for day in data_object.dat[animal]
                            for sub_file in data_object.dat[animal][day]])# if stim in sub_file])

zscored_responses_list = [data_object.dat[animal][day][sub_file]['zscored_responses'][:max_n_neurons,:]
                            for animal in data_object.dat
                            for day in data_object.dat[animal]
                            for sub_file in data_object.dat[animal][day]]# if stim in sub_file]

def flatten_upper_triangle(matrix):
    return matrix[np.triu_indices_from(matrix, k=1)]

# correlation matrices for each subject
correlation_matrices = [np.corrcoef(subject) for subject in zscored_responses_list]
#Flatten the upper triangular part of each correlation matrix
correlation_vectors = [flatten_upper_triangle(corr_mat) for corr_mat in correlation_matrices]  # List of 1D arrays
# Compute pairwise distances
distance_matrix = pairwise_distances(correlation_vectors, metric='euclidean')

mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
embedding = mds.fit_transform(distance_matrix)

group_indices = defaultdict(list)
for idx, row in enumerate(specs):
    key = tuple(row[:1])  # Use the first two elements (animal, day) as the key
    group_indices[key].append(idx)
grouped_indices = list(group_indices.values())

#coloring according to monitor
big_indices = [i for i, row in enumerate(specs) if 'big' in row[2]]
small_indices = [i for i, row in enumerate(specs) if 'small' in row[2]]
plt.figure(figsize=(8, 6))
sns.scatterplot(x=embedding[big_indices, 0], y=embedding[big_indices, 1], s=100, c = 'red', label = 'dim')
sns.scatterplot(x=embedding[small_indices, 0], y=embedding[small_indices, 1], s=100, c='blue', label = 'ultrabright')
plt.title("MDS of Correlation Structures")
plt.xlabel("MDS Dimension 1")
plt.ylabel("MDS Dimension 2")
plt.legend()
plt.show()

#coloring according to layer
big_indices = [i for i, row in enumerate(specs) if 'l4' not in row[2]]
small_indices = [i for i, row in enumerate(specs) if 'l4' in row[2]]
plt.figure(figsize=(8, 6))
sns.scatterplot(x=embedding[big_indices, 0], y=embedding[big_indices, 1], s=100, c = 'red', label = 'l23')
sns.scatterplot(x=embedding[small_indices, 0], y=embedding[small_indices, 1], s=100, c='blue', label = 'l4')
plt.title("MDS of Correlation Structures")
plt.xlabel("MDS Dimension 1")
plt.ylabel("MDS Dimension 2")
plt.legend()
plt.show()

# coloring according to group
control_indices = [i for i, row in enumerate(specs) if 'GCaMP6s' in row[0]]
rd1_indices = [i for i, row in enumerate(specs) if 'RD1' in row[0]]
opto_indices = [i for i, row in enumerate(specs) if 'opto' in row[0]]
plt.figure(figsize=(8, 6))

for i, animal in enumerate(grouped_indices):
    group = list(group_indices.keys())[i][0]
    #print(group)
    if 'GCaMP6s' in group:
        colour = 'red'
    elif 'RD1' in group and 'opto' not in group:
        colour = 'blue'
    elif 'opto' in group:
        colour = 'purple'
    #print(colour)
    plt.plot(embedding[animal,0], embedding[animal,1], c = colour)
sns.scatterplot(x=embedding[control_indices, 0], y=embedding[control_indices, 1], s=100, c = 'red', label = 'control')
sns.scatterplot(x=embedding[rd1_indices, 0], y=embedding[rd1_indices, 1], s=100, c='blue', label = 'rd1')
sns.scatterplot(x=embedding[opto_indices, 0], y=embedding[opto_indices, 1], s=100, c='purple', label = 'opto')
plt.title("MDS of Correlation Structures")
plt.xlabel("MDS Dimension 1")
plt.ylabel("MDS Dimension 2")
plt.legend()
plt.show()

mds = MDS(n_components=3, dissimilarity='precomputed', random_state=42)
embedding = mds.fit_transform(distance_matrix)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(embedding[control_indices, 0], embedding[control_indices, 1], embedding[control_indices, 2], s=100, c = 'red', label = 'control')
ax.scatter(embedding[rd1_indices, 0], embedding[rd1_indices, 1], embedding[rd1_indices, 2], s=100, c='blue', label = 'rd1')
ax.scatter(embedding[opto_indices, 0], embedding[opto_indices, 1], embedding[opto_indices, 2], s=100, c='purple', label = 'opto')
ax.set_title("MDS of Correlation Structures")
ax.set_xlabel("dim 1")
ax.set_ylabel("dim 2")
ax.set_zlabel("dim 3")
plt.show()


[[data_object.dat['EC_GCaMP6s_09'][day][subfile]['thresholded_cells_zscore3'] for subfile in data_object.dat['EC_GCaMP6s_09'][day].keys() if 'SFxO' in subfile ] for day in data_object.dat['EC_GCaMP6s_09'].keys()]


data_object = load(r'E:\vision_restored\figures\dat.z')

for animal in data_object.dat.keys():
    for day in data_object.dat[animal].keys():
        for sub_file in [key for key in data_object.dat[animal][day].keys() if 'chirp' in key]:
            # mean across repeats and cells
            response = data_object.dat[animal][day][sub_file]['responses_ttls'][1:].mean(axis=0).mean(axis=0)

            if 'opto' in animal:
                color = 'blue'
                alpha = 0.9
                zorder = 2
            else:
                color = 'red'
                alpha = 0.5
                zorder = 1
            plt.plot(range(response.shape[0]), response, c=color, alpha=alpha, zorder=zorder)
plt.show()



animals_DOB = { 'EC_RD1_05': '20240606',
                'EC_RD1_06': '20240704',
                'EC_RD1_07': '20240704',
                'EC_RD1_08': '20240704',
                'EC_RD1_09': '20240704',
                'EC_RD1_10': '20240704',
                }


for animal in data_object.dat.keys():
    if animal in animals_DOB:
        for day in data_object.dat[animal].keys():
            elapsed_days = (datetime.strptime(day, "%Y%m%d") - datetime.strptime(animals_DOB[animal], "%Y%m%d")).days
            for sub_file in data_object.dat[animal][day].keys():
                data_object.dat[animal][day][sub_file]['age_days'] = elapsed_days

plt.figure()
for animal in data_object.dat.keys():
    if animal in animals_DOB:
        for day in data_object.dat[animal].keys():
            for sub_file in data_object.dat[animal][day].keys():
                age = data_object.dat[animal][day][sub_file]['age_days']
                response_amplitude = data_object.dat[animal][day][sub_file]['responses_ttls'][1:].mean(axis=0).mean(axis=0).max() - data_object.dat[animal][day][sub_file]['responses_ttls'][1:].mean(axis=0).mean(axis=0).min()

                color = 'red'
                print(age, response_amplitude)
                plt.plot(age, response_amplitude, 'o' ,alpha = 0.8)
plt.show()

# UMAP CLUSTERING

# list with 40 elements, each with array of shape n_repeats, n_cells, n_timepoints
min_frames = np.array(list(chain.from_iterable(
    [data_object.dat[animal][day][sub_file]['zscored_responses'].shape[-1]
     for sub_file in data_object.dat[animal][day].keys() if (('SF' in sub_file) and ('big' in sub_file))]
    for animal in data_object.dat.keys()
    for day in data_object.dat[animal].keys()))).min()

data = np.vstack(list(chain.from_iterable(
    [data_object.dat[animal][day][sub_file]['zscored_responses'][data_object.dat[animal][day][sub_file]['thresholded_cells'], :min_frames]
     for sub_file in data_object.dat[animal][day].keys() if ('SF' in sub_file) and ('big' in sub_file)]
    for animal in data_object.dat.keys()
    for day in data_object.dat[animal].keys())))
animals = list(chain.from_iterable(
    [(animal, data_object.dat[animal][day][sub_file]['thresholded_cells'].sum())
     for sub_file in data_object.dat[animal][day].keys() if ('SF' in sub_file) and ('big' in sub_file)]
    for animal in data_object.dat.keys()
    for day in data_object.dat[animal].keys()))

d = {}
cell_count = 0
for i, (animal, cell_i) in enumerate(animals):
    group = animal.split('_')[1]
    if group not in d:
        d[group] = [cell_count, cell_i]
        cell_count += cell_i
    else: # change 'end' number
        cell_count += cell_i
        d[group][-1] = cell_count
        d[group] = np.array(d[group])
print(d)

groups = np.unique([animal[0].split('_')[1] for animal in animals])
colours = {'GCaMP6s': 'green', 'RD1': 'red', 'RD1opto': 'blue'}


# Apply UMAP for dimensionality reduction
# reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
# umap_embedding = reducer.fit_transform(data)
umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
umap_coords = umap_model.fit_transform(data)

# Step 3: Apply HDBSCAN for clustering
hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=2, cluster_selection_epsilon=0.1)
cluster_labels = hdbscan_model.fit_predict(umap_coords)

plt.figure(figsize=(10, 8))
for group, indices in d.items():
    plt.scatter(umap_coords[indices[0]:indices[1], 0], umap_coords[indices[0]:indices[1], 1], label = group, c=colours[group], s=50)
#plt.colorbar(scatter, label='Cluster Label')
plt.legend()
plt.title('UMAP')
plt.show()





# Apply UMAP for dimensionality reduction
# reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
# umap_embedding = reducer.fit_transform(data)
umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=3, random_state=42)
umap_coords = umap_model.fit_transform(data)

# Step 3: Apply HDBSCAN for clustering
hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=2, cluster_selection_epsilon=0.1)
cluster_labels = hdbscan_model.fit_predict(umap_coords)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
for group, indices in d.items():
    ax.scatter(umap_coords[indices[0]:indices[1], 0], umap_coords[indices[0]:indices[1], 1], umap_coords[indices[0]:indices[1], 2], label = group, c=colours[group], s=50)
#plt.colorbar(scatter, label='Cluster Label')
plt.legend()
plt.title('UMAP Visualization with HDBSCAN Clusters')
plt.show()

#RESPONSE AMPLITUDE OVER TIME > take the max across timepoints, average across cells, and then average across sessions
fig, ax = plt.subplots (1,2, figsize = (8,4))
screens = ['big', 'small']
for i, screen in enumerate(screens):
    opto = np.array(list(chain.from_iterable(
            [data_object.dat[animal][day][sub_file]['zscored_responses_ttls'][:, data_object.dat[animal][day][sub_file]['thresholded_cells']].max(axis = -1).mean(axis = 1) for sub_file in data_object.dat[animal][day].keys() if (('chirp' in sub_file) and (screen in sub_file))]
            for animal in data_object.dat.keys() if 'opto' in animal
            for day in data_object.dat[animal].keys())))
    rd1 = np.array(list(chain.from_iterable(
            [data_object.dat[animal][day][sub_file]['zscored_responses_ttls'][:, data_object.dat[animal][day][sub_file]['thresholded_cells']].max(axis = -1).mean(axis = 1) for sub_file in data_object.dat[animal][day].keys() if (('chirp' in sub_file) and (screen in sub_file))]
            for animal in data_object.dat.keys() if (('opto' not in animal) and ('RD1' in animal))
            for day in data_object.dat[animal].keys())))
    control = np.array(list(chain.from_iterable(
            [data_object.dat[animal][day][sub_file]['zscored_responses_ttls'][:, data_object.dat[animal][day][sub_file]['thresholded_cells']].max(axis = -1).mean(axis = 1) for sub_file in data_object.dat[animal][day].keys() if (('chirp' in sub_file) and (screen in sub_file))]
            for animal in data_object.dat.keys() if ('GCaMP' in animal)
            for day in data_object.dat[animal].keys())))

    ax[i].plot(opto.mean(axis = 0), c = 'blue', label = 'opto')
    ax[i].fill_between(np.arange(len(opto.mean(axis = 0))), opto.mean(axis = 0) - stats.sem(opto, axis = 0), opto.mean(axis = 0) + stats.sem(opto, axis = 0), color = 'blue', alpha = 0.5)
    ax[i].plot(rd1.mean(axis = 0), c = 'red', label = 'rd1')
    ax[i].fill_between(np.arange(len(rd1.mean(axis = 0))), rd1.mean(axis = 0) - stats.sem(rd1, axis = 0), rd1.mean(axis = 0) + stats.sem(rd1, axis = 0), color = 'red', alpha = 0.5)
    ax[i].plot(control.mean(axis = 0), c = 'black', label = 'ctrl')
    ax[i].fill_between(np.arange(len(control.mean(axis = 0))), control.mean(axis = 0) - stats.sem(control, axis = 0), control.mean(axis = 0) + stats.sem(control, axis = 0), color = 'black', alpha = 0.5)
    ax[i].set_xlabel('Repeat')
    ax[i].set_ylabel('Response amplitude')
    ax[i].set_title ('Regular Monitor' if 'big' in screen else 'Ultrabright Monitor')
plt.suptitle('On/Off response amplitude over time')
plt.legend()
plt.show()

##############
# response amplitude for F3 vs gratings
opto_chirps = np.array(list(chain.from_iterable(
        [data_object.dat[animal][day][sub_file]['responses_ttls'].max(axis = -1).mean(axis = 1) for sub_file in data_object.dat[animal][day].keys() if (('chirp' in sub_file))]
        for animal in data_object.dat.keys() if 'opto' in animal
        for day in data_object.dat[animal].keys())))
opto_chirps_mean, opto_chirps_sem = opto_chirps.mean(), stats.sem(opto_chirps, axis = (0,1))

opto_gratings = np.array(list(chain.from_iterable(
        [np.squeeze(data_object.dat['EC_GCaMP6s_06']['20241121']['big100_SFxO_000_001']['mean_ordered_grat_responses'].max(axis = -1).mean(axis = (0,2)))
         for sub_file in data_object.dat[animal][day].keys() if (('chirp' not in sub_file))]
        for animal in data_object.dat.keys() if 'opto' in animal
        for day in data_object.dat[animal].keys())))
opto_grat_mean, opto_grat_sem = opto_gratings.mean(), stats.sem(opto_gratings)

rd1_chirps = np.array(list(chain.from_iterable(
        [data_object.dat[animal][day][sub_file]['responses_ttls'].max(axis = -1).mean(axis = 1) for sub_file in data_object.dat[animal][day].keys() if (('chirp' in sub_file))]
        for animal in data_object.dat.keys() if (('opto' not in animal) and ('RD1' in animal))
        for day in data_object.dat[animal].keys())))
rd1_chirps_mean, rd1_chirps_sem = rd1_chirps.mean(), stats.sem(rd1_chirps, axis = (0,1))

rd1_gratings = np.array(list(chain.from_iterable(
        [data_object.dat[animal][day][sub_file]['mean_ordered_grat_responses'].max(axis = -1).mean() for sub_file in data_object.dat[animal][day].keys() if (('chirp' not in sub_file))]
        for animal in data_object.dat.keys() if (('opto' not in animal) and ('RD1' in animal))
        for day in data_object.dat[animal].keys())))
rd1_grat_mean, rd1_grat_sem = rd1_gratings.mean(), stats.sem(rd1_gratings)

control_chirps = np.array(list(chain.from_iterable(
        [data_object.dat[animal][day][sub_file]['responses_ttls'].max(axis = -1).mean(axis = 1) for sub_file in data_object.dat[animal][day].keys() if (('chirp' in sub_file))]
        for animal in data_object.dat.keys() if ('GCaMP' in animal)
        for day in data_object.dat[animal].keys())))
control_chirps_mean, control_chirps_sem = control_chirps.mean(), stats.sem(control_chirps, axis = (0,1))

ctr_gratings = np.array(list(chain.from_iterable(
        [data_object.dat[animal][day][sub_file]['mean_ordered_grat_responses'].max(axis = -1).mean() for sub_file in data_object.dat[animal][day].keys() if (('chirp' not in sub_file))]
        for animal in data_object.dat.keys() if ('GCaMP' in animal)
        for day in data_object.dat[animal].keys())))
ctr_grat_mean, ctr_grat_sem = ctr_gratings.mean(), stats.sem(ctr_gratings)

fig, ax = plt.subplots(1, 3, figsize = (8, 5), sharey = True)
# ax[0].bar(['$F^3$', 'Gratings'], [opto_chirps_mean, opto_grat_mean ], yerr = [opto_chirps_sem, opto_grat_sem], color ='blue', alpha = 0.5, width = 0.4, capsize = 4)
# #ax[0].errorbar(['$F^3$', 'Gratings'], [opto_chirps_mean, opto_grat_mean ], yerr = [opto_chirps_sem, opto_grat_sem])
# ax[1].bar(['$F^3$', 'Gratings'], [rd1_chirps_mean, rd1_grat_mean ], yerr = [rd1_chirps_sem, rd1_grat_sem], color ='red', alpha = 0.5,width = 0.4, capsize = 4)
# ax[2].bar(['$F^3$', 'Gratings'], [control_chirps_mean, ctr_grat_mean ], yerr = [control_chirps_sem, ctr_grat_sem], color ='black',alpha = 0.5, width = 0.4, capsize = 4)
ax[0].bar(['$F^3$', 'Gratings'], [opto_chirps_mean, opto_grat_mean ], color ='blue', alpha = 0.5, width = 0.4)
#ax[0].errorbar(['$F^3$', 'Gratings'], [opto_chirps_mean, opto_grat_mean ], yerr = [opto_chirps_sem, opto_grat_sem])
ax[1].bar(['$F^3$', 'Gratings'], [rd1_chirps_mean, rd1_grat_mean ], color ='red', alpha = 0.5,width = 0.4)
ax[2].bar(['$F^3$', 'Gratings'], [control_chirps_mean, ctr_grat_mean ], color ='black',alpha = 0.5, width = 0.4)
ax[0].set_ylabel('Response Amplitude')
ax[0].set_title('Opto')
ax[1].set_title('RD1')
ax[2].set_title('Control')
# plt.xlabel("Courses offered")
# plt.ylabel("No. of students enrolled")
# plt.title("Students enrolled in different courses")
plt.show()





##############
# % responsive cells
neurons = np.where(np.any(data_object.dat[animal][day][sub_file]['responses_ttls'][1:].mean(axis = 0)> 2, axis=1))[0]


opto_chirps_responding_cells = np.array(list(chain.from_iterable(
        [len(np.where(np.any(data_object.dat[animal][day][sub_file]['responses_ttls'][1:].mean(axis = 0)> 2, axis=1))[0]) for sub_file in data_object.dat[animal][day].keys() if (('chirp' in sub_file))]
        for animal in data_object.dat.keys() if 'opto' in animal
        for day in data_object.dat[animal].keys())))
opto_chirps_responding_cells_mean, opto_chirps_responding_cells_sem = opto_chirps_responding_cells.mean(), stats.sem(opto_chirps_responding_cells)

opto_gratings_responding_cells = np.array(list(chain.from_iterable(
        [len(np.where(np.any(data_object.dat[animal][day][sub_file]['mean_ordered_grat_responses']> 2, axis=-2))[0]) for sub_file in data_object.dat[animal][day].keys() if (('chirp' not in sub_file))]
        for animal in data_object.dat.keys() if 'opto' in animal
        for day in data_object.dat[animal].keys())))
opto_chirps_responding_cells_mean, opto_chirps_responding_cells_sem = opto_chirps_responding_cells.mean(), stats.sem(opto_chirps_responding_cells)


opto_chirps_mean, opto_chirps_sem = opto_chirps.mean(), stats.sem(opto_chirps, axis = (0,1))

opto_gratings = np.array(list(chain.from_iterable(
        [data_object.dat[animal][day][sub_file]['mean_ordered_grat_responses'].max(axis = -1).mean() for sub_file in data_object.dat[animal][day].keys() if (('chirp' not in sub_file))]
        for animal in data_object.dat.keys() if 'opto' in animal
        for day in data_object.dat[animal].keys())))
opto_grat_mean, opto_grat_sem = opto_gratings.mean(), stats.sem(opto_gratings)

rd1_chirps = np.array(list(chain.from_iterable(
        [data_object.dat[animal][day][sub_file]['responses_ttls'].max(axis = -1).mean(axis = 1) for sub_file in data_object.dat[animal][day].keys() if (('chirp' in sub_file))]
        for animal in data_object.dat.keys() if (('opto' not in animal) and ('RD1' in animal))
        for day in data_object.dat[animal].keys())))
rd1_chirps_mean, rd1_chirps_sem = rd1_chirps.mean(), stats.sem(rd1_chirps, axis = (0,1))

rd1_gratings = np.array(list(chain.from_iterable(
        [data_object.dat[animal][day][sub_file]['mean_ordered_grat_responses'].max(axis = -1).mean() for sub_file in data_object.dat[animal][day].keys() if (('chirp' not in sub_file))]
        for animal in data_object.dat.keys() if (('opto' not in animal) and ('RD1' in animal))
        for day in data_object.dat[animal].keys())))
rd1_grat_mean, rd1_grat_sem = rd1_gratings.mean(), stats.sem(rd1_gratings)

control_chirps = np.array(list(chain.from_iterable(
        [data_object.dat[animal][day][sub_file]['responses_ttls'].max(axis = -1).mean(axis = 1) for sub_file in data_object.dat[animal][day].keys() if (('chirp' in sub_file))]
        for animal in data_object.dat.keys() if ('GCaMP' in animal)
        for day in data_object.dat[animal].keys())))
control_chirps_mean, control_chirps_sem = control_chirps.mean(), stats.sem(control_chirps, axis = (0,1))

ctr_gratings = np.array(list(chain.from_iterable(
        [data_object.dat[animal][day][sub_file]['mean_ordered_grat_responses'].max(axis = -1).mean() for sub_file in data_object.dat[animal][day].keys() if (('chirp' not in sub_file))]
        for animal in data_object.dat.keys() if ('GCaMP' in animal)
        for day in data_object.dat[animal].keys())))
ctr_grat_mean, ctr_grat_sem = ctr_gratings.mean(), stats.sem(ctr_gratings)

fig, ax = plt.subplots(1, 3, figsize = (8, 5), sharey = True)
ax[0].bar(['$F^3$', 'Gratings'], [opto_chirps_mean, opto_grat_mean ], yerr = [opto_chirps_sem, opto_grat_sem], color ='blue', alpha = 0.5, width = 0.4, capsize = 4)
#ax[0].errorbar(['$F^3$', 'Gratings'], [opto_chirps_mean, opto_grat_mean ], yerr = [opto_chirps_sem, opto_grat_sem])
ax[1].bar(['$F^3$', 'Gratings'], [rd1_chirps_mean, rd1_grat_mean ], yerr = [rd1_chirps_sem, rd1_grat_sem], color ='red', alpha = 0.5,width = 0.4, capsize = 4)
ax[2].bar(['$F^3$', 'Gratings'], [control_chirps_mean, ctr_grat_mean ], yerr = [control_chirps_sem, ctr_grat_sem], color ='black',alpha = 0.5, width = 0.4, capsize = 4)
ax[0].set_ylabel('Response Amplitude')
ax[0].set_title('Opto')
ax[1].set_title('RD1')
ax[2].set_title('Control')
# plt.xlabel("Courses offered")
# plt.ylabel("No. of students enrolled")
# plt.title("Students enrolled in different courses")
plt.show()


# from classes import *
# animals_days = {'EC_GCaMP6s_08': ['20240927'], 'EC_GECO920_17': ['20241004']}
# data_object = DataAnalysis ('E:\\vision_restored', dict_animals_days = animals_days, response_type = 'fluorescence', dlc = False, show_plots = True)
#
# # ordered_sfs : shape n_repeats x n_orientations x n_sfs x 1 > represents the SFs shown for each presentation (presentations sorted according to orientation)
# a = data_object.dat['EC_GCaMP6s_08']['20240927']['big100_SFxO_000_000']['ordered_SFs']
# sort_i = np.argsort(a, axis = 2) # indices to sort 'ordered_sfs' in ascending order for each repeat & orientation block
#
# #sort along the SF axis, and then average across repeats
# sorted_sfs = np.take_along_axis(a, sort_i, axis=2).mean(axis = 0)
#
# # sort the responses along the SF axis, and then average across repeats
# expanded_sort_i = np.expand_dims(sort_i, axis=(4, 5))
# sorted_responses = np.take_along_axis(data_object.dat['EC_GCaMP6s_08']['20240927']['big100_SFxO_000_000']['mean_ordered_grat_responses'], expanded_sort_i, axis=2).mean(axis = 0)
#
# # CALCULATE RESPONSE AMPLITUDE FOR EACH SF
# sorted_responses = np.take_along_axis(data_object.dat['EC_GCaMP6s_08']['20240927']['big100_SFxO_000_000']['mean_ordered_grat_responses'], expanded_sort_i, axis=2)
# amplitude_SF = np.squeeze(sorted_responses.max(axis = -1)).mean(axis = 0).mean(axis = 0) # shape SFs x cells > represents the average response amplitude for each SF


def polar_plots_across_days(obj_day_session):
    '''
    plot orientation tuning curves as circular polar plots

    for each ROI, plot tuning curve for each day separately
    :param obj:
    :return:
    '''
    if (obj_day_session['n_theta'] > 1):
        n_cells = 3 # obj_day_session['n_cells']

        #colors = plt.cm.viridis(n_cells)
        # variables and dependencies for colour mapping
        plasma = plt.get_cmap('plasma')
        cNorm  = colors.Normalize(vmin=0, vmax=obj_day_session['n_SF']+5)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=plasma)
        scalarMap.set_array([])

        for cell in range(n_cells):

            fig, ax = plt.subplots(obj_day_session['n_TF'], obj_day_session['n_SF'], subplot_kw={'projection': 'polar'}, figsize=(2*obj_day_session['n_SF'], 2.5*obj_day_session['n_TF']))
            ax = np.atleast_2d(ax)

            for i_sf in range(obj_day_session['n_SF']):
                for i_tf in range(obj_day_session['n_TF']):
                    #rmax = (obj_day_session['tuning_curves'][cell] / obj_day_session['tuning_curves'][cell].sum()).max()
                    rmax = obj_day_session['tuning_curves'][cell].max()

                    # polar plots need to be plotted in radians
                    # subtract the starting angle so all cells's starting preferred angle is at 0 degrees
                    #theta- theta[0]

                    # r = vector of responses for each direction(response vector)
                    r = obj_day_session['tuning_curves'][cell, i_sf, i_tf, :]
                    r /= r.sum() # normalizing responses so they're between 0 and 1
                    theta = obj_day_session['theta'][i_sf, i_tf, :]

                    #to join the last point and first point
                    idx = np.arange(r.shape[0] + 1)
                    idx[-1] = 0

                    # plotting
                    ax[i_tf, i_sf].plot(theta, r, linewidth = 2, color=scalarMap.to_rgba(i_sf), alpha = 0.6)
                    ax[i_tf, i_sf].plot(theta[idx], r[idx], linewidth = 2,color=scalarMap.to_rgba(i_sf), alpha = 0.6)
                    ax[i_tf, i_sf].set_thetagrids([0, 90, 180, 270], y=0.2,
                                            labels=['0', '\u03c0' + '/2', '\u03c0', '3' + '\u03c0' + '/2'],
                                            fontsize=8)  # labels = ['0', '','\u03c0','']
                    ax[i_tf, i_sf].set_rmax(rmax)
                    ax[i_tf, i_sf].set_rlabel_position(45) # r is normalized response
                    ax[i_tf, i_sf].tick_params(axis='y', labelsize=8)
                    ax[i_tf, i_sf].set_rticks(np.round(np.linspace(0, rmax, 2),1))
                    ax[i_tf, i_sf].grid(True)
                    #ax[i_tf, i_sf].set_title(f'sf {obj_day_session['ordered_SFs']}', fontsize = 10)

                plt.tight_layout(pad=0.9)
                fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.suptitle(f'Tuning Curves  \n ROI #{cell}', fontsize = 12)
                plt.show()
                #plt.savefig(fr'C:\Users\erica\OneDrive\Desktop\tuning curves\roi{cell}')

polar_plots_across_days(data_object.dat['EC_GCaMP6s_08']['20240927']['big100_SFxO_000_000'])
polar_plots_across_days(data_object.dat['EC_GECO920_17']['20241004']['big100_000_001'])

responses = data_object.dat['EC_RD1_05']['20240904']['big100_000_000']['mean_ordered_grat_responses_whole']
on_responses, off_responses = responses[:,:,:, :data_object.fps * 2], responses[:,:,:, -(data_object.fps * 2):]

avg_on , avg_off = on_responses.mean(axis = 0).mean(axis = 0), off_responses.mean(axis = 0).mean(axis = 0)

fig,ax = plt.subplots(1,2, figsize = (8,4))
ax[0].plot(avg_on.mean(axis = 0), c = 'red')
ax[0].axvline(x = data_object.fps, c = 'Grey', linestyle = '--', alpha = 0.6)
ax[0].set_title('on responses')
ax[0].set_ylabel('average response')
arrow1 = patches.FancyArrowPatch((0, -0.296), (data_object.fps-0.8, -0.296), mutation_scale=15, color='dimgrey',arrowstyle='<|-')
ax[0].add_patch(arrow1)
ax[0].text(data_object.fps/2, -0.293, 'spontaneous', ha='center', va='center', color='dimgrey')
arrow2 = patches.FancyArrowPatch((data_object.fps+0.8, -0.296), (2*data_object.fps, -0.296), mutation_scale=15, color='blue', arrowstyle='-|>')
ax[0].add_patch(arrow2)
ax[0].text(data_object.fps*(3/2), -0.293, 'grating', ha='center', va='center', color='blue')

ax[1].plot(avg_off.mean(axis = 0), c = 'red')
ax[1].axvline(x = data_object.fps, c = 'Grey', linestyle = '--', alpha = 0.6)
ax[1].set_title('off responses')
ax[1].set_ylabel('average response')
arrow1 = patches.FancyArrowPatch((0, -0.296), (data_object.fps-0.8, -0.296), mutation_scale=15, color='blue',arrowstyle='<|-')
ax[1].add_patch(arrow1)
ax[1].text(data_object.fps/2, -0.293, 'grating', ha='center', va='center', color='blue')
arrow2 = patches.FancyArrowPatch((data_object.fps+0.8, -0.296), (2*data_object.fps, -0.296), mutation_scale=15, color='dimgrey', arrowstyle='-|>')
ax[1].add_patch(arrow2)
ax[1].text(data_object.fps*(3/2), -0.293, 'spontaneous', ha='center', va='center', color='dimgrey')

plt.show()

dil1 = np.loadtxt (os.path.join(r'E:\ao 20240822', 'DIL1.txt'), skiprows = 1) # shape (3078404, 2), 0.03004 ms between each frame (consistently)
enug = np.loadtxt (os.path.join(r'E:\ao 20240822', 'ENug.txt'), skiprows = 1) # (3078401, 2), 0.03004 ms between each frame (consistently) [ends a few frames sooner than dil1]
syncline = np.loadtxt (os.path.join(r'E:\ao 20240822', 'syncline.txt'), skiprows = 1) # shape  (41, 2), 0.03004 ms between each frame (consistently)
syncframe = np.loadtxt (os.path.join(r'E:\ao 20240822', 'syncframe.txt'), skiprows = 1) # shape (4161, 2), 0.03004 ms between each frame (consistently)

plt.figure()
#plt.plot(dil1[:, 0]/1000, dil1[:,1], c = 'blue')
#[plt.axvline(dil1[frame, 0]/1000, c = 'grey', alpha = 0.6) for frame in np.argwhere(acquisition_status)[:,0]]
plt.plot(enug[:, 0]/1000, enug[:,1]+0.00)
#plt.plot(syncline[:, 0]/1000, syncline[:,1]+0.00)
#plt.plot(syncframe[:, 0]/1000, syncframe[:,1]+0.00)
plt.show()

measurement_length_ms = dil1[-1, 0]

# Image frame rate and interval
fps_2p = 0.400108
frame_interval_2p_s = 1/fps_2p
frame_interval_2p_ms = (1/fps_2p)*1000

# Number of frames
num_frames = fps_2p * measurement_length_ms / 1000

frame_times = np.arange(0, num_frames * frame_interval_2p_ms, frame_interval_2p_ms)

# ttl values at times dil1[:,0]
ttl_interpolator = np.interp(frame_times, dil1[:,0], dil1[:,1])

# Create a binary array indicating frame acquisition
acquisition_status = np.zeros(len(dil1[:,0]), dtype=int)  # Initialize with zeros

# Mark acquisition times in the array
for time in frame_times:
    # Find the closest TTL time index
    idx = np.searchsorted(dil1[:,0], time, side='left')
    if idx < len(dil1[:,0]) and np.isclose(dil1[idx,0], time):
        acquisition_status[idx] = 1  # Mark as frame acquisition

# Verify the result
print(acquisition_status)

# plt.figure(figsize = (13,4))
# cell = 5
# clean = dat_dict['responses'][:, dat_dict['ttl_data'][0]:dat_dict['ttl_data'][-1]][cell]
# raw = dat_dict['raw_responses'][:, dat_dict['ttl_data'][0]:dat_dict['ttl_data'][-1]][cell]
# behav = dat_dict['behav_predictions'][:, dat_dict['ttl_data'][0]:dat_dict['ttl_data'][-1]][cell]
#
# plt.plot(behav + 6, c = 'black', alpha = 0.7, label = 'predicted from behav')
# plt.plot(raw, c = 'grey', alpha = 0.7, label = 'raw activity')
# plt.plot(clean, c = 'blue', alpha = 0.7, label = 'residual activity')
# plt.legend(loc = 'upper right')
# plt.axis('off')
# plt.show()
#
# rastermap_responses = fit_rastermap(dat_dict['responses'][:, dat_dict['ttl_data'][0]:dat_dict['ttl_data'][-1]])
# rastermap_raw_responses = fit_rastermap(dat_dict['raw_responses'][:, dat_dict['ttl_data'][0]:dat_dict['ttl_data'][-1]])
# rastermap_behav_responses = fit_rastermap(dat_dict['behav_predictions'][:, dat_dict['ttl_data'][0]:dat_dict['ttl_data'][-1]])
#
# fig,ax = plt.subplots(3,1, figsize = (15,8))
# ax[2].imshow(rastermap_responses, vmin=0, vmax=1.5, cmap="gray_r", aspect="auto")
# ax[2].set_title ('Neural Reponses (behaviour regressed out)', fontsize = 10)
#
# ax[1].imshow(rastermap_raw_responses, vmin=0, vmax=1.5, cmap="gray_r", aspect="auto")
# ax[1].set_title ('Raw Neural Reponses', fontsize = 10)
# ax[0].imshow(rastermap_behav_responses, vmin=0, vmax=1.5, cmap="gray_r", aspect="auto")
# ax[0].set_title ('Predicted Reponses from Behaviour', fontsize = 10)
#
# for i in range(3):
#     [ax[i].axvline(ttl, c='blue', alpha=0.2) for ttl in np.unique(dat_dict['ttl_data'])[:-1]]
#     ax[i].set_xticks (np.arange(0, rastermap_responses.shape[1], (dat.fps*60)))
#     ax[i].set_xticklabels([int(num) for num in np.arange(0, rastermap_responses.shape[1],(dat.fps*60))/(dat.fps*60) ])
#     ax[i].set_xlabel('Time (minutes)')
#     ax[i].set_ylabel('ROI')
# plt.suptitle(f'{animal} ({day})', fontsize = 13)
# plt.tight_layout()
# plt.savefig(fr'E:\vision_restored\figures\{animal}, {day} response.png')
# plt.show()
#
#
# # plot
# fig = plt.figure(figsize=(12, 5))
# ax = fig.add_subplot(111)
# ax.imshow(a['rastermap_responses'], vmin=0, vmax=1.5, cmap="gray_r", aspect="auto")
# plt.show()
#
# # residuals = regress_out_behav(a['raw_responses'], a['behav_predictions'])
# #
# # fig, ax = plt.subplots(2, 1, figsize=(10, 7))
# # ax[0].plot(a['raw_responses'][100], c='black', alpha=0.7, label='raw')
# # ax[0].plot(a['behav_predictions'][100], c='grey', alpha=0.5, label='behav')
# # ax[0].plot(residuals[100], c='red', alpha=0.5, label='residuals')
# #
# # #ax[0].plot(a['responses'][100] + 1000, c='red', alpha=0.7, label='raw - behav')
# # ax[0].legend()
# #
# # ax[1].plot(normalize_arr(a['responses'])[100], c='blue', alpha=0.7, label='norm(raw - behav)')
# # ax[1].legend()
# # plt.show()





def plot_curve_polar(object, day, session):
    '''
    plot polar tuning curves for all ROIs
    :param object:
    :param day:
    :param session:
    :return:
    '''

    fig, ax = plt.subplots(figsize = (9,3))

    # plotting line plot
    # variables and dependencies for colour mapping
    plasma = plt.get_cmap('plasma')
    cNorm  = colors.Normalize(vmin=0, vmax=len(days))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=plasma)
    scalarMap.set_array([])

    responses = object.dat_subject[day][session]['tuning_curves']
    orientations = object.dat_subject[day][session]['theta']

    # plot mean tuning curve across each day
    for i_cell in range(responses.shape[0]):
        ax1 = fig.add_subplot(121)
        p1 = [ax1.plot(orientations, responses_days[i_day], linewidth = 2, alpha = 0.8, c=scalarMap.to_rgba(i_day)) for i_day in range (len(days))]
        ax1.set_xticks(np.arange(0,360,90))
        ax1.tick_params(axis='x', labelsize=8)
        ax1.tick_params(axis='y', labelsize=8)
        ax1.set_xlabel('Orientation (deg)', fontsize = 9)
        ax1.set_ylabel('Response', fontsize = 9)
        ax1.set_title('Tuning Curves', fontsize = 11)

        # plotting polar plot
        r = np.array([object.dat_subject[day]['OSI'][roi] for day in object.dat_subject.keys()])
        theta = np.deg2rad(np.array([object.dat_subject[day]['pref_orientation'][roi] for day in object.dat_subject.keys()]))

        ax2 = fig.add_subplot(122, polar=True)
        #p2 = ax2.scatter(theta, r, s=25, c=np.arange(theta.shape[0]), cmap='plasma')
        p2 = [ax2.scatter(theta[i_day], r[i_day], s=25, color=scalarMap.to_rgba(i_day)) for i_day in range (len(days))]
        ax2.plot(theta, r, c='black', alpha=0.5)
        ax2.set_thetamin(-90)
        ax2.set_thetamax(90)
        ax2.set_thetagrids([0, 90, -90], y=0.05, labels=['0', '\u03c0' + '/2', '3' + '\u03c0' + '/2'], fontsize=8)  # labels = ['0', '','\u03c0','']
        ax2.set_rlabel_position(45)
        ax2.set_rticks(np.round(np.linspace(0, r.max(), 4),2))
        ax2.tick_params(axis='y', labelsize=8)
        ax2.set_title('Preferred Orientation \n & Magnitude ', fontsize =  10)
        ax2.grid(True)

    # ax3 = fig.add_subplot(122)
    # p3 = ax3.colorbar(scalarMap, cax = ax3, ticks=range(4),orientation='vertical')
    # #p3.set_label('Day', labelpad=10)# , y=1.05, rotation=0)
    cax = fig.add_axes([0.92, 0.13, 0.02, 0.8])
    cb = plt.colorbar(scalarMap, cax = cax, ticks=np.arange(4),orientation='vertical')
    cb.set_label('Day', labelpad=8)# , y=1.05, rotation=0)

    plt.tight_layout(pad = 0.2)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle(f'ROI # {roi}', fontsize = 15)
    plt.show()
