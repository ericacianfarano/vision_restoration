import matplotlib.pyplot as plt

from helpers import *
from plotting import *
from behavioural_analysis import *


def zscore_thresholding (object, animal, day, subfile, zscore_threshold = 2):
    '''
    Considering responses at the best 'sf', calculate a z score based on the baseline
    Determines which cells were responsive

    :param object:
    :param parameter_matrix:  shape ((n_repeats, n_orientation, n_sf, n_cells, n_timepoints))
    :param zscore_threshold:
    :return:
    '''

    # exclude first repeat
    response_matrix = object.dat[animal][day][subfile]['responses_ttls'][1:] # shape n_repeats, cells, timepoints

    # take first 4sec of response (baseline) > mean/std across time/repeat axis > shape (n_cells)
    # we can assume that we have consistent baselines across repeats and orientations
    baseline_mean = response_matrix[...,:object.fps*4].mean(axis = (0,-1), keepdims = True)
    baseline_std = response_matrix[...,:object.fps*4].std(axis = (0,-1), keepdims = True)

    # z score the response, and then average across repeats (shape = ((n_repeats, n_cells, n_timepoints)
    z_scored_response = ((response_matrix - baseline_mean) / baseline_std)

    # take the z scored responses during the 'ON' period > average over repeats (shape = ((n_cells, n_timepoints)
    # 0-4 = grey, 4-8 = black, 8-12 = white, 12-16 = black, 16-20 = grey
    z_scored_response_mean = z_scored_response[:,:, object.fps * 4 : object.fps * 16].mean(axis = 0)
    z_scored_response_median = np.median(z_scored_response[:, :, object.fps * 4: object.fps * 16], axis=0)

    # see if the response at peak exceeds the zscore threshold (across any orientation, SF, and timepoints)
    cell_exceeds_threshold = (z_scored_response_mean > zscore_threshold-1).any(axis=(-1))
    #cell_exceeds_threshold = ((z_scored_response_mean > zscore_threshold) & (z_scored_response_median > (zscore_threshold-1) )).any(axis=(-1))

    return cell_exceeds_threshold


class OlfactionAnalysis:

    def __init__(self, path_list, dict_animals_days, pose_likelihood_threshold=0.7, zscore_threshold = 2.5, response_type='deconvolved', fps=20,
                 dlc=False, facemap=False, show_plots=False):

        self.dict_animals_days = dict_animals_days
        self.path_list = path_list
        # self.data_path = os.path.join(path, 'data')
        self.save_path = os.path.join(self.path_list[0], 'figures')
        self.dlc = dlc
        self.show_plots = show_plots
        self.pose_likelihood_threshold = pose_likelihood_threshold
        self.response_type = response_type
        self.fps = fps  # sampling rate of the data acquisition
        self.zscore_threshold = zscore_threshold
        self.facemap = facemap  # whether or not we have pushed the data through the facemap pipeline to get neural predictions
        self.preprocessing()
        # self.grating_responses = {}
        # self.orientations = {}

    def preprocessing(self):

        self.dat = {}

        for animal in self.dict_animals_days.keys():
            self.dat[animal] = {}
            for day in tqdm(self.dict_animals_days[animal], desc=f'Processing: {animal}'):

                # if the animal / day isn't stored on the first drive, check the second
                self.data_path = os.path.join(self.path_list, 'data')

                self.dat[animal][day] = {}

                # self.stim_dict[animal][day], self.responses_cells[animal][day],self.ttl_data[animal][day] = {}, {}, {}
                # self.list_stim_types[animal][day], self.list_stim_names[animal][day] = {}, {}
                # self.ttls[animal][day], self.stimulus_responses[animal][day] = {}, {}

                recordings = [file for file in os.listdir(os.path.join(self.data_path, animal, day)) if (
                            (not file.endswith('.mat')) and (not file.endswith('.png')) and (
                        not file.endswith('.tif')))]

                for recording in recordings:
                    sub_files = [file for file in os.listdir(os.path.join(self.data_path, animal, day, recording)) if (
                                (not file.endswith('.mat')) and (not file.endswith('.png')) and (
                            not file.endswith('.tiff')))]
                    for sub_file in sub_files:  # sub_files:

                        self.dat[animal][day][sub_file] = {}

                        # load suite2p information
                        suite2p_path = os.path.join(self.data_path, animal, day, recording, sub_file, 'experiments', 'suite2p', 'plane0')
                        responses, iscell, ops, stat = suite2p_files(suite2p_path, response_type=self.response_type)
                        self.dat[animal][day][sub_file]['meanImg'] = ops['meanImg']

                        # load ttl / neural information
                        ttl_file = [file for file in os.listdir(
                            os.path.join(self.data_path, animal, day, recording, sub_file, 'experiments')) if
                                    (file.endswith('.mat') and (not file.endswith('realtime.mat')))][0]
                        ttl_path = os.path.join(self.data_path, animal, day, recording, sub_file, 'experiments',
                                                ttl_file)
                        self.ttl_file = loadmat(ttl_path)
                        event_id = np.squeeze(loadmat(ttl_path)['info']['event_id'][0][0])
                        ttls = np.squeeze(loadmat(ttl_path)['info']['frame'][0][0])  # [event_id == 1]

                        # if we're showing chirps dont need to add extra missing 0 for wait period.
                        #self.dat[animal][day][sub_file]['ttl_data'] = check_ttls(self, ttls,self.dat[animal][day][sub_file]['log']['list_stim_names'])
                        self.dat[animal][day][sub_file]['ttl_data'] = ttls

                        # we calculate this so that we can shift the behavioural data / neural data over so everything starts as of the first TTL (start of wait period)
                        start_recording, end_recording = self.dat[animal][day][sub_file]['ttl_data'][0], self.dat[animal][day][sub_file]['ttl_data'][-1]


                        responses = responses[
                            iscell == 1]  # only take ROIs that suite2p has identified as a proper cell
                        # responses -= responses.min(axis=1, keepdims=True)           # normalize - ensure all responses are non-negative

                        self.dat[animal][day][sub_file]['responses'] = responses
                        #self.dat[animal][day][sub_file]['zscored_responses'] = zscore(responses, axis=1)

                        #store_metrics(self, animal, day, sub_file, zscore_threshold=3)

                        self.dat[animal][day][sub_file]['n_SF'], self.dat[animal][day][sub_file]['n_TF'], \
                        self.dat[animal][day][sub_file]['n_theta'] = 0, 0, 0

                        # shape (n_trial, 2) >  only take start ttl
                        ttl_arr = self.dat[animal][day][sub_file]['ttl_data'].reshape((-1, 2))

                        # shape n_trials, n_cells, n_timepooints (take 4 seconds)
                        self.dat[animal][day][sub_file]['responses_ttls'] = np.array([self.dat[animal][day][sub_file]['responses'][:, t:t + self.fps * 4] for t in ttl_arr[:,0]])
                        #print(self.dat[animal][day][sub_file]['responses_ttls'].shape)

                        # baseline is first 9 seconds > shape n_cells, n_timepoints
                        self.dat[animal][day][sub_file]['baseline'] = self.dat[animal][day][sub_file]['responses'][:, :self.fps * 9]
                        #baseline is first second of each trial
                        self.dat[animal][day][sub_file]['baseline_ttl'] = self.dat[animal][day][sub_file]['responses_ttls'][:,:, :self.fps]

                        # shape n_cells
                        baseline_mean = self.dat[animal][day][sub_file]['baseline'].mean(axis=-1, keepdims=True)
                        baseline_std = self.dat[animal][day][sub_file]['baseline'].std(axis=-1, keepdims=True)

                        # average the baseline for each trial across the time axis
                        baseline_mean_ttl = self.dat[animal][day][sub_file]['baseline_ttl'].mean(axis=-1, keepdims=True)
                        baseline_std_ttl = self.dat[animal][day][sub_file]['baseline_ttl'].std(axis=-1, keepdims=True)
                        self.dat[animal][day][sub_file]['baseline_std_ttl'] = baseline_std_ttl

                        # delta F over F with the baseline before each trial > shape n_trials, n_cells, n_timepoints
                        # subtract the average baseline from each response, and divide by average baseline > this is df/F
                        self.dat[animal][day][sub_file]['dF_over_F'] = 100*(self.dat[animal][day][sub_file]['responses_ttls'] - baseline_mean_ttl) / baseline_mean_ttl
                        # take each trials' responsive period (1-3 s after stim presentation_, and average over trials

                        # for each cell, calculate the std of the baseline period (1st second before stim presentation)
                        # then mulitply each value by 2-3 (pick same multiplicative factor for each cell) > this is how many stds above the mean you want to set above the threshold
                        self.dat[animal][day][sub_file]['cell_threshold'] = self.dat[animal][day][sub_file]['dF_over_F'][:, :,:self.fps].std(axis=(0, -1))*1.5
                        dff_mean_response = self.dat[animal][day][sub_file]['dF_over_F'][..., self.fps : self.fps * 3].mean(axis = (0,-1))
                        self.dat[animal][day][sub_file]['thresholded_cells'] = dff_mean_response > self.dat[animal][day][sub_file]['cell_threshold']

                        # z score responses according to baseline period
                        #self.dat[animal][day][sub_file]['dF_over_F'] = ((self.dat[animal][day][sub_file]['responses_ttls'] - baseline_mean) / baseline_mean)
                        self.dat[animal][day][sub_file]['z_scored_response'] = ((self.dat[animal][day][sub_file]['responses_ttls'] - baseline_mean) / baseline_std)

                        # take the z scored responses during the 1-3s after TTL > average over repeats (shape = ((n_cells, n_timepoints)
                        z_scored_response_mean = self.dat[animal][day][sub_file]['z_scored_response'][..., self.fps : self.fps * 3].mean(axis=0)
                        z_scored_response_median = np.median(self.dat[animal][day][sub_file]['z_scored_response'][..., self.fps : self.fps * 3], axis=0)

                        # see if the mean response at peak exceeds the zscore threshold (across any timepoint)
                        #cell_exceeds_threshold = ((z_scored_response_mean > self.zscore_threshold) & (z_scored_response_median > (self.zscore_threshold - 1))).any(axis=-1)
                        #self.dat[animal][day][sub_file]['thresholded_cells'] = cell_exceeds_threshold


# olfaction stuff
animals_days = {'EC_RD1_06': ['20250223'], 'EC_RD1_08': ['20250223']}           #simple  8 orientation x 6 repeats
data_object = OlfactionAnalysis ('I:\\olfaction', dict_animals_days = animals_days, zscore_threshold=0.7, response_type = 'fluorescence', dlc = False, show_plots = False)


# population average response for all responsive neurons
for animal in data_object.dat.keys():
    for day in data_object.dat[animal].keys():
        plt.figure(figsize=(6, 6))
        colors = plt.cm.plasma(np.linspace(0, 0.85, len(data_object.dat[animal][day].keys())))
        for i, sub_file in enumerate(data_object.dat[animal][day].keys()):
            passing_cells = data_object.dat[animal][day][sub_file]['thresholded_cells'].sum()
            total_cells = len(data_object.dat[animal][day][sub_file]['thresholded_cells'])
            plt.plot(data_object.dat[animal][day][sub_file]['z_scored_response'][:, data_object.dat[animal][day][sub_file]['thresholded_cells'], :].mean(axis = (0,1)), c=colors[i],
                     alpha = 0.8, label = f'{sub_file[4:-8]}, {passing_cells}/{total_cells} cells')

        plt.axvline(data_object.fps, c = 'grey')
        plt.suptitle(animal)
        plt.legend()
        plt.show()

# single cell response of olfaction-responsive neurons
for animal in data_object.dat.keys():
    for day in data_object.dat[animal].keys():
        for i, sub_file in enumerate(data_object.dat[animal][day].keys()):

            n_cells_to_plot = data_object.dat[animal][day][sub_file]['thresholded_cells'].sum()
            n_cols = 4
            n_rows = int(np.ceil(n_cells_to_plot/n_cols))

            fig, ax = plt.subplots(n_rows, n_cols, figsize=(11, 6))
            ax = ax.ravel()
            colors = plt.cm.plasma(np.linspace(0, 0.85, n_cells_to_plot))

            # shape n_trial, n_cells, n_timepoints
            thresholded_responses = data_object.dat[animal][day][sub_file]['dF_over_F'][:,data_object.dat[animal][day][sub_file]['thresholded_cells'], :]

            for i_cell in range(n_cells_to_plot):
                [ax[i_cell].plot(thresholded_responses[i_trial, i_cell,:], c=colors[i_cell], alpha = 0.5) for i_trial in range(thresholded_responses.shape[0])]
                ax[i_cell].plot(thresholded_responses[:, i_cell, :].mean(axis = 0), c='black', alpha=0.8, linewidth = 2)

                ax[i_cell].axvline(data_object.fps, c='grey')
                ax[i_cell].set_title(f'cell # {i_cell}')


            plt.suptitle((animal, sub_file))
            plt.tight_layout()
            plt.show()

##############
all_thresholded_cells = []
for animal in data_object.dat.keys():
    for day in data_object.dat[animal].keys():
        for i, sub_file in enumerate(data_object.dat[animal][day].keys()):
            thresholded_responses = data_object.dat[animal][day][sub_file]['dF_over_F'][:,
                                    data_object.dat[animal][day][sub_file]['thresholded_cells'], :].mean(axis = 0)
            #print(data_object.dat[animal][day][sub_file]['thresholded_cells'].sum())
            all_thresholded_cells.append(thresholded_responses)
responses_all = np.vstack(all_thresholded_cells)
plt.figure()
plt.plot(responses_all.T, color = 'lightsteelblue', alpha = 0.7)
mean_response = responses_all.mean(axis = 0)
plt.plot(mean_response, color = 'mediumblue', linewidth = 2, alpha = 0.8)
plt.fill_between(np.arange(responses_all.shape[1]), mean_response - sem(responses_all, axis = 0), mean_response + sem(responses_all, axis = 0), color = 'mediumblue', alpha = 0.3)
plt.axvline(data_object.fps, c = 'grey', linestyle = '--')
plt.ylabel('Response % (dF/F)')
plt.xlabel('Time since stim onset (s)')
plt.title('Responsive cells across all animals')
#plt.ylim([0,5])
plt.xticks(np.arange(0, mean_response.shape[0], data_object.fps), np.arange(0, mean_response.shape[0], data_object.fps) - data_object.fps)
plt.show()



#### from rd106 v1, plot cell 1 trial by trial
dat = data_object.dat['EC_RD1_06']['20250223']['olf_v1_000_000']
cell_idx = np.where(dat['thresholded_cells'])[0][1]
responses = dat['dF_over_F'][:, cell_idx, :]
fig,ax = plt.subplots(3,5, figsize = (10,5), sharex = True)
ax = ax.ravel()
[ax[i_repeat].plot(responses[i_repeat], c = 'black', alpha = 0.8) for i_repeat in range(responses.shape[0])]
[ax[i_repeat].axvline(data_object.fps, c = 'grey', alpha = 0.5) for i_repeat in range(responses.shape[0])]
plt.tight_layout()
plt.show()

# 5 best trials for one cell, and the average
dat = data_object.dat['EC_RD1_06']['20250223']['olf_v1_000_000']
cell_idx = np.where(dat['thresholded_cells'])[0][1]
responses = dat['dF_over_F'][[0,10,6,8,13], cell_idx, ]
mean_response = responses.mean(axis = 0)
plt.figure(figsize = (7,5))
plt.plot(responses.T, c = 'cornflowerblue', alpha = 0.5)
plt.plot(mean_response, c = 'mediumblue', alpha =0.8, linewidth = 2.5)
plt.fill_between(np.arange(responses.shape[1]), mean_response - sem(responses, axis = 0), mean_response + sem(responses, axis = 0), color = 'mediumblue', alpha = 0.3)
plt.axvline(data_object.fps, c = 'grey', linestyle='--')
plt.xticks(np.arange(0, mean_response.shape[0], data_object.fps), (np.arange(0, mean_response.shape[0], data_object.fps) - data_object.fps)/data_object.fps)
plt.ylabel('Response % (dF/F)')
plt.xlabel('Time since stim onset (s)')
plt.title(f'Average response of cell # {cell_idx}')
plt.show()




            # # now need to fix & parse ttl data
                        # get_neuronal_responses_ttls(self, self.dat[animal][day][sub_file])
                        #
                        # self.dat[animal][day][sub_file]['n_cells'] = self.dat[animal][day][sub_file]['responses'].shape[0]
                        #
                        # # on_off_responses(self, animal, day, sub_file)
                        #
                        # self.dat[animal][day][sub_file]['thresholded_cells'] = zscore_thresholding(self, animal, day, sub_file, zscore_threshold=3)
