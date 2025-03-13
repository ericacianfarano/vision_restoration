from helpers import *
from plotting import *
from behavioural_analysis import *

class DataAnalysis:

    def __init__(self, path_list, dict_animals_days, pose_likelihood_threshold = 0.7, response_type = 'deconvolved', fps = 20, dlc = False, facemap = False, show_plots = False):

        self.dict_animals_days = dict_animals_days
        self.path_list = path_list
        #self.data_path = os.path.join(path, 'data')
        self.save_path = os.path.join(self.path_list[0], 'figures')
        self.dlc = dlc
        self.show_plots = show_plots
        self.pose_likelihood_threshold = pose_likelihood_threshold
        self.response_type = response_type
        self.fps = fps              # sampling rate of the data acquisition
        self.facemap = facemap  # whether or not we have pushed the data through the facemap pipeline to get neural predictions
        self.preprocessing()
        # self.grating_responses = {}
        # self.orientations = {}

    def preprocessing (self):

        self.dat = {}

        for animal in self.dict_animals_days.keys():
            self.dat[animal] = {}
            for day in tqdm(self.dict_animals_days[animal], desc = f'Processing: {animal}'):

                # if the animal / day isn't stored on the first drive, check the second
                if os.path.exists(os.path.join(self.path_list[0], 'data', animal, day)):
                    self.data_path = os.path.join(self.path_list[0], 'data')
                else:
                    self.data_path = os.path.join(self.path_list[1], 'data')

                self.dat[animal][day] = {}

                # self.stim_dict[animal][day], self.responses_cells[animal][day],self.ttl_data[animal][day] = {}, {}, {}
                # self.list_stim_types[animal][day], self.list_stim_names[animal][day] = {}, {}
                # self.ttls[animal][day], self.stimulus_responses[animal][day] = {}, {}

                recordings = [file for file in os.listdir(os.path.join(self.data_path, animal, day)) if ((not file.endswith('.mat')) and (not file.endswith('.png')) and (not file.endswith('.tif')))]
                for recording in recordings:
                    sub_files = [file for file in os.listdir(os.path.join(self.data_path, animal, day, recording)) if ((not file.endswith('.mat')) and (not file.endswith('.png')) and (not file.endswith('.tiff')))]
                    for sub_file in sub_files: #sub_files:

                        self.dat[animal][day][sub_file] = {}


                        # load log file information
                        log_file = [file for file in os.listdir(os.path.join(self.data_path, animal, day, recording, sub_file, 'logfiles')) if file.endswith('_log.txt')][0]
                        log_path = os.path.join(self.data_path, animal, day, recording, sub_file, 'logfiles', log_file)

                        self.dat[animal][day][sub_file]['log'] = {}
                        self.dat[animal][day][sub_file]['log']['stim_dict'], self.dat[animal][day][sub_file]['log']['list_stim_types'], self.dat[animal][day][sub_file]['log']['list_stim_names'], self.dat[animal][day][sub_file]['ntheta'], self.dat[animal][day][sub_file]['nrepeats'] = load_stims (log_path)

                        # load suite2p information
                        suite2p_path = os.path.join(self.data_path, animal, day, recording, sub_file, 'experiments', 'suite2p', 'plane0')
                        responses, iscell, ops, stat = suite2p_files (suite2p_path, response_type ='fluorescence')
                        spikes, _, _, _ = suite2p_files(suite2p_path, response_type= 'deconvolved')
                        self.dat[animal][day][sub_file]['meanImg'] = ops['meanImg']

                        # load ttl / neural information
                        ttl_file = [file for file in os.listdir(os.path.join(self.data_path, animal, day, recording, sub_file, 'experiments')) if (file.endswith('.mat') and (not file.endswith('realtime.mat')))][0]
                        ttl_path = os.path.join(self.data_path, animal, day, recording, sub_file, 'experiments', ttl_file)
                        self.ttl_file = loadmat(ttl_path)
                        event_id = np.squeeze(loadmat(ttl_path)['info']['event_id'][0][0])
                        ttls = np.squeeze(loadmat(ttl_path)['info']['frame'][0][0])  # [event_id == 1]

                        # if we're showing chirps dont need to add extra missing 0 for wait period.
                        if 'chirps' not in sub_file:
                            self.dat[animal][day][sub_file]['ttl_data'] = check_ttls (self, ttls, self.dat[animal][day][sub_file]['log']['list_stim_names'])
                        else:
                            self.dat[animal][day][sub_file]['ttl_data'] = ttls

                        # we calculate this so that we can shift the behavioural data / neural data over so everything starts as of the first TTL (start of wait period)
                        start_recording, end_recording = self.dat[animal][day][sub_file]['ttl_data'][0], self.dat[animal][day][sub_file]['ttl_data'][-1]

                        # only keep ROIs that qualify as cells
                        # z-score responses > z = (x-mu)/sigma) ; where mu = mean of the responses, sigma = standard deviation of the responses

                        if self.facemap:
                            self.dat[animal][day][sub_file]['raw_responses'] = zscore(responses[iscell == 1], axis=1)#[:, start_recording:end_recording]
                        else:
                            responses = responses[iscell == 1]                          # only take ROIs that suite2p has identified as a proper cell
                            spikes = spikes[iscell == 1]  # only take ROIs that suite2p has identified as a proper cell
                            #responses -= responses.min(axis=1, keepdims=True)           # normalize - ensure all responses are non-negative

                            self.dat[animal][day][sub_file]['responses'] = responses
                            self.dat[animal][day][sub_file]['deconvolved_responses'] = spikes
                            self.dat[animal][day][sub_file]['zscored_responses'] = zscore(responses, axis=1)

                            # un comment when ready - takes long
                            #self.dat_subject[day][sub_file]['rastermap_responses'] = fit_rastermap(self.dat_subject[day][sub_file]['responses'])

                        # since we're cutting out the start and the end of the recording, the ttls need to start at '0'
                        #if self.dat[animal][day][sub_file]['ttl_data'][0] != 0:
                            # self.dat[animal][day][sub_file]['ttl_data'] = self.dat[animal][day][sub_file]['ttl_data'] - self.dat[animal][day][sub_file]['ttl_data'][0]


                        # after converting .mj2 file to .avi file (with openCV) and performing pose estimation (with DLC) (in VScode notebook)
                        # note that camera acquires at same rate as 2p - so raw files is the entire length of the recroding

                        behav_path = os.path.join(self.data_path, animal, day, recording, sub_file, 'behav')

                        if self.dlc:
                            self.dat[animal][day][sub_file]['poses'] = load_dlc_pose_estimates(behav_path, self.pose_likelihood_threshold)[self.dat[animal][day][sub_file]['start_recording']:self.dat[animal][day][sub_file]['end_recording'], :]

                            ellipse_behav_video(behav_path, self.dat[animal][day][sub_file]['poses'],
                                                    self.dat[animal][day][sub_file]['start_recording'], self.dat[animal][day][sub_file]['end_recording'])


                        if self.facemap:
                            # when predicting activity from behavoiour feed in raw suite2p spks.npy file
                            # load behaviour video to process SVDs in faemap https://github.com/MouseLand/facemap)
                            behav_predictions = np.load(os.path.join(behav_path, f'neural_predictions_{self.response_type}.npy'), allow_pickle=True).item()['predictions']
                            self.dat[animal][day][sub_file]['behav_predictions'] = behav_predictions[iscell == 1]       # only keep ROIs that qualify as cells

                            # regress out behavioural activity, then normalize activity by converting it into a distribution with a mean of 0 and a standard deviation of 1 (along the specified aXis)
                            self.dat[animal][day][sub_file]['behav_predictions'] = zscore(
                                self.dat[animal][day][sub_file]['behav_predictions'], axis=1)

                            self.dat[animal][day][sub_file]['responses'] = regress_out_behav(self.dat[animal][day][sub_file]['raw_responses'], self.dat[animal][day][sub_file]['behav_predictions'])

                        # fit rastermap (https://github.com/MouseLand/rastermap) (https://www.biorxiv.org/content/10.1101/2023.07.25.550571v2)
                        #self.dat[animal][day][sub_file]['rastermap_responses'] = fit_rastermap (self.dat[animal][day][sub_file]['responses'])

                        if self.dlc:
                            nystagmus(self, self.dat[animal][day][sub_file]['poses'])

                        #parsed = parse_grating_array(self.dat[animal][day][sub_file]['log']['stim_dict']['GratingStim 0'])
                        #print(parsed)
                        #self.dat[animal][day][sub_file]['SFs'] = parsed[:, 1]
                        #self.dat[animal][day][sub_file]['TFs'] = parsed[:, 2]
                        #self.dat[animal][day][sub_file]['n_theta'], self.dat[animal][day][sub_file]['n_SF'], self.dat[animal][day][sub_file]['n_TF'] = len(np.unique(parsed[:, 0])), len(np.unique(parsed[:, 1])), len(np.unique(parsed[:, 2]))

                        if 'chirp' not in sub_file:
                            parsed = np.array([parse_grating_array(
                                self.dat[animal][day][sub_file]['log']['stim_dict'][grating_block]) for grating_block in
                                      self.dat[animal][day][sub_file]['log']['stim_dict'].keys() if 'Grating' in grating_block])

                            self.dat[animal][day][sub_file]['SFs'] = parsed[:,:, 1]
                            self.dat[animal][day][sub_file]['TFs'] = parsed[:, :, 2]
                            self.dat[animal][day][sub_file]['thetas'] = parsed[:, :, 0]

                            self.dat[animal][day][sub_file]['n_SF'] = len(np.unique(self.dat[animal][day][sub_file]['SFs']))
                            self.dat[animal][day][sub_file]['n_TF'] = len(np.unique(self.dat[animal][day][sub_file]['TFs']))
                            self.dat[animal][day][sub_file]['n_theta'] = len(np.unique(self.dat[animal][day][sub_file]['thetas']))

                            # now need to fix & parse ttl data
                            get_neuronal_responses_ttls(self, self.dat[animal][day][sub_file])

                            # reshape into shape n_repeats x n_theta x n_SF x n_TF (+ squeeze gets rid of singleton dimension) x n_cells x n_timepoints
                            self.dat[animal][day][sub_file]['n_cells'] = self.dat[animal][day][sub_file]['responses'].shape[0]

                            #spatial_footprint(self, animal, day, recording, sub_file)
                            #self.dat[animal][day][sub_file][f'thresholded_cells'] = zscore_thresholding(self, animal,day, sub_file,zscore_threshold=3)

                            # 8 orientations x 6 repeats x 1 SF x 1 TF
                            if (self.dat[animal][day][sub_file]['n_SF'] == 1) and (self.dat[animal][day][sub_file]['n_TF'] == 1):
                                print('8 ori x 6 repeats')

                            # # calculating orientation selectivity > average responses to the same orientation within the same 'block'
                            # responses_gratings(self.dat[animal][day][sub_file])
                            #
                            # get_complex_num(self.dat[animal][day][sub_file], avg_over_sf = False)

                            # (8 orientations x 6 repeats x 5 SF x 1 TF) OR (1 orientations x 6 repeats x 5 SF x 5 TF)

                            # #plot_raw_responses(self, animal)

                            else:
                                print('8 ori x SF x TF 6 repeats')

                                # for each cell, find the SF with the peak response at each cell's preferred orientation
                                #find_pref_SF(self.dat[animal][day][sub_file])

                                #plot_tuning_curves(self, animal, day, sub_file)

                                # if (self.dat[animal][day][sub_file]['n_SF'] > 1) and (self.dat[animal][day][sub_file]['n_TF'] == 1): #8 orientations, 5 SF, 1TF
                                #     sf_tuning (self, animal, day, sub_file)
                                # elif (self.dat[animal][day][sub_file]['n_TF'] > 1): # 1 orientation, 5 SF, 5 TF
                                #     sf_tf_tuning(self, animal, day, sub_file)

                                #plot_rasters(self, animal, day, sub_file)
                                #plot_activity_rasters(self, animal, day, sub_file)
                                #hist_osi_angle(self, animal, day, sub_file

                            store_metrics(self, animal, day, sub_file, zscore_threshold=3)
                            #parameter_matrix_plot(self, animal, day, sub_file, zscore_threshold = 3)

                        elif 'chirp' in sub_file:
                            self.dat[animal][day][sub_file]['n_SF'], self.dat[animal][day][sub_file]['n_TF'], self.dat[animal][day][sub_file]['n_theta'] = 0,0,0

                            # now need to fix & parse ttl data
                            get_neuronal_responses_ttls(self, self.dat[animal][day][sub_file])

                            self.dat[animal][day][sub_file]['n_cells'] = self.dat[animal][day][sub_file]['responses'].shape[0]

                            #on_off_responses(self, animal, day, sub_file)

                            self.dat[animal][day][sub_file]['zscored_matrix_baseline'],  self.dat[animal][day][sub_file]['thresholded_cells'] = zscore_thresholding(self, animal, day, sub_file, zscore_threshold=3)
                        

                            self.dat[animal][day][sub_file]['on_off_index'] = on_off_scores(self, animal, day, sub_file)
                            self.dat[animal][day][sub_file]['peak_response_idx'] = response_speed(self, animal, day, sub_file)
                            self.dat[animal][day][sub_file]['response_amplitude'] = response_amplitude(self, animal, day, sub_file)

        #dump(self,  os.path.join(self.save_path, 'dat.z'))
        #avg_onoff_response(self) # needs to get fixed
        # osi_pd_hist(self)

