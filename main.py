from helpers import *
from behavioural_analysis import *

class DataAnalysis:

    def __init__(self, path, dict_animals_days, pose_likelihood_threshold = 0.7, response_type = 'deconvolved', fps = 20, facemap = False):

        self.dict_animals_days = dict_animals_days
        self.data_path = os.path.join(path, 'data')
        self.save_path = os.path.join(path, 'figures')
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
                        responses, iscell, ops, stat = suite2p_files (suite2p_path, response_type =self.response_type)
                        spatial_footprint(suite2p_path)

                        # only keep ROIs that qualify as cells
                        # z-score responses > z = (x-mu)/sigma) ; where mu = mean of the responses, sigma = standard deviation of the responses
                        if self.facemap:
                            self.dat[animal][day][sub_file]['raw_responses'] = zscore(responses[iscell == 1], axis=1)
                        else:
                            self.dat[animal][day][sub_file]['responses'] = zscore(responses[iscell == 1], axis=1)
                            # un comment when ready - takes long
                            #self.dat_subject[day][sub_file]['rastermap_responses'] = fit_rastermap(self.dat_subject[day][sub_file]['responses'])

                        # load ttl / neural information
                        ttl_file = [file for file in os.listdir(os.path.join(self.data_path, animal, day, recording, sub_file, 'experiments')) if (file.endswith('.mat') and (not file.endswith('realtime.mat')))][0]
                        ttl_path = os.path.join(self.data_path, animal, day, recording, sub_file, 'experiments', ttl_file)
                        self.ttl_file = loadmat(ttl_path)
                        event_id = np.squeeze(loadmat(ttl_path)['info']['event_id'][0][0])
                        self.dat[animal][day][sub_file]['ttl_data'] = np.squeeze(loadmat(ttl_path)['info']['frame'][0][0])  # [event_id == 1]
                        end_recording = self.dat[animal][day][sub_file]['ttl_data'][-1]

                        # after converting .mj2 file to .avi file (with openCV) and performing pose estimation (with DLC) (in VScode notebook)
                        # note that camera acquires at same rate as 2p - so raw files is the entire length of the recroding

                        behav_path = os.path.join(self.data_path, animal, day, recording, sub_file, 'behav')
                        self.dat[animal][day][sub_file]['poses'] = load_dlc_pose_estimates(behav_path, self.pose_likelihood_threshold)

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
                            self.dat[animal][day][sub_file]['rastermap_responses'] = fit_rastermap (self.dat[animal][day][sub_file]['responses'])

                        # now need to fix & parse ttl data
                        get_neuronal_responses_ttls(self.dat[animal][day][sub_file], self.fps)
                        #
                        # # calculating orientation selectivity > average responses to the same orientation within the same 'block'
                        # responses_gratings(self, day, sub_file)
                        #
                        # get_complex_num(self)



def spatial_footprint (suite2p_path):
    responses, iscell, ops, stat = suite2p_files(suite2p_path, response_type='deconvolved')
    stat_iscell = stat[iscell == 1]
    plt.figure()
    im = np.zeros((ops['Ly'], ops['Lx']))
    for n in range(0, stat_iscell.shape[0]):
        ypix = stat_iscell[n]['ypix'][~stat_iscell[n]['overlap']]
        xpix = stat_iscell[n]['xpix'][~stat_iscell[n]['overlap']]
        im[ypix, xpix] = n + 1
    plt.axis('off')
    plt.imshow(im, cmap='magma')
    plt.title(suite2p_path.split('\\')[3])
    animal, day = suite2p_path.split('\\')[3], suite2p_path.split('\\')[4]
    plt.savefig(fr'G:\vision_restored\figures\{animal}, {day}.png')
    plt.show()

#spatial_footprint(r'E:\vision_restored\data\EC_RD1_01\20240527\gra\gra_000_100\experiments\suite2p\plane0')

animals_days = {'EC_RD1_01': ['20240627'], 'EC_RD1_02': ['20240627'],'EC_RD1_03': ['20240627'],'EC_RD1_04': ['20240627']}
obj = DataAnalysis ('G:\\vision_restored', dict_animals_days = animals_days, response_type = 'fluorescence')

brightness = 100
# dat_dict = dat.dat_subject[day][f'gra_000_{str(brightness)}']
# #print(f"{dat_dict['responses'].shape[0]} cells")
#
# #plot_raw_responses (dat, '20240527', 'gra_000_100')
# plot_tuning_curves (dat, day, f'gra_000_{str(brightness)}')
# plot_rasters(dat, day, f'gra_000_{str(brightness)}')
#
#
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
