from imports import *


def suite2p_files (suite2p_path, response_type = 'deconvolved'):
    '''
    :param response_type: 'deconvolved' or 'fluorescence' - specifies if we want deconvolved (binarized) traces or raw fluorescnece values

    :param suite2p_path: path to the suite2p plane0 folder
    :return: neuronal responses file, is cell file, ops file, stat file

    - F.npy > array of fluorescence traces (ROIs by timepoints)
    - Fneu.npy > array of neuropil fluorescence traces (ROIs by timepoints)
    - spks.npy > array of deconvolved traces (ROIs by timepoints)
    - stat.npy > list of statistics computed for each cell (ROIs by 1)
    - ops.npy > options and intermediate outputs (dictionary)
    -
    '''

    if response_type == 'deconvolved': # spks.npy > array of deconvolved traces (ROIs by timepoints) (generally cleaner)
        responses_file = np.load(os.path.join(suite2p_path, 'spks.npy'), allow_pickle=True)  # array of fluorescence traces (cells x timepoints)
    elif response_type == 'fluorescence': # F.npy > array of fluorescence traces (ROIs by timepoints)
        # if you load in F, you need to subtract the neuropil (F - 0.7*Fneu) > this is the real fluorescence signal
        F = np.load(os.path.join(suite2p_path, 'F.npy'),allow_pickle=True)  # array of fluorescence traces (cells x timepoints)
        Fneu = np.load(os.path.join(suite2p_path, 'Fneu.npy'), allow_pickle=True)  # array of fluorescence traces (cells x timepoints)

        responses_file = F - (0.7 * Fneu)  # array of fluorescence traces (cells x timepoints)

    #iscell.npy > specifies whether an ROI is a cell, first column is 0/1, and second column is probability that the ROI is a cell based on the default classifier
    iscell_file = np.load(os.path.join(suite2p_path, 'iscell.npy'), allow_pickle=True)[:,0]  # specifies whether an ROI is a cell, either 0/1

    # ops.npy > options and intermediate outputs (dictionary)
    ops_file = np.load(os.path.join(suite2p_path, 'ops.npy'), allow_pickle=True).item()

    #stat.npy > list of statistics computed for each cell (ROIs by 1)
    stat_file = np.load(os.path.join(suite2p_path, 'stat.npy'), allow_pickle=True)

    return responses_file, iscell_file, ops_file, stat_file

def load_stims (path_to_log):
    '''
    Reads all lines in log file
    :param path_to_log:
    :return:
    - stim_dict: ex, {'Wait 0': ['Wait600.00'], 'GratingStim 0': ['grating_143_SF0.02_TF1.00', 'grating_323_SF0.02_TF1.00',...]...}
    - list_stim_types: ex, ['Wait', 'GratingStim', 'GratingStim', 'GratingStim', 'GratingStim', 'GratingStim', 'GratingStim', ...]
    - list_stim_names ex, ['Wait600.00', 'grating_143_SF0.02_TF1.00', 'grating_323_SF0.02_TF1.00', 'grating_359_SF0.02_TF1.00',...]
    '''

    # read through all the line in the log file, and add each line to 'list_stims'
    list_stims = []
    with open(path_to_log) as f:
        lines = f.readlines()  # all the lines in the log file

        for i, l in enumerate(lines):  # enumerate through all lines in log file
            if i > 0:  # ignore first line, which just has info on experiment
                l_new = ast.literal_eval(l)
                list_stims.append(l_new)  # store in list

    # store the name and type of each stimulus in sepearate lists
    list_stim_types = [stim['type'] for stim in list_stims]
    list_stim_names = [stim['name'] for stim in list_stims]

    #number of orientations in a 'repeat'
    ntheta = len(np.unique(list_stim_names)) - 2 # subtract 1 to account for two 'wait' times of different duration
    nrepeats = int((len(list_stim_names)-1)/ntheta)

    # create a dictionary where we sort the stimulus block together with the names of the stimuli (wait stims together, and grating stims together, in blocks of ntheta)
    stim_dict = {}
    wait_count, grating_count = 0, 0
    for i_stim, stim in enumerate(list_stim_types):

        if 'Wait' in stim:
            stim_dict[f'{stim} {wait_count}'] = [list_stim_names[i_stim]]
            wait_count += 1

        elif ('Grating' in stim):
            if (grating_count%ntheta) == 0:
                list_gratings = []
                for i_grating in np.arange(i_stim, i_stim+ntheta):
                    list_gratings.append(list_stim_names[i_grating])
                stim_dict[f'{stim} {grating_count//ntheta}'] = list_gratings
            grating_count += 1

    #return stim_dict, list_stim_types[1:], list_stim_names[1:], ntheta, nrepeats
    return stim_dict, list_stim_types, list_stim_names, ntheta, nrepeats

def check_ttls (ttl_arr, stim_names_list):
    '''
    Check if ttls are aligned

    :param ttl_arr:
    :return: corrected ttl_array, if necessary
    '''

    difference = np.diff(ttl_arr)

    n_ttls_needed = len(stim_names_list) * 2

    # since each recording is flanked by a long wait time, the first and last 'difference' period should be very long
    # check that the wait times are each over 400 frames long and that all ttl periods are less than 100 frames long
    if (difference[0] > 400) and (difference[-1] > 400) and (np.all(np.diff(ttl_arr)[1:-1] < 100)):
        if len(ttl_arr) + 1 == n_ttls_needed: # missing ONE ttl between end of wait and first stim
            ttl_arr = np.insert(ttl_arr, 2, ttl_arr[1])

    elif (len(ttl_arr) + 2 == n_ttls_needed): # have 2 less ttls than we need (missing start of wait and end of wait)
        #add duplicate 'start'
        ttl_arr = np.insert(ttl_arr, 0, ttl_arr[0])

        # there is a ttl missing, so we'd want to add a 0 at the start
        ttl_arr = np.insert(ttl_arr, 0, 0)

    return ttl_arr

def zip_ttl_data(object_datsubject_day):
    '''
    Put ttl data into a tuple of (stim_type, stim_name, [ttl_start, ttl_stop], ...)

    :param object_datsubject_day: example: suite2p_obj.dat_subject['20231106']
    :return:
    '''

    stim_names = object_datsubject_day['log']['list_stim_names']                            # list of stimuli names, ex: [ 'Wait600.00', 'ILSVRC2012_val_00000385.JPEG', 'grating_240_SF0.02_TF1.00'...]
    stim_types = object_datsubject_day['log']['list_stim_types']                            # list of stimuli types, ex: [ 'Wait', 'ImageStim', 'GratingStim'...]
    object_datsubject_day['ttl_data'] = check_ttls (object_datsubject_day['ttl_data'], stim_names)
    ttl = object_datsubject_day['ttl_data']                                                 # 1d array of TTLs, after being corrected for data collection errors, ex:      [ 102 6086 6086 6094...]

    zipped_ttl_dat = []
    for i, (stim_name, stim_type) in enumerate(zip(stim_names, stim_types)):
        zipped_ttl_dat.append((stim_type, stim_name, [ttl[i * 2], ttl[(2 * i) + 1]]))

    return zipped_ttl_dat


def get_ttl_dicts (obj_dat_subject_day):
    '''

    :param obj_dat_subject_day: example: suite2p_obj.dat_subject['20231106']
    :return:
        - dict_stim_names (dict): dictionary with stimulus type (key) and list of stimuli names (value) for each 'block'
        - dict_stim_ttls (dict): dictionary with stimulus type (key) and array of ttls (value) for each 'block' (shape n_stim_presentations x 2)
    '''

    ttl_tuple = zip_ttl_data(obj_dat_subject_day)

    obj_dat_subject_day['dict_stim_names'] = {}
    obj_dat_subject_day['dict_stim_ttls'] = {}

    stim_names, stim_ttls = {}, {}
    wait_count, grating_count = 0, 0
    for i_zip in range(len(ttl_tuple)):

        if 'Wait' == ttl_tuple[i_zip][0]:
            type = ttl_tuple[i_zip][0]  # stimulus type, eg 'WaitStim'
            obj_dat_subject_day['dict_stim_names'][f'{type}_{wait_count}'] = ttl_tuple[i_zip][1] #ttl_tuple[i_zip][1] # specifications, eg Grating_90_sf0.02, etc
            obj_dat_subject_day['dict_stim_ttls'][f'{type}_{wait_count}'] = [ttl_tuple[i_zip][2]]  #ttl_tuple[i_zip][2] #list of ttls
            wait_count += 1

        # grating stim, and the remainder is 1 (to account for the wait)
        elif (i_zip % obj_dat_subject_day['ntheta']) == 1:
            #print('ooop', print(i_zip))
            z_names = [ttl_tuple[i][1] for i in range(i_zip, i_zip + obj_dat_subject_day['ntheta'])]
            z_ttls = [ttl_tuple[i][2] for i in range(i_zip, i_zip + obj_dat_subject_day['ntheta'])]

            type = ttl_tuple[i_zip][0]                  # stimulus type, eg 'GratingStim'
            obj_dat_subject_day['dict_stim_names'][f'{type}_{grating_count}'] = z_names # specifications, eg Grating_90_sf0.02, etc
            obj_dat_subject_day['dict_stim_ttls'][f'{type}_{grating_count}'] = z_ttls #list of ttls
            grating_count += 1

    return obj_dat_subject_day['dict_stim_ttls']


def get_neuronal_responses_ttls (obj_animal_day_session, fps):
    '''
    Indexes the neuronal responses according to the ttl values for each stimulus

    :param obj: ex: suite2p_obj
    :param day (str): yearmonthday, ex: '20231106'
    :return: void - modifies obj in place
    '''

    # load corrected ttl data into a dictionary with stim type (key) and array of ttls (value) (shape: n_stim_presentations x 2)

    dict_stim_ttls = get_ttl_dicts(obj_animal_day_session)

    fluorescence = obj_animal_day_session['responses']

    # create a dictionary that holds the neural activity for each pair of TTLs (in chronological order)
    obj_animal_day_session['responses_ttls'] = {}
    obj_animal_day_session['responses_ttls_whole'] = {}

    for stim in dict_stim_ttls.keys():
        obj_animal_day_session['responses_ttls'][stim] = []
        obj_animal_day_session['responses_ttls_whole'][stim] = []

        for [start_ttl, end_ttl] in dict_stim_ttls[stim]:
            if 'Wait' in stim:  # take entire wait period
                obj_animal_day_session['responses_ttls'][stim].append(fluorescence[:, int(start_ttl):int(end_ttl)])
            elif 'Grating' in stim:
                # 1s static + 3s moving + 1s off

                # only use 3 sec moving
                obj_animal_day_session['responses_ttls'][stim].append(fluorescence[:, int(start_ttl + (fps * 1)): int(end_ttl - (fps * 1))])

                # use 1 sec before stim + 1 sec static + 3 sec moving
                obj_animal_day_session['responses_ttls_whole'][stim].append(fluorescence[:, int(start_ttl - (fps * 1)): int(end_ttl - (fps * 1))])
            elif 'Image' in stim:  # 0.5s on + 1.5s off + 1.3-1.7s jitter > only use 0.5s on
                obj_animal_day_session['responses_ttls'][stim].append(fluorescence[:, int(start_ttl):int(start_ttl + (fps * 0.5))])

        # broadcasting list into array
        if 'Grating' in stim:  # in order to have an array of shape stimulus x cells x time, everything needs to be the same shape
            max_size = min([l.shape[-1] for l in obj_animal_day_session['responses_ttls'][stim]])
            obj_animal_day_session['responses_ttls'][stim] = np.array([n[:, :max_size] for n in obj_animal_day_session['responses_ttls'][stim]])

            max_size = min([l.shape[-1] for l in obj_animal_day_session['responses_ttls_whole'][stim]])
            obj_animal_day_session['responses_ttls_whole'][stim] = np.array([n[:, :max_size] for n in obj_animal_day_session['responses_ttls_whole'][stim]])

        obj_animal_day_session['responses_ttls'][stim] = np.array(obj_animal_day_session['responses_ttls'][stim])
        obj_animal_day_session['responses_ttls_whole'][stim] = np.array(obj_animal_day_session['responses_ttls_whole'][stim])

def responses_gratings(object, day, subfile):
    '''
    In order to calculate orientation tuning selectivity
    - average neural responses to the same orientation within the same block
    - order neural responses to gratings of increasing orientation

    'mean_ordered_grat' is an array of shape (n_blocks, n_orientations_shown), which stores orientation grating angles in increasing order
    'mean_ordered_grat_responses' is an array of shape (n_blocks, n_orientations_shown, n_cells, n_frames), which stores the neural response to each orientation in 'mean_ordered_grat'

    :param object: data object
    :param n_grating_repeats: For each grating stimulus 'block', each orientation is repeated a certain number of times.
    :return:
    '''

    dict_stim = object.dat_subject[day][subfile]['dict_stim_names']
    f = object.dat_subject[day][subfile]['responses_ttls']
    f_whole = object.dat_subject[day][subfile]['responses_ttls_whole']

    angles = np.array([[int(d[8:-14]) for d in dict_stim[g]] for g in [k for k in dict_stim if
                                                                       'GratingStim' in k]])  # array of shape n_grating_blocks (4) x n_orientations_in_block (12)
    responses = np.array([f[response] for response in f if 'GratingStim' in response])  # array of shape n_repeats (6) x n_orientations_in_repeat (8) x n_neurons x n_frames
    responses_whole = np.array([f_whole[response] for response in f_whole if 'GratingStim' in response])  # array of shape n_repeats (6) x n_orientations_in_repeat (8) x n_neurons x n_frames

    sort_i = np.argsort(angles, axis=1)  # indices on which to sort the 'angles' & 'responses' array

    # sorting angles / responses in order of increasing orientation
    sorted_angles = np.array([angles[i][sort_i[i]] for i in range(angles.shape[0])])  # ex: 0,0,30,30,60,60,...
    sorted_responses = np.array([responses[i][sort_i[i]] for i in range(angles.shape[0])])  # indexing neural responses so we can index/sort them similarly to to 'sorted_angles'
    sorted_responses_whole = np.array([responses_whole[i][sort_i[i]] for i in range(angles.shape[0])])  # indexing neural responses so we can index/sort them similarly to to 'sorted_angles'

    object.dat_subject[day][subfile]['mean_ordered_grat'] = sorted_angles
    object.dat_subject[day][subfile]['mean_ordered_grat_responses'] = sorted_responses
    object.dat_subject[day][subfile]['mean_ordered_grat_responses_whole'] = sorted_responses_whole

def normalize_arr(x):
    '''
    Scale/normalize values in array between 0 & 1, along the cell/row axis
    :return:
    '''
    return np.array([(row - row.min()) / (row.max() - row.min()) for row in x])


def plot_raw_responses (obj, day, session):
    data_array = obj.dat_subject[day][session]['responses']
    ttl = obj.dat_subject[day][session]['ttl_data']

    fig, ax = plt.subplots(figsize=(16, 9))
    fig.tight_layout()
    global_counter = 0
    for i_cell, cell in enumerate(data_array):
        colors = plt.cm.turbo(np.linspace(0, 1, data_array.shape[0]))
        plt.plot(global_counter + cell, color=colors[i_cell])
        global_counter += 0.5
        [ax.axvline(x, c= 'grey', ls = '--', linewidth = 0.8, alpha = 0.4) for x in np.unique(ttl)]
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title(f'{obj.animal})', fontsize=15)
    ax.set_xlabel('Time', fontsize=14)
    ax.set_ylabel('Cell #', fontsize=14)
    plt.subplots_adjust(top=0.95)
    plt.show()


def get_complex_num(obj):
    '''
    calculate tuning curves > sum [r_theta * e ^ (i2theta)] / sum [r_theta]
    theta = direction of the kth condition (array of all the directions)
    r_theta = neuronal responses to the directions

    average the responses
    '''

    for day in obj.dat_subject.keys():
        for session in obj.dat_subject[day].keys():

            theta = obj.dat_subject[day][session]['mean_ordered_grat'].mean(axis = 0) # > shape: n_orientations
            obj.dat_subject[day][session]['theta'] = theta

            # mean response to the stimulus (average over frames) > shape: n_cells x n_orientations
            responses_theta = ((obj.dat_subject[day][session]['mean_ordered_grat_responses'].mean(axis = 0)).mean(axis = -1)).T
            # np.exp outputs shape n_orientations

            obj.dat_subject[day][session]['tuning_curves'] = responses_theta

            # in python, j is the imaginary component
            complex_nums = (responses_theta * np.exp(1j * 2 * theta)).sum(axis=-1) / (responses_theta.sum(axis=-1))  # > shape n_cells

            obj.dat_subject[day][session]['complex'] = complex_nums

            # magnitude (absolute value) of the complex number > represents the amplitude of the response (how strongly neuron responds) = OSI (should be between 0 & 1)
            obj.dat_subject[day][session]['OSI'] = np.abs(complex_nums)

            # the argument (angle / preferred angle) of the complex number encodes the preferred orientation (the orientation to which the neuron responds most strongly).
            obj.dat_subject[day][session]['pref_orientation'] = np.rad2deg(np.angle(complex_nums))  # preferred angle
            # divide by 2 because of the i*theta*2 in the original equation



def plot_tuning_curves (object, day, session):
    orientations = object.dat_subject[day][session]['theta']
    tuning_curves_mean = ((object.dat_subject[day][session]['mean_ordered_grat_responses'].mean(axis=0)).mean(axis=-1)).T
    tuning_curves = object.dat_subject[day][session]['mean_ordered_grat_responses'].mean(axis=-1) # shape n_repeats, n_ori, n_cells

    # variables and dependencies for colour mapping
    plasma = plt.get_cmap('plasma')
    cNorm  = colors.Normalize(vmin=0, vmax=tuning_curves.shape[0]+1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=plasma)
    scalarMap.set_array([])

    n_cells = tuning_curves_mean.shape[0]
    nrows = 7
    ncols = 5
    n_pages = int(np.ceil(n_cells / (nrows*ncols)))

    for n_page in range(n_pages):
       cells = np.arange((n_page + 0) * nrows * ncols, (n_page + 1) * nrows * ncols)
       fig, ax = plt.subplots (nrows, ncols, figsize = (18,9))
       ax = ax.ravel()

       for i, i_cell in enumerate(cells):
           if i_cell < n_cells:
               ax[i].plot(orientations, tuning_curves_mean[i_cell], c = 'black', linewidth = 1.5)
               [ax[i].plot(orientations, tuning_curves[repeat,:,i_cell], c = scalarMap.to_rgba(repeat), alpha = 0.5, linewidth = 0.8) for repeat in range(tuning_curves.shape[0])]

               # # mean response across frames for each block / orientation / cell
               ax[i].set_xlabel('Orientation (deg)', fontsize = 8)
               #ax[i_cell].set_ylabel(f'Response')
               #ax[i_cell].set_title(f'ROI {i_cell})')
               ax[i].set_xticks(orientations[::2])
           if i_cell >= n_cells:
               ax[i].axis('off')
       plt.tight_layout()

       save_folder = os.path.join(object.save_path, object.animal, day, session, 'tuning_curves')
       if not os.path.exists(save_folder):
           os.makedirs(save_folder)

       plt.savefig(os.path.join(save_folder, f'tuning curve pg {n_page}.png'))
       #plt.show()


def plot_rasters(object, day, session):
    orientations = object.dat_subject[day][session]['theta']

    # shape n_timepoints, n_cells, n_orientations
    tuning_curves_mean = ((object.dat_subject[day][session]['mean_ordered_grat_responses_whole'].mean(axis=0))).T

    n_cells = tuning_curves_mean.shape[1]
    nrows = 8
    ncols = 5
    n_pages = int(np.ceil(n_cells / (nrows*ncols)))

    for n_page in range(n_pages):
        cells = np.arange((n_page + 0) * nrows * ncols, (n_page + 1) * nrows * ncols)

        fig, ax = plt.subplots (nrows, ncols, figsize = (18,9))
        ax = ax.ravel()

        for i, i_cell in enumerate(cells):
            if i_cell < n_cells:
                ax[i].imshow(tuning_curves_mean[:, i_cell, :].T, vmin = tuning_curves_mean.min(), vmax = tuning_curves_mean.max())
                ax[i].set_xlabel('Time (s)', fontsize = 8)
                ax[i].set_ylabel(f'Ori (deg)', fontsize = 8)
                #ax[i_cell].set_title(f'ROI {i_cell})')
                ax[i].set_xticks(np.arange(tuning_curves_mean.shape[0])[::int(object.fps)])
                ax[i].set_xticklabels([int(np.round(x)) for x in (np.arange(-object.fps, tuning_curves_mean.shape[0]-object.fps)[::int(object.fps)])/int(object.fps)])
                ax[i].set_yticks(np.arange(tuning_curves_mean.shape[-1])[::4])
                ax[i].set_yticklabels ([int(x) for x in object.dat_subject[day][session]['theta'][::4]])
                ax[i].axvline (int(object.fps), c = 'white')
                ax[i].axvline(2*int(object.fps), c='white')
            if i_cell >= n_cells:
                ax[i].axis('off')

        plt.tight_layout()

        save_folder = os.path.join(object.save_path, object.animal, day, session, 'rasters')
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        plt.savefig(os.path.join(save_folder, f'rasters pg {n_page}.png'))

        #plt.show()


def regress_out_behav(rawY, predX):
    '''
    Project out neural activity predicted from behaviour (from facemap) from the true neural activity, using Linear Regression
    Remove the part of the neural activity that can be explained by the predicted spontaneous behaviour
    :param rawY: raw neural activity, numpy matrix of shape n_neurons x n_frames
    :param predX: predicted neural activity from behavioural variables, numpy matrix of shape n_neurons x n_frames
    :return:
    '''
    n_neurons = rawY.shape[0]
    residuals = np.zeros_like(rawY)

    # for each neuron, perform a linear regression: regress its true activity on its predicted behavioural activity
    for i in range(n_neurons):
        y = rawY[i, :]
        x = predX[i, :]

        # perform linear regression: y = a * x + b > np.polyfit finds the best fit line / coefficients that explain the spontaneous behaviour
        a, b = np.polyfit(x, y, 1)

        # Calculate the predicted activity based on the regression
        y_pred = a * x + b

        # Calculate the residual (stimulus-related activity, or neural activity not explained by the spontaneous behaviour)
        # subtract predicted neural activity (based on regression coefficients) from true neural activity
        residuals[i, :] = y - y_pred

    return residuals


def fit_rastermap (response_array):
    '''
    Visualization framework for neural data > sorts neual responses along a one-dimensional manifold
    https://github.com/MouseLand/rastermap
    https://www.biorxiv.org/content/10.1101/2023.07.25.550571v2
    :param response_array: array of neurons x time
    :return:
    '''
    # fit rastermap (https://github.com/MouseLand/rastermap)
    # https://www.biorxiv.org/content/10.1101/2023.07.25.550571v2
    model = Rastermap(n_PCs=200, n_clusters=100, locality=0.75, time_lag_window=5).fit(response_array, compute_X_embedding=True)
    y = model.embedding  # neurons x 1
    isort = model.isort
    X_embedding = model.X_embedding             # visualize binning over neurons

    return X_embedding
