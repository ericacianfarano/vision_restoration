import matplotlib.pyplot as plt

from helpers import *

def plot_raw_responses (object, animal):
    '''
    Plot the trace of 20 random cells + the start of each TTL (grey dotted line)
    :param object:
    :param animal:
    :return:
    '''

    n_cells_to_plot = 20
    first_minutes = 8

    for day in object.dat[animal].keys():
        for session in object.dat[animal][day].keys():

            # there is only one SF and TF
            if (object.dat[animal][day][session]['n_SF'] > 0) and (object.dat[animal][day][session]['n_TF'] > 0) and (object.dat[animal][day][session]['n_theta'] > 0):

                data_array = object.dat[animal][day][session]['responses'][:, :first_minutes*60*object.fps]
                ttl = object.dat[animal][day][session]['ttl_data']
                cells = [int(num) for num in np.linspace(0, data_array.shape[0]-1, n_cells_to_plot)]

                fig, ax = plt.subplots(figsize=(16, 9))
                colors = plt.cm.turbo(np.linspace(0, 1, data_array.shape[0]))
                fig.tight_layout()
                global_counter = 0
                for i_cell, cell in enumerate(cells):
                    plt.plot( global_counter+data_array[cell,:]/data_array[cell,:].max(), color=colors[cell])
                    global_counter += 1
                    [ax.axvline(x, c= 'grey', ls = '--', linewidth = 0.8, alpha = 0.4) for x in np.unique(ttl) if x < (first_minutes*60*object.fps)]
            else:
                data_array = object.dat[animal][day][session]['responses']
                ttl = np.array([l[2][0] for l in object.dat[animal][day][session]['dict_stim_ttls'].values()])
                cells = [int(num) for num in np.linspace(0, data_array.shape[0] - 1, n_cells_to_plot)]
                #print(ttl)

                fig, ax = plt.subplots(figsize=(16, 9))
                colors = plt.cm.turbo(np.linspace(0, 1, data_array.shape[0]))
                fig.tight_layout()
                global_counter = 0
                for i_cell, cell in enumerate(cells):
                    plt.plot(global_counter + data_array[cell, :] / data_array[cell, :].max(), color=colors[cell])
                    global_counter += 1
                    [ax.axvline(x, c='grey', ls='--', linewidth=0.8, alpha=0.4) for x in ttl]

            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_title(f'{animal} ({day}, {session})', fontsize=15)
            ax.set_xlabel('Time', fontsize=14)
            ax.set_ylabel('Cell #', fontsize=14)
            plt.subplots_adjust(top=0.95)

            if not os.path.exists(os.path.join(object.save_path, 'responses', animal)):
                os.makedirs(os.path.join(object.save_path, 'responses', animal))

            plt.savefig(os.path.join(object.save_path,'responses',animal, f'responses-{animal}_{session}_{day}.png' ))

            if not object.show_plots:
                plt.close('all')

def spatial_footprint (object, animal, day, recording, sub_file):
    suite2p_path = os.path.join(object.data_path, animal, day, recording, sub_file, 'experiments', 'suite2p', 'plane0')

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
    animal, day, session = suite2p_path.split('\\')[3], suite2p_path.split('\\')[4], suite2p_path.split('\\')[6]

    if not os.path.exists(os.path.join(object.save_path, 'spatial footprints')):
        os.mkdir(os.path.join(object.save_path, 'spatial footprints'))

    plt.savefig(os.path.join(object.save_path, 'spatial footprints', f'sf-{animal}-{day}-{session}.png' ))
    if not object.show_plots:
        plt.close('all')


def plot_tuning_curves (object, animal, day, session):

    if (object.dat[animal][day][session]['n_SF'] == 1) and (object.dat[animal][day][session]['n_TF'] == 1):
        orientations = np.squeeze(object.dat[animal][day][session]['theta'])
        #tuning_curves_mean = np.squeeze(((object.dat[animal][day][session]['mean_ordered_grat_responses'].mean(axis=0)).mean(axis=-1)).T)
        tuning_curves_mean = object.dat[animal][day][session]['tuning_curves'].mean(axis=0).T  # mean across repeats
        tuning_curves = np.squeeze(object.dat[animal][day][session]['mean_ordered_grat_responses'].mean(axis=-1)) # shape n_repeats, n_ori, n_cells

        #plt.savefig(os.path.join(object.save_path, 'tuning curves', animal, f'tc#{n_page}-{animal}-{day}-{session}.png'))
    elif (object.dat[animal][day][session]['n_theta'] > 1) and (object.dat[animal][day][session]['n_SF'] > 1):  # there is more than one orientaiton or SF
        orientations = object.dat[animal][day][session]['theta']
        tuning_curves_mean = object.dat[animal][day][session]['tuning_curves'].mean(axis = 0).T # mean across repeats
        tuning_curves = object.dat[animal][day][session]['tuning_curves'] # shape n_repeats, n_ori, n_cells

        # orientations = object.dat[animal][day][session]['theta']
        # tuning_curves_mean = object.dat[animal][day][session]['tuning_curves']  # mean across repeats
        #print(tuning_curves_mean.shape)
        #tuning_curves = object.dat[animal][day][session]['tuning_curves']  # shape n_repeats, n_ori, n_cells

    else:
        return

    plasma = plt.get_cmap('plasma')
    cNorm = colors.Normalize(vmin=0, vmax=tuning_curves.shape[0] + 1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=plasma)
    scalarMap.set_array([])

    n_cells = tuning_curves_mean.shape[0]
    nrows = 7
    ncols = 5
    n_pages = int(np.ceil(n_cells / (nrows * ncols)))

    if not os.path.exists(os.path.join(object.save_path, 'tuning curves', animal)):
        os.makedirs(os.path.join(object.save_path, 'tuning curves', animal))

    with PdfPages(os.path.join(object.save_path, 'tuning curves', animal,
                               f'{animal} tuning curves ({day}, {session}).pdf')) as pdf:

        for n_page in range(n_pages):
            cells = np.arange((n_page + 0) * nrows * ncols, (n_page + 1) * nrows * ncols)
            fig, ax = plt.subplots(nrows, ncols, figsize=(18, 9))
            ax = ax.ravel()

            for i, i_cell in enumerate(cells):
                if i_cell < n_cells:
                    # print(orientations.shape, tuning_curves_mean[i_cell].shape)
                    ax[i].plot(orientations, tuning_curves_mean[i_cell], c='black', linewidth=1.5)
                    [ax[i].plot(orientations, tuning_curves[repeat, :, i_cell], c=scalarMap.to_rgba(repeat), alpha=0.5,
                                linewidth=0.8) for repeat in range(tuning_curves.shape[0])]

                    # # mean response across frames for each block / orientation / cell
                    ax[i].set_xlabel('Orientation (deg)', fontsize=8)
                    # ax[i_cell].set_ylabel(f'Response')
                    # ax[i_cell].set_title(f'ROI {i_cell})')
                    ax[i].set_xticks(orientations[::2])
                    ax[i].set_title(f'OSI: {np.round(object.dat[animal][day][session]["OSI"][i_cell], 3)}')
                if i_cell >= n_cells:
                    ax[i].axis('off')
            plt.tight_layout()
            pdf.savefig()
            if not object.show_plots:
                plt.close()


def plot_rasters(object, animal, day, session):

    if (object.dat[animal][day][session]['n_SF'] == 1) and (object.dat[animal][day][session]['n_TF'] == 1):
        orientations = np.squeeze(object.dat[animal][day][session]['theta'])

        # shape n_timepoints, n_cells, n_orientations
        tuning_curves_mean = np.squeeze(((object.dat[animal][day][session]['mean_ordered_grat_responses_whole'].mean(axis=0))).T)

        n_cells = tuning_curves_mean.shape[1]
        nrows = 8
        ncols = 5
        n_pages = int(np.ceil(n_cells / (nrows*ncols)))

        if not os.path.exists(os.path.join(object.save_path, 'rasters', animal)):
            os.makedirs(os.path.join(object.save_path, 'rasters', animal))

        with PdfPages(os.path.join(object.save_path, 'rasters', animal, f'{animal} rasters ({day}, {session}).pdf')) as pdf:

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
                        ax[i].set_yticklabels ([int(x) for x in orientations[::4]])
                        ax[i].axvline (int(object.fps), c = 'white')
                        ax[i].axvline(2*int(object.fps), c='white')
                    if i_cell >= n_cells:
                        ax[i].axis('off')

                plt.tight_layout()
                pdf.savefig()
                if not object.show_plots:
                    plt.close()

                #plt.savefig(os.path.join(object.save_path, 'rasters', f'rasters#{n_page}-{animal}-{day}-{session}.png'))
    # else:

def plot_activity_rasters (object, animal, day, session):

    ttls_whole = object.dat[animal][day][session]['responses_ttls_whole']

    # shape repeats x n_orientations x n_cells x timepoints (1s before + 1s static + 3s moving)
    stacked = np.array([ttls_whole[key] for key in [t for t in ttls_whole.keys() if 'Grating' in t]])

    # reshape to stack repeats and orientations
    stacked = stacked.reshape(stacked.shape[0]*stacked.shape[1], stacked.shape[2], -1)

    n_cells = stacked.shape[1]
    nrows = 7
    ncols = 7
    n_pages = int(np.ceil(n_cells / (nrows * ncols)))

    if not os.path.exists(os.path.join(object.save_path, 'activity rasters', animal)):
        os.makedirs(os.path.join(object.save_path, 'activity rasters', animal))

    with PdfPages(os.path.join(object.save_path, 'activity rasters', animal, f'{animal} activity rasters ({day}, {session}).pdf')) as pdf:

        for n_page in range(n_pages):
            cells = np.arange((n_page + 0) * nrows * ncols, (n_page + 1) * nrows * ncols)
            fig, ax = plt.subplots(nrows, ncols, figsize=(18, 9))
            ax = ax.ravel()

            for i, i_cell in enumerate(cells):
                if i_cell < n_cells:

                    ax[i].imshow(stacked[:,i_cell,:], vmin=0, vmax=1.5, cmap="gray_r", aspect="auto")
                    ax[i].axvline(object.fps, c = 'blue', alpha = 0.5)
                    ax[i].axvline(object.fps*2, c='red', alpha = 0.5)
                    ax[i].set_xticks([])
                    ax[i].set_yticks([])
                    ax[i].set_aspect('auto')
                    ax[i].set_title(f'ROI {i_cell}')

                if i_cell >= n_cells:
                    ax[i].axis('off')

            plt.tight_layout()
            pdf.savefig()
            plt.show()
            plt.close()


def on_off_responses(object, animal, day, session):

    # excluding first response
    response = object.dat[animal][day][session]['zscored_responses_ttls'][1:]

    # variables and dependencies for colour mapping
    plasma = plt.get_cmap('plasma')
    cNorm = colors.Normalize(vmin=0, vmax=response.shape[0] + 1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=plasma)
    scalarMap.set_array([])

    n_cells = response.shape[1]
    nrows = 7
    ncols = 5
    n_pages = int(np.ceil(n_cells / (nrows * ncols)))

    if not os.path.exists(os.path.join(object.save_path, 'on-off responses', animal)):
        os.makedirs(os.path.join(object.save_path, 'on-off responses', animal))

    with PdfPages(os.path.join(object.save_path, 'on-off responses', animal,
                               f'{animal} on-off responses ({day}, {session}).pdf')) as pdf:

        for n_page in range(n_pages):
            cells = np.arange((n_page + 0) * nrows * ncols, (n_page + 1) * nrows * ncols)
            fig, ax = plt.subplots(nrows, ncols, figsize=(18, 9))
            ax = ax.ravel()

            for i, i_cell in enumerate(cells):
                if i_cell < n_cells:

                    x = np.arange(response.shape[-1])
                    # print(orientations.shape, tuning_curves_mean[i_cell].shape)

                    [ax[i].plot(x, response[repeat, i_cell, :], c=scalarMap.to_rgba(repeat), alpha=0.4,
                                linewidth=0.8) for repeat in range(response.shape[0])]
                    ax[i].plot(x, response[:, i_cell, :].mean(axis=0), c='black', linewidth=1.5)

                    # # mean response across frames for each block / orientation / cell
                    ax[i].set_xlabel('Orientation (deg)', fontsize=8)
                    # ax[i_cell].set_ylabel(f'Response')
                    # ax[i_cell].set_title(f'ROI {i_cell})')
                    ax[i].axvline(object.fps * 0, color='grey', linewidth=1, alpha=0.5)
                    ax[i].axvline(object.fps * 4, color='black', linewidth = 1, alpha=0.5)
                    ax[i].axvline(object.fps * 8, color='red', linewidth = 1, alpha=0.5)
                    ax[i].axvline(object.fps * 12, color='black', linewidth = 1, alpha=0.5)
                    ax[i].axvline(object.fps * 16, color='grey', linewidth=1, alpha=0.5)
                    ax[i].set_xticks(x[::object.fps*4]) #int((x[::object.fps])// object.fps))


                if i_cell >= n_cells:
                    ax[i].axis('off')
            plt.tight_layout()
            pdf.savefig()
            if not object.show_plots:
                plt.close()


#data_object.dat['EC_RD1opto_04']['20241112']['small_chirp_000_006']['responses_ttls']
def osi_pd_hist (object, responsive):
    '''
    :param object:
    :param responsive: whether or not to only consider responive cells that pass the z score threshold
    :return:
    '''

    screens = ['big', 'small']
    fig, ax = plt.subplots(2, 2, sharey = True, figsize=(13, 10))

    for k, metric in enumerate(['OSI', 'DSI']):
        for i, screen in enumerate(screens):

            #np.nonzero(object.dat[animal][day][sub_file][metric] * object.dat[animal][day][sub_file]['thresholded_cells'])
            # average response across recordings, repeats, across cells
            opto = list(chain.from_iterable(
                [object.dat[animal][day][sub_file][metric]
                 for sub_file in object.dat[animal][day].keys() if (('SFxO' in sub_file) and (screen in sub_file))]
                for animal in object.dat.keys() if 'opto' in animal
                for day in object.dat[animal].keys() if (day in object.dict_animals_days[animal])))
            rd1 = list(chain.from_iterable(
                [object.dat[animal][day][sub_file][metric]
                 for sub_file in object.dat[animal][day].keys() if (('SFxO' in sub_file) and (screen in sub_file))]
                for animal in object.dat.keys() if (('opto' not in animal) and ('RD1' in animal))
                for day in object.dat[animal].keys() if (day in object.dict_animals_days[animal])))
            control = list(chain.from_iterable(
                [object.dat[animal][day][sub_file][metric]
                 for sub_file in object.dat[animal][day].keys() if (('SFxO' in sub_file) and (screen in sub_file))]
                for animal in object.dat.keys() if ('GCaMP6s' in animal)
                for day in object.dat[animal].keys() if (day in object.dict_animals_days[animal])))

            ax[k, i].hist(list(chain.from_iterable(opto)), color='blue', alpha=0.5,
                       label=f'opto ({len(list(chain.from_iterable(opto)))} cells)', density=True,
                       bins=np.linspace(0, 1, 50))
            ax[k, i].hist(list(chain.from_iterable(rd1)), color='red', alpha=0.5,
                       label=f'rd1 ({len(list(chain.from_iterable(rd1)))} cells)', density=True,
                       bins=np.linspace(0, 1, 50))
            ax[k, i].hist(list(chain.from_iterable(control)), color='black', alpha=0.5,
                       label=f'control ({len(list(chain.from_iterable(control)))} cells)', density=True,
                       bins=np.linspace(0, 1, 50))
            if k == 0:
                #ax[k, i].set_title(f'{screen} monitor \n \n {metric} distribution')
                ax[k, i].set_title(f'Ultrabright Monitor \n \n {metric} distribution' if screen == 'small' else f'Regular Monitor \n \n {metric} distribution')
            else:
                ax[k, i].set_title(f'{metric} distribution')
            ax[k, i].set_xlabel(metric)
            ax[k, i].set_ylabel('proportion of cells')
        #ax[i].set_ylim([0,1.05])

    #plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(object.save_path, 'OSI-DSI histogram'))
    plt.show()

def responsive_cells_hist (object):

    stims = ['chirps', 'SFxO']
    screens = ['big', 'small']
    fig, ax = plt.subplots(2, 2, sharey = True, figsize=(8, 8))

    for i_stim, stim in enumerate(stims):
        for i, screen in enumerate(screens):
            # average response across recordings, repeats, across cells
            opto = np.array(list(chain.from_iterable(
                [100*object.dat[animal][day][sub_file]['thresholded_cells'].sum()/len(object.dat[animal][day][sub_file]['thresholded_cells'])
                 for sub_file in object.dat[animal][day].keys() if ((stim in sub_file) and (screen in sub_file))]
                for animal in object.dat.keys() if 'opto' in animal
                for day in object.dat[animal].keys() if (day in object.dict_animals_days[animal]))))
            rd1 = np.array(list(chain.from_iterable(
                [100*object.dat[animal][day][sub_file]['thresholded_cells'].sum()/len(object.dat[animal][day][sub_file]['thresholded_cells'])
                 for sub_file in object.dat[animal][day].keys() if ((stim in sub_file) and (screen in sub_file))]
                for animal in object.dat.keys() if (('opto' not in animal) and ('RD1' in animal))
                for day in object.dat[animal].keys() if (day in object.dict_animals_days[animal]))))
            control = np.array(list(chain.from_iterable(
                [100*object.dat[animal][day][sub_file]['thresholded_cells'].sum()/len(object.dat[animal][day][sub_file]['thresholded_cells'])
                 for sub_file in object.dat[animal][day].keys() if ((stim in sub_file) and (screen in sub_file))]
                for animal in object.dat.keys() if ('GCaMP6s' in animal)
                for day in object.dat[animal].keys() if (day in object.dict_animals_days[animal]))))

            ax[i_stim, i].scatter(np.array([0] * len(control)), control, s = 65, color = 'grey', alpha = 0.6, label = 'control')
            ax[i_stim, i].scatter(np.array([1] * len(rd1)), rd1, s = 65, color = 'red', alpha = 0.6, label = 'RD1')
            ax[i_stim, i].scatter(np.array([2] * len(opto)), opto,s = 65, color = 'blue', alpha = 0.6, label = 'opto')

            ax[i_stim, i].bar(np.array([0]), control.mean(), color = 'grey', alpha = 0.3, label = 'control')
            ax[i_stim, i].bar(np.array([1]), rd1.mean(), color = 'red', alpha = 0.3, label = 'RD1')
            ax[i_stim, i].bar(np.array([2]), opto.mean(), color = 'blue', alpha = 0.3, label = 'opto')

            #screen_label = 'regular monitor' if 'big' in screen else 'ultrabright monitor'
            ax[i_stim, i].set_title(f'{"FFF" if "chirps" in stim else "SFxO"} ({"regular monitor" if "big" in screen else "ultrabright monitor"})')

            # if i == 0:
            #     ax[i_stim, i].set_ylabel('Proportion of responsive cells (%)')
            ax[i_stim, i].set_xticks(np.array([0,1,2]))
            ax[i_stim, i].set_xticklabels(['Control', 'RD1', 'Opto'])

    fig.text(0.04, 0.5, 'Proportion of responsive cells (%)', va='center', rotation='vertical', fontsize=12)

    #plt.tight_layout()
    #plt.legend()
    plt.savefig(os.path.join(object.save_path, '% responsive cells'))
    plt.show()

def responsive_cells_project (object, project = 'RD1'):
    '''
    :param object:
    :param project: either 'RD1' or 'GNAT'
    :return:
    '''
    if project == 'RD1':
        stims = ['chirps', 'SFxO']
        screens = ['big', 'small']
        fig, ax = plt.subplots(2, 2, sharey = True, figsize=(8, 8))

        for i_stim, stim in enumerate(stims):
            for i, screen in enumerate(screens):
                # average response across recordings, repeats, across cells
                opto = np.array(list(chain.from_iterable(
                    [100*object.dat[animal][day][sub_file]['thresholded_cells'].sum()/len(object.dat[animal][day][sub_file]['thresholded_cells'])
                     for sub_file in object.dat[animal][day].keys() if ((stim in sub_file) and (screen in sub_file))]
                    for animal in object.dat.keys() if 'opto' in animal
                    for day in object.dat[animal].keys() if (day in object.dict_animals_days[animal]))))
                rd1 = np.array(list(chain.from_iterable(
                    [100*object.dat[animal][day][sub_file]['thresholded_cells'].sum()/len(object.dat[animal][day][sub_file]['thresholded_cells'])
                     for sub_file in object.dat[animal][day].keys() if ((stim in sub_file) and (screen in sub_file))]
                    for animal in object.dat.keys() if (('opto' not in animal) and ('RD1' in animal))
                    for day in object.dat[animal].keys() if (day in object.dict_animals_days[animal]))))
                control = np.array(list(chain.from_iterable(
                    [100*object.dat[animal][day][sub_file]['thresholded_cells'].sum()/len(object.dat[animal][day][sub_file]['thresholded_cells'])
                     for sub_file in object.dat[animal][day].keys() if ((stim in sub_file) and (screen in sub_file))]
                    for animal in object.dat.keys() if ('GCaMP6s' in animal)
                    for day in object.dat[animal].keys() if (day in object.dict_animals_days[animal]))))

                ax[i_stim, i].scatter(np.array([0] * len(control)), control, s = 65, color = 'grey', alpha = 0.6, label = 'control')
                ax[i_stim, i].scatter(np.array([1] * len(rd1)), rd1, s = 65, color = 'red', alpha = 0.6, label = 'RD1')
                ax[i_stim, i].scatter(np.array([2] * len(opto)), opto,s = 65, color = 'blue', alpha = 0.6, label = 'opto')

                ax[i_stim, i].bar(np.array([0]), control.mean(), color = 'grey', alpha = 0.3, label = 'control')
                ax[i_stim, i].bar(np.array([1]), rd1.mean(), color = 'red', alpha = 0.3, label = 'RD1')
                ax[i_stim, i].bar(np.array([2]), opto.mean(), color = 'blue', alpha = 0.3, label = 'opto')

                #screen_label = 'regular monitor' if 'big' in screen else 'ultrabright monitor'
                ax[i_stim, i].set_title(f'{"FFF" if "chirps" in stim else "SFxO"} ({"regular monitor" if "big" in screen else "ultrabright monitor"})')

                # if i == 0:
                #     ax[i_stim, i].set_ylabel('Proportion of responsive cells (%)')
                ax[i_stim, i].set_xticks(np.array([0,1,2]))
                ax[i_stim, i].set_xticklabels(['Control', 'RD1', 'Opto'])

        fig.text(0.04, 0.5, 'Proportion of responsive cells (%)', va='center', rotation='vertical', fontsize=12)

        #plt.tight_layout()
        #plt.legend()
        plt.savefig(os.path.join(object.save_path, '% responsive cells'))
        plt.show()

    elif project == 'GNAT':
        #stims = ['chirps', 'O']
        screens = ['big', 'small']
        fig, ax = plt.subplots(1, 2, sharey=True, figsize=(8, 8))

        for i, screen in enumerate(screens):
            # average response across recordings, repeats, across cells
            gnat = np.array(list(chain.from_iterable(
                [100 * object.dat[animal][day][sub_file]['thresholded_cells'].sum() / len(
                    object.dat[animal][day][sub_file]['thresholded_cells'])
                 for sub_file in object.dat[animal][day].keys() if ((screen in sub_file) and ('chirps' not in sub_file))]
                for animal in object.dat.keys() if ('GNAT' in animal)
                for day in object.dat[animal].keys() if (day in object.dict_animals_days[animal]))))
            control = np.array(list(chain.from_iterable(
                [100 * object.dat[animal][day][sub_file]['thresholded_cells'].sum() / len(
                    object.dat[animal][day][sub_file]['thresholded_cells'])
                 for sub_file in object.dat[animal][day].keys() if ((screen in sub_file) and ('chirps' not in sub_file))]
                for animal in object.dat.keys() if ('GCaMP6s' in animal)
                for day in object.dat[animal].keys() if (day in object.dict_animals_days[animal]))))

            ax[i].scatter(np.array([0] * len(control)), control, s=65, color='grey', alpha=0.6, label='control')
            ax[i].scatter(np.array([1] * len(gnat)), gnat, s=65, color='red', alpha=0.6, label='RD1')
            ax[i].bar(np.array([0]), control.mean(), color='grey', alpha=0.3, label='control')
            ax[i].bar(np.array([1]), gnat.mean(), color='red', alpha=0.3, label='RD1')

            # screen_label = 'regular monitor' if 'big' in screen else 'ultrabright monitor'
            # ax[i].set_title(
            #     f'{"FFF" if "chirps" in stim else "SFxO"} ({"regular monitor" if "big" in screen else "ultrabright monitor"})')
            ax[i].set_title(
                f'{"regular monitor" if "big" in screen else "ultrabright monitor"}')

            # if i == 0:
            #     ax[i_stim, i].set_ylabel('Proportion of responsive cells (%)')
            ax[i].set_xticks(np.array([0, 1]))
            ax[i].set_xticklabels(['Control', 'GNAT'])

        fig.text(0.04, 0.5, 'Proportion of responsive cells (%)', va='center', rotation='vertical', fontsize=12)

        # plt.tight_layout()
        # plt.legend()
        plt.savefig(os.path.join(object.save_path, f'% responsive cells ({project})'))
        plt.show()


def hist_osi_angle (obj, animal, day, session):
    '''
    plots histogram distribution of OSIs/preferred orientation for each day
    :param obj:
    :return:
    '''

    if not os.path.exists(os.path.join(obj.save_path, 'histogram', animal)):
        os.makedirs(os.path.join(obj.save_path, 'histogram', animal))

    with PdfPages(os.path.join(obj.save_path, 'histogram', animal,f'{animal} OSI_PD histogram ({day}, {session}).pdf')) as pdf:
        fig, ax = plt.subplots(ncols = 2, figsize = (6,5))

        ax[0].hist(obj.dat[animal][day][session]['pref_orientation'], color =  'r', alpha = 0.5, label = f'Cell count, {day}', bins = np.linspace(-180,180,25))
        ax[1].hist(obj.dat[animal][day][session]['OSI'], color =  'b', alpha = 0.5, label = f'Cell count, {day}', bins = np.linspace(0,1,25))

        # ax[i,0].set_ylim([0, 6])
        # ax[i, 0].set_xlim([-100, 100])
        # ax[i, 1].set_ylim([0, 18])
        ax[0].set_xlabel('Orientation (deg)')
        ax[0].set_ylabel('Cell count')
        ax[0].set_title("Preferred Orientation")
        ax[1].set_title("Orientation Selectivity Index")
        ax[1].set_xlabel('OSI')
        ax[1].set_ylabel('Cell count')

        #font_kwargs = dict(fontsize="large")
        #add_headers(fig, col_headers=["Preferred Orientation", "Orientation Selectivity Index"], row_headers=['Cell count \n (' + day + ')' for day in list(obj.dat_subject.keys())], **font_kwargs)
        plt.suptitle(f'{animal}, {day}, {session}')

        plt.tight_layout()
        pdf.savefig()
        if not obj.show_plots:
            plt.close()


def sf_tuning_curves(obj, animal, day, session, amplitude_curve_SFs, sorted_sfs_arr):

    if not os.path.exists(os.path.join(obj.save_path, 'sf_tuning_curve', animal)):
        os.makedirs(os.path.join(obj.save_path, 'sf_tuning_curve', animal))

    n_cells = amplitude_curve_SFs.shape[-1]
    nrows = 6
    ncols = 8
    n_pages = int(np.ceil(n_cells / (nrows*ncols)))

    with PdfPages(os.path.join(obj.save_path, 'sf_tuning_curve', animal, f'{animal} sf tuning ({day}, {session}).pdf')) as pdf:

        for n_page in range(n_pages):
            cells = np.arange((n_page + 0) * nrows * ncols, (n_page + 1) * nrows * ncols)

            fig, ax = plt.subplots(nrows, ncols, figsize=(23, 12), sharex=True, sharey=False)
            ax = ax.ravel()

            for i, i_cell in enumerate(cells):
                if i_cell < n_cells:

                    ax[i].plot(amplitude_curve_SFs[:, :, i_cell].mean(axis = 0), c='black', alpha=1, linewidth = 3)
                    [ax[i].plot(amplitude_curve_SFs[i_repeat, :, i_cell], c = 'blue', alpha = 0.4) for i_repeat in range(amplitude_curve_SFs.shape[0])]
                    #print(np.arange(amplitude_curve_SFs[:, :, i_cell].shape[0]).shape)
                    ax[i].set_xticks(np.arange(sorted_sfs_arr.shape[1]))
                    #print(np.squeeze(sorted_sfs_arr.mean(axis=0)).shape)
                    ax[i].set_xticklabels(np.squeeze(sorted_sfs_arr.mean(axis=0)))
                    ax[i].set_title(f'ROI {i_cell}')
                    ax[i].set_xlabel('SF')
                    ax[i].set_ylabel('Amplitude')

                if i_cell >= n_cells:
                    ax[i].axis('off')

            plt.tight_layout()
            pdf.savefig()
            if not obj.show_plots:
                plt.close()

def sf_tf_tuning_curves(obj, animal, day, session, sorted_sfs_arr, sorted_tfs_arr):

    sf_tf_tuning = obj.dat[animal][day][session]['SF_TF_tuning']

    if not os.path.exists(os.path.join(obj.save_path, 'sf_tf_tuning_curve', animal)):
        os.makedirs(os.path.join(obj.save_path, 'sf_tf_tuning_curve', animal))

    n_cells = sf_tf_tuning.shape[-1]
    nrows = 6
    ncols = 9
    n_pages = int(np.ceil(n_cells / (nrows*ncols)))

    with PdfPages(os.path.join(obj.save_path, 'sf_tf_tuning_curve', animal, f'{animal} sf tf tuning ({day}, {session}).pdf')) as pdf:

        for n_page in range(n_pages):
            cells = np.arange((n_page + 0) * nrows * ncols, (n_page + 1) * nrows * ncols)

            fig, ax = plt.subplots(nrows, ncols, figsize=(23, 12), sharex=True, sharey=True)
            ax = ax.ravel()

            for i, i_cell in enumerate(cells):
                if i_cell < n_cells:

                    ax[i].imshow(sf_tf_tuning[:, :, i_cell].T, cmap = 'plasma', aspect='auto')

                    ax[i].set_xticks(np.arange(len(sorted_sfs_arr)))
                    ax[i].set_xticklabels(np.squeeze(sorted_sfs_arr))
                    ax[i].set_yticks(np.arange(len(sorted_tfs_arr)))
                    ax[i].set_yticklabels(np.squeeze(sorted_tfs_arr))
                    ax[i].set_title(f'ROI {i_cell}')
                    ax[i].set_xlabel('SF')
                    ax[i].set_ylabel('TF')

                if i_cell >= n_cells:
                    ax[i].axis('off')

            plt.tight_layout()
            pdf.savefig()
            if not obj.show_plots:
                plt.close()

#parameter_matrix_plot(data_object, 'EC_GCaMP6s_09', '20241122', 'small_SFxO_000_012')


