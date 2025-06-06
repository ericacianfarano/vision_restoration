import matplotlib.pyplot as plt

from classes import *
from full_field_plots import *

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
                'EC_GNAT_03': ['20240924'],
                'EC_GNAT_05': ['20240924'],
                'EC_GNAT_06': ['20240924'],
                'EC_GNAT_04': ['20241004'],
                }


#animals_days = {'EC_GCaMP6s_06': ['20241121']}
data_object = DataAnalysis (['E:\\vision_restored', 'I:\\vision_restored'], dict_animals_days = animals_days, response_type = 'fluorescence', dlc = False, show_plots = False)

del data_object.dat['EC_MWopto_03']
del data_object.dat['EC_MWopto_04']['20241129'] #l23 recordings
del data_object.dat['EC_MWopto_04']['20241113']
del data_object.dat['EC_MWopto_05']['20241127']['big100_chirps_000_008']
del data_object.dat['EC_MWopto_08']['20250306']['big100_chirps_000_003']
del data_object.dat['EC_MWopto_10']['20250306']['big100_chirps_000_007']
del data_object.dat['EC_MWopto_10']['20250311']
del data_object.dat['EC_GCaMP6s_09']['20241122']['big100_chirps_000_008']


animals_days = {'EC_GCaMP6s_11': ['20241113'],
                'EC_GCaMP6s_12': ['20250123'],
                'EC_GCaMP6s_13': ['20250123'],
                'EC_GNAT_03': ['20240924'],
                'EC_GNAT_05': ['20240924'],
                'EC_GNAT_06': ['20240924'],
                'EC_GNAT_04': ['20241004'],
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



###############
#clustering FFF
groups = np.unique([animal[3:-3] for animal in data_object.dat.keys()])
screen = 'small'
recording = 'chirp'
min_cells = np.array(list(chain.from_iterable(
    [np.array(data_object.dat[animal][day][sub_file]['zscored_matrix_baseline'].mean(axis = 0).shape)
     for sub_file in data_object.dat[animal][day].keys() if ((recording in sub_file) and (screen in sub_file))]
    for animal in data_object.dat.keys()
    for day in data_object.dat[animal].keys())))[:,0].min()
min_timepoints = np.array(list(chain.from_iterable(
    [np.array(data_object.dat[animal][day][sub_file]['zscored_matrix_baseline'].mean(axis = 0).shape)
     for sub_file in data_object.dat[animal][day].keys() if ((recording in sub_file) and (screen in sub_file))]
    for animal in data_object.dat.keys()
    for day in data_object.dat[animal].keys())))[:,1].min()
control = np.array(list(chain.from_iterable(
    [data_object.dat[animal][day][sub_file]['zscored_matrix_baseline'].mean(axis = 0)[:min_cells, :min_timepoints].reshape(-1)
     for sub_file in data_object.dat[animal][day].keys() if ((recording in sub_file) and (screen in sub_file))]
    for animal in data_object.dat.keys() if 'GCaMP6s' in animal
    for day in data_object.dat[animal].keys())))
restored = np.array(list(chain.from_iterable(
    [data_object.dat[animal][day][sub_file]['zscored_matrix_baseline'].mean(axis = 0)[:min_cells, :min_timepoints].reshape(-1)
     for sub_file in data_object.dat[animal][day].keys() if ((recording in sub_file) and (screen in sub_file))]
    for animal in data_object.dat.keys() if 'MWopto' in animal
    for day in data_object.dat[animal].keys())))
rd1 = np.array(list(chain.from_iterable(
    [data_object.dat[animal][day][sub_file]['zscored_matrix_baseline'].mean(axis = 0)[:min_cells, :min_timepoints].reshape(-1)
     for sub_file in data_object.dat[animal][day].keys() if ((recording in sub_file) and (screen in sub_file))]
    for animal in data_object.dat.keys() if 'RD1' in animal
    for day in data_object.dat[animal].keys())))

############ PCA of tunign curves
recording = 'SFxO'
screen = 'big'
min_cells = np.array(list(chain.from_iterable(
    [np.array(data_object.dat[animal][day][sub_file]['tuning_curves_sf_pref'].mean(axis = 0).shape)
     for sub_file in data_object.dat[animal][day].keys() if (('l4' not in sub_file )and (recording in sub_file) and (screen in sub_file))]
    for animal in data_object.dat.keys()
    for day in data_object.dat[animal].keys())))[:,1].min()

def pop_dat (object, group, recording, screen):
    g = np.array(list(chain.from_iterable(
        [object.dat[animal][day][sub_file]['tuning_curves_sf_pref'].mean(axis=0)[:, :min_cells].T
         for sub_file in object.dat[animal][day].keys() if ((recording in sub_file) and (screen in sub_file))]
        for animal in object.dat.keys() if group in animal
        for day in object.dat[animal].keys())))
    return g

def pop_dat (object, group, recording, screen, thresholded = True, label = 'OSI'):
    '''
    for a single animal
    :param object:
    :param group:
    :param recording:
    :param screen:
    :param thresholded:
    :param label: 'OSI' or 'pref_ori'
    :return:
    '''
    if thresholded:
        g = np.vstack((list(chain.from_iterable(
            [(object.dat[animal][day][sub_file]['ori_tuning_curves_sf_pref'].mean(axis=0)/object.dat[animal][day][sub_file]['tuning_curves_sf_pref'].mean(axis = 0).max(axis = 0)).T [(object.dat[animal][day][sub_file]['thresholded_cells']==1) & (object.dat[animal][day][sub_file]['OSI']>0.)]
             for sub_file in object.dat[animal][day].keys() if ((recording in sub_file) and (screen in sub_file))]
            for animal in object.dat.keys() if group in animal
            for day in object.dat[animal].keys()))))
        if label == 'pref_ori':
            labels = np.round(np.hstack((list(chain.from_iterable(
                [object.dat[animal][day][sub_file]['preferred_orientation'][(object.dat[animal][day][sub_file]['thresholded_cells']==1) & (object.dat[animal][day][sub_file]['OSI']>0.)]#[(object.dat[animal][day][sub_file]['thresholded_cells']==1) & (object.dat[animal][day][sub_file]['OSI']>0.7)]
                 for sub_file in object.dat[animal][day].keys() if ((recording in sub_file) and (screen in sub_file))]
                for animal in object.dat.keys() if group in animal
                for day in object.dat[animal].keys()))))/ 10) * 10
        elif label == 'OSI':
            labels = np.round(np.hstack((list(chain.from_iterable(
                [object.dat[animal][day][sub_file]['OSI'][(object.dat[animal][day][sub_file]['thresholded_cells']==1) & (object.dat[animal][day][sub_file]['OSI']>0.)]
                 for sub_file in object.dat[animal][day].keys() if ((recording in sub_file) and (screen in sub_file))]
                for animal in object.dat.keys() if group in animal
                for day in object.dat[animal].keys())))),2)
    else:
        g = np.vstack((list(chain.from_iterable(
            [(object.dat[animal][day][sub_file]['ori_tuning_curves_sf_pref'].mean(axis=0)/object.dat[animal][day][sub_file]['tuning_curves_sf_pref'].mean(axis = 0).max(axis = 0)).T[object.dat[animal][day][sub_file]['OSI']>0.3]
             for sub_file in object.dat[animal][day].keys() if ((recording in sub_file) and (screen in sub_file))]
            for animal in object.dat.keys() if group in animal
            for day in object.dat[animal].keys()))))
        if label == 'pref_ori':
            labels = np.round(np.hstack((list(chain.from_iterable(
                [object.dat[animal][day][sub_file]['preferred_orientation'][object.dat[animal][day][sub_file]['OSI']>0.3] #[(object.dat[animal][day][sub_file]['thresholded_cells']==1) & (object.dat[animal][day][sub_file]['OSI']>0.7)]
                 for sub_file in object.dat[animal][day].keys() if ((recording in sub_file) and (screen in sub_file))]
                for animal in object.dat.keys() if group in animal
                for day in object.dat[animal].keys()))))/ 10) * 10
        elif label == 'OSI':
            labels = np.round(np.hstack((list(chain.from_iterable(
                [object.dat[animal][day][sub_file]['OSI'][object.dat[animal][day][sub_file]['OSI']>0.3]
                 for sub_file in object.dat[animal][day].keys() if ((recording in sub_file) and (screen in sub_file))]
                for animal in object.dat.keys() if group in animal
                for day in object.dat[animal].keys())))),2)
    return g,labels


def pop_dat_fff (object, group, recording, screen, thresholded = True):
    '''
    for a single animal
    :param object:
    :param group:
    :param recording:
    :param screen:
    :param thresholded:
    :param label: 'OSI' or 'pref_ori'
    :return:
    '''
    if thresholded:
        g = np.vstack((list(chain.from_iterable(
            [(object.dat[animal][day][sub_file]['zscored_matrix_baseline'][...,4*data_object.fps:16*data_object.fps].mean(axis=0)/object.dat[animal][day][sub_file]['zscored_matrix_baseline'][...,4*data_object.fps:16*data_object.fps].mean(axis = 0).max(axis = 1)[:,None]) [object.dat[animal][day][sub_file]['thresholded_cells']==1]
             for sub_file in object.dat[animal][day].keys() if ((recording in sub_file) and (screen in sub_file))]
            for animal in object.dat.keys() if group in animal
            for day in object.dat[animal].keys()))))
    else:
        g = np.vstack((list(chain.from_iterable(
            [(object.dat[animal][day][sub_file]['zscored_matrix_baseline'].mean(axis=0)/object.dat[animal][day][sub_file]['zscored_matrix_baseline'].mean(axis = 0).max(axis = 1)[:,None])
             for sub_file in object.dat[animal][day].keys() if ((recording in sub_file) and (screen in sub_file))]
            for animal in object.dat.keys() if group in animal
            for day in object.dat[animal].keys()))))
    return g


def plot_embedding (array, labels, method='PCA', components = [0,1]):
    if method == 'PCA':
        pca = PCA(n_components=4)
        embedding = pca.fit_transform(array)
    elif method == 'UMAP':
        reducer = umap.UMAP(n_components=4, random_state=42)
        embedding = reducer.fit_transform(array)
    elif method == 'ISOMAP':
        isomap = Isomap(n_components=4, n_neighbors=5)
        embedding = isomap.fit_transform(array)

    # Sort the unique numeric conditions so Seaborn will color them in ascending order
    unique_conditions_sorted = np.sort(np.unique(labels))

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=embedding[:, components[0]],
        y=embedding[:, components[1]],
        hue=labels,
        palette="turbo",
        hue_order=unique_conditions_sorted,  # <---- ensures numeric ordering
        s=100, alpha=0.8
    )
    plt.title("Embedding")
    plt.legend(title="Condition")
    plt.show()

# control,conditions_num = pop_dat (data_object, 'GCaMP6s', 'SFxO', 'big100', thresholded = True)
# rd1,conditions_num = pop_dat (data_object, 'RD1', 'SFxO', 'big100', thresholded = False)
# restored,conditions_num = pop_dat (data_object, 'MWopto', 'SFxO', 'big100', thresholded = True)
# rd1,conditions_num = pop_dat (data_object, 'RD1', 'SFxO', 'big100', thresholded = False)
# plot_embedding(rd1, conditions_num, method = 'ISOMAP', components = [0,1])

control,conditions_num = pop_dat (data_object, 'GCaMP6s', 'SFxO', 'big100', thresholded = True)
plot_embedding(control, conditions_num, method = 'ISOMAP', components = [0,1])
plot_embedding(control, conditions_num, method = 'PCA', components = [0,1])

# orientation data
control,conditions_num = pop_dat (data_object, 'GCaMP6s', 'SFxO', 'big100', thresholded = True)
rd1,conditions_num = pop_dat (data_object, 'RD1', 'SFxO', 'big100', thresholded = True)
restored,conditions_num = pop_dat (data_object, 'MWopto', 'SFxO', 'big100', thresholded = True)
data_flattened = np.vstack((control, restored, rd1)) # shape n_animals, n_cells x n_timepoints
groups = np.unique([animal.split('_')[1] for animal in data_object.dat.keys()])
conditions = [groups[0]]*control.shape[0] + [groups[1]]*restored.shape[0] + [groups[2]]*rd1.shape[0]
plot_embedding(data_flattened, conditions, method = 'PCA', components = [0,1])

## fff data
control = pop_dat_fff (data_object, 'GCaMP6s', 'chirp', 'big100', thresholded = True)
rd1 = pop_dat_fff (data_object, 'RD1', 'chirp', 'big100', thresholded = True)
restored = pop_dat_fff (data_object, 'MWopto', 'chirp', 'big100', thresholded = True)
data_flattened = np.vstack((control, restored, rd1)) # shape n_animals, n_cells x n_timepoints
groups = np.unique([animal.split('_')[1] for animal in data_object.dat.keys()])
conditions = [groups[0]]*control.shape[0] + [groups[1]]*restored.shape[0] + [groups[2]]*rd1.shape[0]


# Apply UMAP to reduce to 2D
reducer = umap.UMAP(n_components=2, random_state=42)
embedding = reducer.fit_transform(control)
# pca = PCA(n_components=5)
# embedding = pca.fit_transform(data_flattened)
# control = data_flattened

# isomap = Isomap(n_components=6, n_neighbors=5)
# embedding = isomap.fit_transform(control)

# plt.figure(figsize=(8, 6))
# sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=conditions, palette="turbo", s=100, alpha=0.8)
# plt.title("Embedding")
# plt.legend(title="Condition")
# plt.show()




#orientations
recording = 'SFxO'
min_cells = np.array(list(chain.from_iterable(
    [np.array(data_object.dat[animal][day][sub_file]['zscored_matrix_baseline'].mean(axis = 0).shape)
     for sub_file in data_object.dat[animal][day].keys() if ((recording in sub_file) and (screen in sub_file))]
    for animal in data_object.dat.keys()
    for day in data_object.dat[animal].keys())))[:,-2].min()
min_timepoints = np.array(list(chain.from_iterable(
    [np.array(data_object.dat[animal][day][sub_file]['zscored_matrix_baseline'].mean(axis = 0).shape)
     for sub_file in data_object.dat[animal][day].keys() if ((recording in sub_file) and (screen in sub_file))]
    for animal in data_object.dat.keys()
    for day in data_object.dat[animal].keys())))[:,-1].min()
control = np.array(list(chain.from_iterable(
    [data_object.dat[animal][day][sub_file]['zscored_matrix_baseline'].mean(axis = 0)[:,:,:min_cells, :min_timepoints].reshape(-1)
     for sub_file in data_object.dat[animal][day].keys() if ((recording in sub_file) and (screen in sub_file))]
    for animal in data_object.dat.keys() if 'GCaMP6s' in animal
    for day in data_object.dat[animal].keys())))
restored = np.array(list(chain.from_iterable(
    [data_object.dat[animal][day][sub_file]['zscored_matrix_baseline'].mean(axis = 0)[:,:,:min_cells, :min_timepoints].reshape(-1)
     for sub_file in data_object.dat[animal][day].keys() if ((recording in sub_file) and (screen in sub_file))]
    for animal in data_object.dat.keys() if 'MWopto' in animal
    for day in data_object.dat[animal].keys())))
rd1 = np.array(list(chain.from_iterable(
    [data_object.dat[animal][day][sub_file]['zscored_matrix_baseline'].mean(axis = 0)[:,:,:min_cells, :min_timepoints].reshape(-1)
     for sub_file in data_object.dat[animal][day].keys() if ((recording in sub_file) and (screen in sub_file))]
    for animal in data_object.dat.keys() if 'RD1' in animal
    for day in data_object.dat[animal].keys())))

data_flattened = np.vstack((control, restored, rd1)) # shape n_animals, n_cells x n_timepoints
conditions = [groups[0]]*control.shape[0] + [groups[1]]*restored.shape[0] + [groups[2]]*rd1.shape[0]
# Apply UMAP to reduce to 2D
# reducer = umap.UMAP(n_components=2, random_state=42)
# embedding = reducer.fit_transform(data_flattened)
pca = PCA(n_components=10)
embedding = pca.fit_transform(data_flattened)
isomap = Isomap(n_components=6, n_neighbors=5)
embedding = isomap.fit_transform(data_flattened)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=embedding[:, 6], y=embedding[:, 7], hue=conditions, palette="plasma", s=100, alpha=0.8)
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.title("Embedding of Neural Data")
plt.legend(title="Condition")
plt.show()
########################
# Step 2: K-Means Clustering
# n_clusters = 6  # Choose number of clusters
# kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
# cluster_labels = kmeans.fit_predict(neural_embedding)


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


animals_days = {
                'EC_GCaMP6s_05': ['20240925'],
                'EC_GCaMP6s_06': ['20240925', '20241113', '20241121'],
                'EC_GCaMP6s_08': ['20240927'], #0927
                'EC_GCaMP6s_09': ['20241113', '20241122'],
                'EC_RD1_06': ['20250314'],
                'EC_RD1_08': ['20250314'],
                'EC_RD1asleep_06': ['20250315'],
                'EC_RD1asleep_08': ['20250315']
                }


#animals_days = {'EC_GCaMP6s_06': ['20241121']}
data_object = DataAnalysis (['H:\\vision_restored', 'G:\\vision_restored'], dict_animals_days = animals_days, response_type = 'fluorescence', dlc = False, show_plots = False)

# plotting tunign curves
tc = data_object.dat['EC_GCaMP6s_09']['20241122']['big100_SFxO_000_010']['tuning_curves_sf_pref'].mean(axis = 0)
colors_cells = plt.cm.plasma(np.linspace(0, 0.8, 6))
plt.figure(figsize = (5,6))
for i, cell in enumerate([8,3,1,10,22,9]):
    plt.plot(tc[:,cell]/tc[:,cell].max() + i*1.2 , c = colors_cells[i], linewidth = 2)

plt.xticks(np.arange(0,8,2), labels = [int(a) for a in np.unique(data_object.dat['EC_GCaMP6s_09']['20241122']['big100_SFxO_000_010']['thetas'])[::2]])
#plt.xticklabels(np.unique(data_object.dat['EC_GCaMP6s_09']['20241122']['big100_SFxO_000_010']['thetas'])[::2])
plt.show()


# full field flash response
plot_on_off_index(data_object, thresholded_cells = True)
avg_onoff_response(data_object)
avg_onoff_response_groups(data_object)
plot_response_speed(data_object, thresholded_cells = True)
plot_response_amplitude(data_object, thresholded_cells = True)
responsive_cells_hist (data_object)
responsive_cells_hist_gnat (data_object)

# olfaction stuff
animals_days = {'EC_RD1_06': ['20250223'], 'EC_RD1_08': ['20250223']}           #simple  8 orientation x 6 repeats

data_object = OlfactionAnalysis ('I:\\olfaction', dict_animals_days = animals_days, zscore_threshold=3, response_type = 'fluorescence', dlc = False, show_plots = False)

arr = data_object.dat['EC_RD1_06']['20250223']['olf_v1_000_000']['responses']
ttl = data_object.dat['EC_RD1_06']['20250223']['olf_v1_000_000']['ttl_data']

ttl_responses = np.array([arr [:, t:t+data_object.fps*4] for t in ttl])



def avg_onoff_response_asleep(object):

    groups = ['GCaMP6s', 'RD1_', 'RD1asleep']
    colours = ['black', 'blue', 'red']
    screen = 'big'

    min_timepoints = np.array(
        list(chain.from_iterable([object.dat[animal][day][sub_file]['zscored_matrix_baseline'].shape[-1]
                                  for sub_file in object.dat[animal][day].keys() if ('chirp' in sub_file)]
                                 for animal in object.dat.keys()
                                 for day in object.dat[animal].keys()))).min()

    fig, ax = plt.subplots(1, len(groups), figsize=(12, 8), sharex = True, sharey=True)

    for i_group, group_name in enumerate(groups):
        group_dat = np.array(list(chain.from_iterable(
            [object.dat[animal][day][sub_file]['zscored_matrix_baseline'][1:,
             object.dat[animal][day][sub_file]['thresholded_cells'] == 1, :min_timepoints].mean(axis=(0, 1))
             for sub_file in object.dat[animal][day].keys() if (('chirp' in sub_file) and (screen in sub_file))]
            for animal in object.dat.keys() if (group_name in animal)
            for day in object.dat[animal].keys())))


        ax[i_group].plot(np.arange(min_timepoints),group_dat.T, alpha = 0.7, color=colours[i_group])
        #ax[i_group].plot(np.arange(min_timepoints), np.nanmean(group_dat, axis=0), linewidth = 2, color='black')
        screen_label = 'Ultrabright' if screen == 'small' else 'Regular'
        ax[i_group].set_title (f'{group_name}, {screen_label} monitor', fontsize = 10)

        ax[i_group].axvline(object.fps * 0, c='grey', alpha=0.5)
        ax[i_group].axvline(object.fps * 4, c='black', alpha=0.5)
        ax[i_group].axvline(object.fps * 8, c='black', alpha=1, linestyle='--')
        ax[i_group].axvline(object.fps * 12, c='black', alpha=0.5)
        ax[i_group].axvline(object.fps * 16, c='grey', alpha=0.5)
        ax[i_group].set_xlabel('Time since stimulus onset (s)')
        ax[i_group].set_ylabel('Z-scored response')
        ax[i_group].set_xticks(np.arange(min_timepoints)[::4 * object.fps],
                         (np.arange(min_timepoints) // object.fps)[::4 * object.fps])

    plt.suptitle('Full-Field Flash response')
    plt.tight_layout()
    #save_fig(object, 'Full-Field Flash', 'average on-off responses groups')
    plt.show()



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
