#Load your raster of neuronal activity.
# Get the coactivity.
# Get the neuronal vectors.
# Compute similarity and hierarchical clustering.
# Cluster vectors, get the neuronal ensembles.
# Get the functional connection neuronal ensembles.
# Plot measures.

def adjacency_from_raster(raster, connectivity_method='coactivity'):
    """
    Get adjacency matrix from raster peaks > to determine functional connectivity

    Parameters:
    - raster: 2D numpy array of shape (n_cells, n_timepoints)
    - connectivity_method: 'coactivity', 'jaccard', or 'pearson' (correlation-based method)

    Returns:
    - adjacency: 2D numpy array representing the adjacency matrix
    """

    if connectivity_method == 'coactivity':
        # Binarize data > 1 where activity is above the 95th percentile
        threshold = np.percentile(raster, 95)  # Example threshold
        binary_raster = (raster > threshold).astype(int)  # shape n_neurons, n_timepoints
        # dot product > computes pairwise co-activation counts betwen cells across timepoints
        adjacency = np.dot(binary_raster, binary_raster.T)           # Coactivity (count coactivation)
        np.fill_diagonal(adjacency, 0)                               # Remove diagonal (self-loops)
    elif connectivity_method == 'jaccard':
        # Binarize data > 1 where activity is above the 95th percentile
        threshold = np.percentile(raster, 95)  # Example threshold
        binary_raster = (raster > threshold).astype(int)  # shape n_neurons, n_timepoints
        dist_matrix = squareform(pdist(binary_raster, metric='jaccard'))     # Jaccard distance to similarity (1 - Jaccard distance)
        adjacency = 1 - dist_matrix                                   # Convert to similarity
        adjacency[np.isnan(dist_matrix)] = 0                          # Handle NaNs

    elif connectivity_method == 'pearson':   # pearson correlation > correlation-based method (calculating pairwise correlations)
        adjacency = np.corrcoef(raster)
        adjacency[adjacency < 0] = 0                                # Set negative correlations to 0
        adjacency[np.isnan(adjacency)] = 0                          # Handle NaNs
        np.fill_diagonal(adjacency, 0)                              # Remove diagonal (self-loops)

    return adjacency

def threshold_in_cumulative_distribution(data, alpha):
    """
    Calculate the threshold value in the cumulative distribution of a dataset (data) at a specified significance level, alpha
    - Determining the point where a certain percentage (defined by alpha) of the data is less than or equal to that threshold.

    Parameters:
    - data: 1D numpy array of data
    - alpha: significance threshold (e.g., 0.05 for the 5% threshold)

    Returns:
    - threshold: the threshold value
    """
    threshold = np.percentile(data, 100 * (1 - alpha))
    return threshold


'''def threshold_in_cumulative_distribution(data, alpha):
    """
    Find the threshold for a given alpha in the cumulative distribution of data.

    Parameters:
    - data: 1D numpy array or list of data points
    - alpha: significance threshold (e.g., 0.05 for the 5% threshold)

    Returns:
    - th: the threshold value
    """
    if np.min(data) == np.max(data):
        th = np.max(data) + 1
    else:
        # Create bins based on the maximum value of data
        if np.max(data) <= 1:
            bins = np.arange(0, 1.001, 0.001)
        elif np.max(data) <= 10:
            bins = np.arange(0, 10.1, 0.01)
        else:
            bins = np.arange(0, np.max(data) + 1)

        # Compute the histogram
        hist, edges = np.histogram(data, bins=bins, density=True)

        # Compute the cumulative distribution
        cdy = np.cumsum(hist) * (edges[1] - edges[0])  # Multiply by bin width for correct scaling

        # Normalize the cumulative distribution
        cdy /= np.max(cdy)

        # Find the threshold corresponding to alpha
        id = np.argmax(cdy > (1 - alpha))  # Find first index where cumulative distribution exceeds (1-alpha)
        th = edges[id]

    return th'''

def significant_network_from_raster(raster, shuffle_iterations =1000, alpha=0.05, network_method='coactivity', single_th=True):
    """
    Get significant network from raster by comparing original adjacency with shuffled data.

    Parameters:
    - raster: 2D numpy array (cells x timepoints)
    - iterations: number of randomizations
    - alpha: significance threshold
    - network_method: method for calculating network adjacency ('coactivity', 'jaccard', or 'correlation')
    - shuffle_method: how to shuffle data ('time_shift', 'random')
    - single_th: whether to apply single threshold per edge

    Returns:
    - A_significant: binary adjacency matrix with significant connections
    - A_raw: raw adjacency matrix
    - th: threshold used for significance
    - As: shuffled adjacency matrices
    """

    # Get original adjacency
    adjacency_raw = adjacency_from_raster(raster, connectivity_method=network_method)

    # Initialize shuffled adjacency matrices
    n_neurons = adjacency_raw.shape[0]

    '''
    A network with n_neurons has n_neurons ** 2 potential connections (edges) if each neuron were connected to every other neuron, including itself.
    But we ignore the diagonal elements, which are self-connections (neuron connected to itself) > we are interested in the connections between different neurons
    We divide by 2 because each pair of neurons forms an edge (pair (i,j) is equivalent to pair (j,i)) > don't want to count same edge twice
    Total # unique non-diagonal connections/edges between n_neurons different neurons is calculated by : (n_neurons ** 2 - n_neurons) // 2
    '''
    n_edges = (n_neurons ** 2 - n_neurons) // 2

    # generate n_shuffles shuffled raster matrices > to calculate chance co-activity
    adjacency_shuffled = np.zeros((shuffle_iterations, n_edges))

    for i in tqdm(range(shuffle_iterations), desc = f'Calculating shuffled adjacency matrices'):
        # shifts the time trace of each cell by a random amount each cell maintains its pattern of activity
        shuffled = circular_shift(raster)  # shape n_neurons, n_timepoints

        '''
        Squareform converts a square, symmetric adjacency matrix into a condensed (vector) form
        'tovector' specifies to take the upper triangle of the square matrix (excluding the diagonal)
        So the shape of the upper triangle of the matrix (excluding the diagonal) = total # unique non-diagonal edges betwen different neurons
        '''
        adjacency_shuffled[i, :] = squareform(adjacency_from_raster(shuffled, connectivity_method=network_method), 'tovector')

    # Calculate threshold based on shuffled adjacency matrices
    if single_th:
        th = np.zeros(n_edges)
        for i in range(n_edges):
            # calculating the threshold at one specific node across all iterations
            th[i] = threshold_in_cumulative_distribution(adjacency_shuffled[:, i], alpha)
        th = squareform(th)     # reshape neuron into square matrix
    else:
        th = threshold_in_cumulative_distribution(adjacency_shuffled.flatten(), alpha)

    # Significant adjacency matrix >  binary matrix indicating significant connections (exceeding the threshold)
    #  If the actual number of coactivations Co_ab is above threshold T_ab, we put a functional connection between neuron a and b.
    #  Doing this with every pair of neurons, we got a functional neuronal network,
    #  where every node is a neuron and every link represents a significant coactivity between them
    adjacency_significant = adjacency_raw > th
    return adjacency_significant, adjacency_raw, th, adjacency_shuffled

def remove_noisy_spikes(raster, network):
    """
    Reduce the noisy spikes from a given raster based on the connectivity between neurons.
    Remove noisy spikes based on functional connections

    Parameters:
        raster (np.ndarray): binarized matrix (neurons x time frames) (1 indicates a spike)
        network (np.ndarray): Connectivity matrix (neurons x neurons), where 1 indicates a connection.

    Returns:
        np.ndarray: Cleaned raster with noisy spikes removed.
    """

    for frame in range(raster.shape[1]):
        # Find active neurons in the current frame
        active = np.where(raster[:, frame])[0]

        if active.size > 0:
            # Identify active neurons with no significant coactivation
            no_sig = np.where(np.sum(network[active[:, None], active], axis=0) == 0)[0]

            if no_sig.size > 0:
                # Remove spikes from neurons with no significant coactivity
                raster[active[no_sig], frame] = 0

    return raster

def get_peak_or_valley_indices(data, threshold, detect_peaks):
    """
    Get peak or valley indices.

    Parameters:
    data (numpy.ndarray): The input data array.
    threshold (float): The threshold value for detecting peaks or valleys.
    detect_peaks (bool): Whether to detect peaks (True) or valleys (False).

    Returns:
    - indices of the peaks/valleys and their count.
    """
    if detect_peaks:
        indices = np.where(data > threshold)[0]
    else:  # detect valleys
        indices = np.where(data < threshold)[0]

    return indices, len(indices)

def find_peaks_or_valleys(data, threshold=0, detect_peaks=True, join = True):
    """
    Find peaks or valleys based on a given threshold and other optional parameters.

    Parameters:
    data (numpy.ndarray): Input data vector (1D array). (shape n_frames)
    threshold (float): Threshold value for detecting peaks or valleys.
    join (bool): Whether to join adjacent peaks/valleys. (0 = each vector above threshold is a peak; 1 = joining of adjacent vectors above the threshold is a peak)
    detect_peaks (bool): Whether to detect peaks (True) or valleys (False).

    Returns:
    tuple: (indices, widths, amplitudes, ini_fin_times)
    - indices = Fx1 vector containing the peak indices

    # 1. Get indices above threshold
    # 2. Set the number at each peak (or valley)
    """
    original_data = np.copy(data)

    # Ensure data is 1D
    data = data.flatten()

    # 1. Get peak or valley indices above threshold
    idx, count = get_peak_or_valley_indices(data, threshold, detect_peaks)
    F = len(data)
    indices = np.zeros(F, dtype=int)

    if count == 0:
        print("No peaks found!" if detect_peaks else "No valleys found!")
        return indices, np.array([]), np.array([]), np.empty((0, 2), dtype=int)

    # 2. Assign numbers to peaks
    indices[idx] = np.arange(1, count + 1)

    if join:
        # Find indices where peak numbering is not consecutive
        is_ = np.where(idx[:-1] + 1 != idx[1:])[0] + 1  # Equivalent to `find(idx~=[0; idx(1:numel(idx)-1)+1])`
        is_ = np.insert(is_, 0, 0)  # Include first peak
        count = len(is_)

        if count:
            for j in range(count - 1):
                indices[idx[is_[j]]: idx[is_[j + 1]]] = j + 1  # Set peak number
            indices[idx[is_[-1]]: max(idx)] = count  # Last peak assignment

        '''
        for i in range(1, len(arr)):
            if arr[i] != 0 and arr[i] == arr[i - 1] + 1:  # Consecutive numbers
                result[i] = result[i - 1]  # Group consecutive numbers together

        '''

    # Get widths, start and end times (initial_final_times) > find + store [the first instance, the last instance] > find time diff
    ini_fin_times = np.array([
        (np.where(indices == i)[0][0], np.where(indices == i)[0][-1])
        for i in range(1, count + 1)])
    widths = np.diff(ini_fin_times, axis=1).flatten() + 1  # Width = (end - start + 1)

    # Get peaks or valley amplitudes
    amplitudes = np.zeros(count)
    for i in range(count):
        if detect_peaks:
            amplitudes[i] = np.max(original_data[indices == i + 1])
        else:
            amplitudes[i] = np.min(original_data[indices == i + 1])

    return indices, widths, amplitudes, ini_fin_times



animal, day, subfile = 'EC_GCaMP6s_09', '20241122', 'big100_SFxTF_000_011'

raster = np.hstack([data_object.dat[animal][day][subfile]['zscored_responses_ttls'][stim].reshape(-1, data_object.dat[
    animal][day][subfile]['zscored_responses_ttls'][stim].shape[0] * data_object.dat[animal][day][subfile]['zscored_responses_ttls'][stim].shape[-1])
                for stim in data_object.dat[animal][day][subfile]['zscored_responses_ttls'].keys() if 'Grating' in stim])

# Get significant network
network, _, _, _ = significant_network_from_raster(raster, shuffle_iterations=100, network_method='coactivity')    # shape n_neurons, n_neurons

# Binarize data > 1 (spike) where activity is above the 95th percentile
threshold = np.percentile(raster, 95)  # Example threshold
raster_bin = (raster > threshold).astype(int)  # shape n_neurons, n_timepoints

# Remove noisy spikes based on functional connections
cleaned_raster_bin = remove_noisy_spikes(raster_bin, network)


# Detection of coactivity peaks to identify ensembles (Pérez-Ortega et al., 2021) > extract ensembles based on significant population coactivity (Pérez-Ortega et al., 2016).
'''
- Obtained a 1 × F vector, where F is the number of frames, by summing the coactive neurons from the raster matrix E.
- Then, we perform 1000 surrogated raster matrices by randomly circular shifting in time the activity of every single neuron and computing the coactivity given by chance.
- We determined a significant coactivity threshold (p<0.05) from shuffled coactivity
- Vectors with significant co-activity were clustered to extract neuronal ensembles.
'''

def circular_shift(data_array):
    '''
    :param data_array: shape n_neurons, n_timepoints
    Circularly shift activity of each neuron across time > shifts the time trace of each cell by a random amount each cell maintains its pattern of activity

    Since data is in the shape n_neurons, n_timepoints > circularly shift along the timepoints axis
    This shifts the temporal sequence for each neuron
    Want to break the temporal dependencies for each neuron (i.e., remove their coactivity patterns) but maintain the individual temporal structure of each neuron.
    '''

    shifted_data = np.zeros_like(data_array)

    for i_cell in range(data_array.shape[0]):

        # choose a random integer in range [-timepoints, + timepoints]
        shift_amount = random.randint(-data_array.shape[-1], data_array.shape[-1])

        # shift each neuron's activity by random number (maintain temporal structure of each neuron but remove co-activity patterns)
        shifted_data[i_cell] = np.roll(data_array[i_cell, :], shift_amount)

    # circularly shift array on timepoints axis by random integer (independently for each neuron)
    return shifted_data

def compute_coactivity(data_array):
    '''
    Calculate co-activity (# neurons active at each frame) for each frame in the raster
    :param data_array: shape n_neurons, n_timepoints
    :return: array: shape n_timepoints
    '''

    # Sum across neurons at each timepoint
    return np.sum(data_array, axis=0)

def jaccard_similarity_matrix(vectors):
    '''
    Calculate Jaccard similarity between population vectors

    - The Jaccard similarity is computed by comparing the sets of active neurons between two timepoints.
    - Jaccard similarity = (A ∩ B) / (A ∪ B), where A and B are the set of active neurons at two different timepoints.

    Relative to ensembles:
    - The Jaccard index measures the proportion of co-active neurons that are shared between two frames, relative to their total activity.
    - Frames with higher similarity would be more likely to belong to the same ensemble.

    :param vectors: array shape n_timepoints, n_neurons
    :return: similarity matrix of shape n_timepoints, n_timepoints
        - off-diagonal elements represent the Jaccard similarity between pairs of timepoints,
        - diagonal represents the similarity of each timepoint with itself (which will always be 1).
    '''

    # Convert vectors to boolean
    vectors_bool = vectors.astype(bool)

    #  Compute pairwise Jaccard distances
    jaccard_dist = pairwise_distances(vectors_bool, metric="jaccard")

    # compute similarity
    return 1- jaccard_dist

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
                binary_data = (data[i] > threshold).astype(int)         # shape n_neurons, n_timepoints

                # 3) Identify significant vectors > test significance of each population vector against a null hypothesis (random activity).

                # calculate co-activity (# neurons active at each frame) for each frame in the raster
                coactivity = compute_coactivity(binary_data)                # shape n_timepoints

                # generate n_shuffles shuffled raster matrices > to calculate chance co-activity
                n_shuffles = 100
                shuffled_coactivity = np.zeros((n_shuffles, binary_data.shape[-1]))     # shape n_shuffles, n_timepoints
                for i_shuffle in range(n_shuffles):

                    # Circularly shuffle the activity across neurons while preserving the temporal structure (by random integer)
                    shuffled = circular_shift(binary_data)  # shape n_neurons, n_timepoints

                    # calculate co-activity given by chance (# neurons active at each frame) for each frame in the raster
                    shuffled_coactivity[i] = compute_coactivity(shuffled)  # shape n_timepoints

                # significance testing
                # For each timepoint (frame) > compare the real coactivity against the distribution of coactivities from shuffles
                # The p-value for each timepoint = how many shuffled coactivities >= real coactivity
                p_values = np.mean(shuffled_coactivity >= coactivity[None, :] , axis=0) # shape n_timepoints

                # a population vector at time t is a binary vector (shape n_neurons) representing the activity of all neurons at that time.
                # dimension = # neurons
                population_vectors = binary_data.T  # n_timepoints x n_neurons

                # retain only population vectors where p <0.05 --> these are significantly different from random activity
                significant_vectors = population_vectors[p_values < 0.05]

                # Jaccard similarity between all single frames (column vectors) of the rebuilt raster matrix (significant population vectors)
                #  represents the co-activity of neurons across time.
                # measures the proportion of co-active neurons that are shared between two frames, relative to their total activity.
                # Frames with higher similarity would be more likely to belong to the same ensemble.
                jaccard_similarity = jaccard_similarity_matrix(significant_vectors)     # shape n_timepoints, n_timepoints

                # Group similar vectors together via hierarchical clustering with single linkage
                # linkage matrix encodes the hierarchical clustering tree > cluster timepoints based on similarity
                linkage_matrix = linkage(1 - jaccard_similarity, method="single")  # Convert similarity to distance

                # Define the clustering threshold (2/3 similarity = 0.67 distance threshold) > keep the branch with more than 2/3 of Jaccard similarity
                # fcluster: form flat clusters from the hierarchical clustering defined by  linkage matrix (cut tree off at t)
                # original observations in each flat cluster have no greater a cophenetic distance than t.
                cluster_labels = fcluster(linkage_matrix, t = 2.2, criterion='distance')

                # Select timepoints that belong to large clusters
                filtered_timepoints = np.where(np.bincount(cluster_labels) > 2)[0]  # Keeping clusters with >1 frame
                filtered_vectors = significant_vectors[filtered_timepoints]  # Subset of significant vectors > we will cluster these

                # Compute a new Jaccard similarity matrix for the filtered timepoints > convert to distance
                filtered_distance_matrix = 1 - jaccard_similarity_matrix(filtered_vectors)

                # Convert the condensed distance matrix to a linkage matrix
                condensed_linkage_matrix = linkage(filtered_distance_matrix, method='single')

                # Apply the threshold (2/3 similarity → 0.33 distance)
                cluster_labels = fcluster(condensed_linkage_matrix, t=1.3, criterion='distance')

                ##################
                # Perform Ward clustering
                clusterer = AgglomerativeClustering(affinity="precomputed", linkage="ward", distance_threshold=0.33, n_clusters = None)

                ensemble_labels = clusterer.fit_predict(filtered_distance_matrix)

                # Group timepoints by their cluster labels
                ensemble_timepoints = defaultdict(list)
                for t, label in enumerate(ensemble_labels):
                    ensemble_timepoints[label].append(filtered_timepoints[t])

                # Extract neurons active in each ensemble
                ensemble_neurons = {}
                for ensemble, timepoints in ensemble_timepoints.items():
                    ensemble_neurons[ensemble] = significant_vectors[timepoints, :].mean(
                        axis=0) > 0  # Binary activation

                # Print results
                print(f"Identified {len(ensemble_neurons)} ensembles")
                for ensemble, active_neurons in ensemble_neurons.items():
                    print(f"Ensemble {ensemble}: {np.where(active_neurons)[0]} active neurons")

                # Filter non-similar & infrequency co-activations
                # Count how many timepoints are in each cluster
                cluster_counts = Counter(cluster_labels)

                # Keep only clusters with at least a few coactive timepoints
                min_cluster_size = 3  # Adjust this threshold if needed
                valid_clusters = {k for k, v in cluster_counts.items() if v >= min_cluster_size}

                # Mask non-similar or infrequent clusters
                filtered_labels = np.array([label if label in valid_clusters else -1 for label in cluster_labels])

                # group timepoints that belong to the same ensemble
                ensemble_timepoints = {k: np.where(filtered_labels == k)[0] for k in np.unique(filtered_labels) if
                                       k != -1}

                for ensemble, timepoints in ensemble_timepoints.items():
                    print(f"Ensemble {ensemble}: {timepoints[:10]} ...")  # Print first 10 timepoints

                # identify neurons that are co-active in each ensemble
                #Each ensemble has a binary vector (length n_neurons) > True means that neuron was consistently active in that ensemble.
                ensemble_neurons = {ens: significant_vectors[times].mean(axis=0) > 0 for ens, times in
                                    ensemble_timepoints.items()}

                # Print first ensemble's active neurons
                print("Neurons in first ensemble:", np.where(ensemble_neurons[list(ensemble_neurons.keys())[0]])[0])

                # Perform hierarchical clustering using Ward linkage > groups the frames with high similarity into clusters
                # Neurons involved in these clusters are considered to be part of the same neuronal ensemble.
                # After clustering, each group (or cluster) corresponds to a neuronal ensemble.
                # The neuronal ensemble is defined as a group of neurons that are co-active during a given time window or set of frames.
                # paper used used single linkage for the first clustering step
                clusterer = AgglomerativeClustering(
                    metric="precomputed",  # Use the Jaccard distance matrix
                    linkage="single"  # Single linkage is compatible with Jaccard
                )
                # indicates which timepoints belong to the same coactivity pattern
                cluster_labels = clusterer.fit_predict(jaccard_matrix)

                # GROUPING TIMEPOINTS INTO ENSEMBLES
                #

                colors = plt.cm.plasma(np.linspace(0, 0.9, significant_vectors.shape[0]))

                # Step 4: Dimensionality reduction with PCA > cluster the significant vectors to extract neuronal ensembles
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

        plt.suptitle(f'Population vectors with significant co-actvity ({animal})')
        #plt.show()
        pdf.savefig(fig)
        plt.close(fig)

