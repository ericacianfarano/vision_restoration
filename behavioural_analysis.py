from helpers import *

def load_dlc_pose_estimates(path_to_behav, likelihood_threshold):
    '''
    :param path_to_behav: path to behavioural folder, ex: r'G:\vision_restored\data\EC_RD1_04\20240626\brs\brs_000_000\behav'
    :param likelihood_threshold (float): flt between 0 and 1. DLC likelihood values below likelihood_threshold will be naned out
    :return: poses (array): DLC pose estimates of shape n_timepoints x n_poses
        [top_pupilX, top_pupilY, right_pupilX, right_pupilY, bottom_pupilX, bottom_pupilY, left_pupilX, left_pupilY,]
    '''
    dlc_file_name = [file for file in os.listdir(path_to_behav) if file.endswith('.csv')][0]
    dlc_file_path = os.path.join(path_to_behav, dlc_file_name)

    # skip the first 2 rows, and the first column
    pose_estimation_csv = np.genfromtxt(dlc_file_path, delimiter=',', skip_header=1)[2:, 1:]

    # extract indices where liklihood values are less than the set threshold
    likelihoods = pose_estimation_csv[:, 2::3]  # columns of the likelihood values
    subthreshold_likelihood = np.where(np.all(likelihoods < likelihood_threshold, axis=1))[0]

    # delete likelihood values from poses > delete every third columns starting from the second column
    poses = np.delete(pose_estimation_csv, np.s_[2::3], axis=1)

    # nan out values that are less than the likelihood threshold
    poses[subthreshold_likelihood, :] *= np.nan

    return poses

def fit_ellipse(data):
    """
    Fit an ellipse to given points data.

    Parameters:
    - data: numpy array of shape (n_timepoints, 8) where columns are [topX, topY, rightX, rightY, bottomX, bottomY, leftX, leftY].

    Returns:
    - x_center, y_center, width, height, angle: Parameters of the fitted ellipse.
    """
    # Extract points from data array
    topX, topY = data[:, 0], data[:, 1]
    rightX, rightY = data[:, 2], data[:, 3]
    bottomX, bottomY = data[:, 4], data[:, 5]
    leftX, leftY = data[:, 6], data[:, 7]

    # Center of the ellipse
    x_center = (leftX + rightX) / 2
    y_center = (topY + bottomY) / 2

    # Horizontal and vertical diameters
    height = np.sqrt((rightX - leftX)**2 + (rightY - leftY)**2)
    width = np.sqrt((topX - bottomX)**2 + (topY - bottomY)**2)

    # Angle of rotation (assuming you have the angle or calculate it)
    angle = np.arctan2(topY - bottomY, topX - bottomX)

    return x_center, y_center, width, height, angle

def plot_ellipse(ax, x_center, y_center, width, height, angle, points=None, color = 'red'):
    """
    Plot an ellipse on a given axis.

    Parameters:
    - ax: Matplotlib axis object where the ellipse will be plotted.
    - x_center, y_center, width, height, angle: Parameters of the ellipse.
    - points: Optional, numpy array of shape (4, 2) containing the top, bottom, left, and right points coordinates.

    Returns:
    - None
    """
    # Create the ellipse
    ellipse = Ellipse((x_center, y_center), width, height, angle=np.degrees(angle), edgecolor=color, fc='None')

    # Add the ellipse to the plot
    ax.add_patch(ellipse)

    # Plot the points if provided
    if points is not None:
        for i in range(4):
            ax.scatter(points[i*2], points[(i*2)+1], color=color)

    ax.set_aspect('equal', adjustable='box')


def moving_average(data, window_size = 5):
    '''
    Smoothing to remove noise > moving average filter

    Move the window over the data one point at a time, calculating the average of the data points within the window at each step.
    :param data:
    :param window_size:
    :return:
    '''

    # np.convolve convolves the data array with the array of ones of length 'window_size', normalizing by window_size > calcualte avg of each window
    # 'valid' > so output array only has values where window fully overlaps with input data > to avoid boundary effects
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def interpolate_nans(signal):
    """Interpolate NaN values in a signal."""
    nans = np.isnan(signal)
    not_nans = ~nans
    indices = np.arange(len(signal))
    interpolator = interp1d(indices[not_nans], signal[not_nans], bounds_error=False, fill_value="extrapolate")
    signal[nans] = interpolator(indices[nans])
    return signal

def analyze_frequency (signal, sampling_rate):
    '''
    Fast Fourier Transform (FFT) > frequency analysis to identify dominant frequencies (i.e., repetitive movements)
    :param signal:
    :param sampling_rate: n fps, acquisition rate, number of frames per second
    :return:
        - freq (array): > array of frequency values
        - np.abs(fourier) (array) > magnitude of each frequency component, which represents the amplitude of that frequency in the signal
    '''

    n = len(signal)                         # N samples in the signal array
    fourier = fft(signal)                   # Compute FFT of the signal > complex array that represents amplitude & phase of each frequency component of the signal
    freq = fftfreq(n, 1 / sampling_rate)    # array of frequencie values that corresponds to the FFT result/components in 'fourier''
    return freq, np.abs(fourier)            # frequencies & magnitude of the FFT result, respectively.

def interpolate_nans_1d(signal):
    """Interpolate NaN values in a signal."""
    nans = np.isnan(signal)
    not_nans = ~nans
    indices = np.arange(len(signal))
    interpolator = interp1d(indices[not_nans], signal[not_nans], bounds_error=False, fill_value="extrapolate")
    signal[nans] = interpolator(indices[nans])
    return signal

# behav_path = r'G:\vision_restored\data\EC_RD1_04\20240627\brs\brs_000_000\behav'
# dlc_file_name =[file for file in os.listdir(behav_path) if file.endswith('.csv')][0]
# dlc_file_path = os.path.join(behav_path, dlc_file_name)
# data = np.genfromtxt(dlc_file_path, delimiter=',', skip_header=1)[2:, 1:]
#
# likelihoods = data[:, 2::3]
# subthreshold_likelihood = np.where(np.all(likelihoods < 0.7, axis=1))[0]
#
# # delete likelihood values from poses > delete every third columns starting from the second column
# poses = np.delete(data, np.s_[2::3], axis=1)
#
# poses[subthreshold_likelihood, :] *= np.nan

#
# fps = 20
# x_center, y_center,width, height, angle = fit_ellipse(poses)
#
# x_center_smooth = moving_average(x_center, window_size = 3)
# y_center_smooth = moving_average(y_center, window_size = 3)
#
# x_center_smooth_interp = interpolate_nans_1d(x_center_smooth)
# y_center_smooth_interp = interpolate_nans_1d(y_center_smooth)
#
#
# # Calculate velocities
# dt = 1/fps # seconds
# velocity_x = np.diff(x_center_smooth_interp) / dt  # velocity components in x
# velocity_y = np.diff(y_center_smooth_interp) / dt  # velocity components in y
# velocity = np.sqrt(velocity_x**2 + velocity_y**2) # velocity magnitude
#
#
# frequency_x, fourier_x = analyze_frequency(velocity_x, fps)
# frequency_y, fourier_y = analyze_frequency(velocity_y, fps)
# frequency, fourier = analyze_frequency(velocity, fps)
#
# plt.figure(figsize=(12, 9))
# plt.subplot(3, 1, 1)
# plt.plot(frequency_x, fourier_x)
# plt.title('Frequency Analysis of X Velocity')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Amplitude')
#
# plt.subplot(3, 1, 2)
# plt.plot(frequency_y, fourier_y)
# plt.title('Frequency Analysis of Y Velocity')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Amplitude')
#
# plt.subplot(3, 1, 3)
# plt.plot(frequency, fourier)
# plt.title('Frequency Analysis of Velocity')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Amplitude')
#
# plt.tight_layout()
# plt.show()