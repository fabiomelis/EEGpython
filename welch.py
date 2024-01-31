import numpy as np
from scipy.signal import welch
import utils


def extract_psd_features(tmp_data, i_sbj, i_channel, n_ch_sel, fs, n_features):

    indices = []

    for i in range(1, n_features + 1):
        indices = [(n_ch_sel * n_features) - n_features + i -1 for i in range(1, n_features + 1)]

    tmp_signal = np.squeeze(tmp_data[i_sbj - 1, i_channel - 1, :])

    # Per evitare errori dovuti alla frequenza bisogna impostare nperseg a 255 e non a 256 come di default
    frequencies, power_spectrum = welch(tmp_signal, fs=fs, nperseg=min(255, len(tmp_signal)))

    frequencies_of_interest = np.linspace(1, 50, 20)

    indices_of_interest = np.searchsorted(frequencies, frequencies_of_interest, side='left')

    psd_of_interest = power_spectrum[indices_of_interest]

    #print("Valori del PSD per le frequenze di interesse:", psd_of_interest)

    #utils.plot_psd_freq_features(frequencies,power_spectrum,indices_of_interest,psd_of_interest)

    #utils.plot_ps_welch(frequencies, power_spectrum)

    return psd_of_interest, indices
