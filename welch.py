import numpy as np
from scipy.signal import welch
import utils
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simps


def extract_area_from_intervals(tmp_data, i_sbj, i_channel, n_ch_sel, fs, n_features, frequency_intervals):

    #frequency_intervals = [(0, 10), (10,20), (20, 30), (30,40) , (40, 50)]

    indices = []

    for i in range(1, n_features + 1):
        indices = [(n_ch_sel * n_features) - n_features + i - 1 for i in range(1, n_features + 1)]

    tmp_signal = np.squeeze(tmp_data[i_sbj - 1, i_channel - 1, :])

    # Per evitare errori dovuti alla frequenza bisogna impostare nperseg a 255 e non a 256 come di default
    frequencies, power_spectrum = welch(tmp_signal, fs=fs, nperseg=min(255, len(tmp_signal)))

    # Calcola l'area totale sotto la curva del PSD
    total_area = simps(power_spectrum, frequencies)

    features = []

    for interval in frequency_intervals:

        # Trova gli indici delle frequenze all'interno dell'intervallo
        indices_in_interval = np.where((frequencies >= interval[0]) & (frequencies <= interval[1]))

        # Estrai le frequenze e il PSD all'interno dell'intervallo
        frequencies_in_interval = frequencies[indices_in_interval]
        psd_in_interval = power_spectrum[indices_in_interval]

        # Calcola l'area sotto la curva del PSD all'interno dell'intervallo
        area_in_interval = simps(psd_in_interval, frequencies_in_interval)

        # Normalizza l'area rispetto all'area totale
        normalized_area = area_in_interval / total_area

        # Aggiungi l'area normalizzata alla lista delle features
        features.append(normalized_area)

    #print(features)
    #print(indices)
    '''
    plt.bar(range(len(frequency_intervals)), features, tick_label=[str(interval) for interval in frequency_intervals])
    plt.xlabel('Intervallo di Frequenza')
    plt.ylabel('Area sotto il PSD')
    plt.title('Area sotto il PSD per intervallo di frequenza')
    plt.show()
    '''

    '''
    plt.figure(figsize=(10, 6))
    plt.semilogy(frequencies, power_spectrum, label='Spettro di Potenza', color='blue')

    # Evidenziamo gli intervalli di frequenza di interesse
    for interval in frequency_intervals:
        plt.fill_between(frequencies, 0, power_spectrum,
                         where=((frequencies >= interval[0]) & (frequencies <= interval[1])), alpha=0.3)

    # Aggiungiamo una legenda e le etichette degli assi
    plt.xlabel('Frequenza (Hz)')
    plt.ylabel('Spettro di Potenza')
    plt.title('Spettro di Potenza con Intervalli di Frequenza di Interesse')
    plt.legend()
    plt.grid(True)
    plt.show()
    '''

    return features, indices



def extract_psd_features(tmp_data, i_sbj, i_channel, n_ch_sel, fs, n_features):

    indices = []

    for i in range(1, n_features + 1):
        indices = [(n_ch_sel * n_features) - n_features + i -1 for i in range(1, n_features + 1)]

    tmp_signal = np.squeeze(tmp_data[i_sbj - 1, i_channel - 1, :])

    # Per evitare errori dovuti alla frequenza bisogna impostare nperseg a 255 e non a 256 come di default
    frequencies, power_spectrum = welch(tmp_signal, fs=fs, nperseg=min(255, len(tmp_signal)))

    frequencies_of_interest = np.linspace(1, 50, 2)

    indices_of_interest = np.searchsorted(frequencies, frequencies_of_interest, side='left')

    psd_of_interest = power_spectrum[indices_of_interest]

    #print("Valori del PSD per le frequenze di interesse:", psd_of_interest)

    #utils.plot_psd_freq_features(frequencies,power_spectrum,indices_of_interest,psd_of_interest)

    #utils.plot_ps_welch(frequencies, power_spectrum)

    return psd_of_interest, indices
