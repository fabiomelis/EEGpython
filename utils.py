import numpy as np
import matplotlib.pyplot as plt


def plot_ps_welch(frequencies, power_spectrum):
    plt.figure(figsize=(10, 6))
    plt.semilogy(frequencies, power_spectrum)
    plt.title('Spettro di Potenza con Metodo di Welch')
    plt.xlabel('Frequenza (Hz)')
    plt.ylabel('Densità di Potenza')
    plt.grid(True)
    plt.show()


def find_ps_fft(signal, fs):

    # Trova lo spettro di frequenza del segnale EEG
    fft_result = np.fft.fft(signal)

    # Calcola le frequenze corrispondenti allo spettro di potenza
    n = len(signal)
    frequencies = np.fft.fftfreq(n, 1 / fs)

    # Calcola lo spettro di potenza
    power_spectrum = np.abs(fft_result) ** 2
    return (frequencies, power_spectrum)


def print_AUC(tw_range , AUCtot):
    plt.figure(figsize=(7, 4))
    plt.plot(tw_range, AUCtot, label='AUC')
    plt.title('AUC vs. tw_range')
    plt.xlabel('tw_range')
    plt.ylabel('AUC')
    plt.legend()
    plt.grid(True)
    plt.show()

def print_EER(tw_range, EERtot):
    plt.figure(figsize=(7, 4))
    plt.plot(tw_range, EERtot, label='EER')
    plt.title('EER vs. tw_range')
    plt.xlabel('tw_range')
    plt.ylabel('EER')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_ROC_curve(FAR, FRR, AUC):
    plt.figure(figsize=(5, 5))
    plt.plot(FAR, FRR, label=f'AUC = {AUC:.2f}')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.xlabel('False Positive Rate (FAR)')
    plt.ylabel('False Rejection Rate (FRR)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_EER_forward(EER_values):
    plt.plot(range(1, len(EER_values) + 1), EER_values, marker='o')
    plt.xlabel('Numero di Canali')
    plt.ylabel('Valore EER')
    plt.title('EER vs Numero di Canali Selezionati')
    plt.xticks(range(1, len(EER_values) + 1))
    plt.tight_layout()
    plt.show()

def plot_AUC_forward(AUC_values):
    plt.plot(range(1, len(AUC_values) + 1), AUC_values, marker='o')
    plt.xlabel('Numero di Canali')
    plt.ylabel('Valore AUC')
    plt.title('AUC vs Numero di Canali Selezionati')
    plt.xticks(range(1, len(AUC_values) + 1))
    plt.tight_layout()
    plt.show()

def plot_EER_backward(EER_values):
    plt.plot(range(len(EER_values), 0, -1), EER_values, marker='o')
    plt.xlabel('Numero di Canali')
    plt.ylabel('Valore EER')
    plt.title('EER con Backward Selection')
    plt.xticks(range(len(EER_values), 0, -1), range(1, len(EER_values) + 1))
    plt.tight_layout()
    plt.show()


def plot_AUC_backward(AUC_values):
    vettore_assex = np.arange(len(AUC_values), 0, -1)
    plt.plot(vettore_assex, AUC_values, marker='o')
    plt.xlabel('Numero di Canali')
    plt.ylabel('Valore AUC')
    plt.title('AUC con Backward Selection')
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.show()

