import mne
import numpy as np
import os
from scipy.io import loadmat

def matrix_3d_109():

    fs = 160

    base_dir = "C:\\Users\\fabio\\Desktop\\ING INFORMATICA\\TESI\\eeg-motor-movementimagery-dataset-1.0.0\\files"

    # Solo alcuni campioni hanno 9760 campioni, quindi prendiamo i primi 9600 di tutti
    n_samples = 9600

    # Lista per memorizzare i dati dei soggetti
    data_list = []

    # Itera attraverso le cartelle
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        if os.path.isdir(folder_path):  # Verifica se è una cartella
            subj_idx = int(folder_name[1:])  # Ottieni l'indice del soggetto dalla cartella
            file_path = os.path.join(folder_path, f"S{subj_idx:03d}R01.edf")
            data = mne.io.read_raw_edf(file_path)
            raw_data = data.get_data()

            raw_data_9600 = raw_data[:, :n_samples]
            data_list.append(raw_data_9600)

    # Converti la lista in un array numpy
    matrix = np.array(data_list)

    print(matrix.shape)

    return matrix, fs


def matrix_4d_DREAMER():

    fs = 128

    datapath = 'C:\\Users\\fabio\\Desktop\\ING INFORMATICA\\TESI\\DREAMER\\DREAMER_base\\dreamer_base.mat'

    data = loadmat(datapath)
    matrix = data['my_base']
    print(matrix.shape)

    return matrix, fs