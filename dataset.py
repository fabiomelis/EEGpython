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

def matrix_3d_10Hz():

    fs = 2048

    import mne
    import numpy as np

    file1 = "C:\\Users\\fabio\\Desktop\\ING INFORMATICA\\TESI\\eeg-signals-from-an-rsvp-task-1.0.0\\10-Hz\\rsvp_10Hz_02a.edf"
    file2 = "C:\\Users\\fabio\\Desktop\\ING INFORMATICA\\TESI\\eeg-signals-from-an-rsvp-task-1.0.0\\10-Hz\\rsvp_10Hz_03a.edf"
    file3 = "C:\\Users\\fabio\\Desktop\\ING INFORMATICA\\TESI\\eeg-signals-from-an-rsvp-task-1.0.0\\10-Hz\\rsvp_10Hz_04a.edf"
    file4 = "C:\\Users\\fabio\\Desktop\\ING INFORMATICA\\TESI\\eeg-signals-from-an-rsvp-task-1.0.0\\10-Hz\\rsvp_10Hz_06a.edf"
    file5 = "C:\\Users\\fabio\\Desktop\\ING INFORMATICA\\TESI\\eeg-signals-from-an-rsvp-task-1.0.0\\10-Hz\\rsvp_10Hz_08a.edf"
    file6 = "C:\\Users\\fabio\\Desktop\\ING INFORMATICA\\TESI\\eeg-signals-from-an-rsvp-task-1.0.0\\10-Hz\\rsvp_10Hz_09a.edf"
    file7 = "C:\\Users\\fabio\\Desktop\\ING INFORMATICA\\TESI\\eeg-signals-from-an-rsvp-task-1.0.0\\10-Hz\\rsvp_10Hz_10a.edf"
    file8 = "C:\\Users\\fabio\\Desktop\\ING INFORMATICA\\TESI\\eeg-signals-from-an-rsvp-task-1.0.0\\10-Hz\\rsvp_10Hz_11a.edf"
    file9 = "C:\\Users\\fabio\\Desktop\\ING INFORMATICA\\TESI\\eeg-signals-from-an-rsvp-task-1.0.0\\10-Hz\\rsvp_10Hz_12a.edf"
    file10 = "C:\\Users\\fabio\\Desktop\\ING INFORMATICA\\TESI\\eeg-signals-from-an-rsvp-task-1.0.0\\10-Hz\\rsvp_10Hz_13a.edf"

    data1 = mne.io.read_raw_edf(file1)
    data2 = mne.io.read_raw_edf(file2)
    data3 = mne.io.read_raw_edf(file3)
    data4 = mne.io.read_raw_edf(file4)
    data5 = mne.io.read_raw_edf(file5)
    data6 = mne.io.read_raw_edf(file6)
    data7 = mne.io.read_raw_edf(file7)
    data8 = mne.io.read_raw_edf(file8)
    data9 = mne.io.read_raw_edf(file9)
    data10 = mne.io.read_raw_edf(file10)

    raw_data1 = data1.get_data()
    raw_data2 = data2.get_data()
    raw_data3 = data3.get_data()
    raw_data4 = data4.get_data()
    raw_data5 = data5.get_data()
    raw_data6 = data6.get_data()
    raw_data7 = data7.get_data()
    raw_data8 = data8.get_data()
    raw_data9 = data9.get_data()
    raw_data10 = data10.get_data()

    # you can get the metadata included in the file and a list of all channels:info = data.info
    channels = data1.ch_names
    print(channels)

    print(raw_data1.shape)

    raw_data1_filt = raw_data1[:8, :368640]
    raw_data2_filt = raw_data2[:8, :368640]
    raw_data3_filt = raw_data3[:8, :368640]
    raw_data4_filt = raw_data4[:8, :368640]
    raw_data5_filt = raw_data5[:8, :368640]
    raw_data6_filt = raw_data6[:8, :368640]
    raw_data7_filt = raw_data7[:8, :368640]
    raw_data8_filt = raw_data8[:8, :368640]
    raw_data9_filt = raw_data9[:8, :368640]
    raw_data10_filt = raw_data10[:8, :368640]

    n_sbjs = 10
    n_channels = 8
    n_samples = 368640  # 3minuti a fs 2048

    matrix = np.zeros((n_sbjs, n_channels, n_samples))

    matrix[0] = raw_data1_filt
    matrix[1] = raw_data2_filt
    matrix[2] = raw_data3_filt
    matrix[3] = raw_data4_filt
    matrix[4] = raw_data5_filt
    matrix[5] = raw_data6_filt
    matrix[6] = raw_data7_filt
    matrix[7] = raw_data8_filt
    matrix[8] = raw_data9_filt
    matrix[9] = raw_data10_filt

    print(matrix.shape)

    return matrix, fs