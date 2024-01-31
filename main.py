import numpy as np
import core
from scipy.io import loadmat
import selection_alg
import time
import utils

start_time = time.time()

datapath = 'C:\\Users\\fabio\\Desktop\\ING INFORMATICA\\TESI\\DREAMER\\DREAMER_base\\dreamer_base.mat'


# Range di lunghezza delle finsetre temporali del segnale (determina la lunghezza delle epoche)
tw_range = np.arange(1, 12.5, 0.5)


# Frequenza di campionamento con cui sono stati presi i dati raw
fs = 128


# Carica i dati (procedura necessaria perche è un file .mat)
data = loadmat(datapath)
EEG_filtt = data['my_base']

n_sbjs, n_clips, n_channels, n_samples = EEG_filtt.shape



# OLD CODE: MATRICE CON CLIP SINGOLA

i_clip = 3


# Estrai i tracciati per la clip specificata (otteniamo una matrice 3D)
reduced_data = EEG_filtt[:, i_clip, :, :]

print(reduced_data.shape)



# Finestra fissata a 10 per tutto
selected_tw = 10



selection_alg.forward_selection_eer(n_channels, reduced_data, selected_tw, fs)
#selection_alg.forward_selection_auc(n_channels, reduced_data, selected_tw, fs)
#selection_alg.backward_selection_eer(n_channels, reduced_data, selected_tw, fs)
#selection_alg.backward_selection_auc(n_channels, reduced_data, selected_tw, fs)


#AUC_values = [1, 2, 3, 8, 4,4, 3, 6, 8, 6, 10, 10, 10 ,10]
#utils.plot_AUC_backward(AUC_values)


'''
# CONCATENIAMO LE 18 CLIPS

matrice_3D = np.concatenate(np.split(EEG_filtt, n_clips, axis=1), axis=3)

print(matrice_3D.shape)

reduced_data = matrice_3D[:, 0, :, :]

print(reduced_data.shape)
'''

#selection_alg.forward_selection_eer(n_channels, reduced_data, selected_tw, fs)
#selection_alg.forward_selection_auc(n_channels, reduced_data, selected_tw, fs)
#selection_alg.backward_selection_eer(n_channels, reduced_data, selected_tw, fs)
#selection_alg.backward_selection_auc(n_channels, reduced_data, selected_tw, fs)


end_time = time.time()

elapsed_time = end_time - start_time

print("Secondi impiegati per l'operazione: ",elapsed_time)