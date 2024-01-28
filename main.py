import numpy as np
import core
import selection_alg_old
from scipy.io import loadmat
import selection_alg



datapath = 'C:\\Users\\fabio\\Desktop\\ING INFORMATICA\\TESI\\DREAMER\\DREAMER_base\\dreamer_base.mat'


# Range di lunghezza delle finsetre temporali del segnale (determina la lunghezza delle epoche)
tw_range = np.arange(1, 12.5, 0.5)


# Frequenza di campionamento con cui sono stati presi i dati raw
fs = 128


# Carica i dati (procedura necessaria perche è un file .mat)
data = loadmat(datapath)
EEG_filtt = data['my_base']

n_sbjs, n_clips, n_channels, n_samples = EEG_filtt.shape

'''
# OLD CODE: MATRICE CON CLIP SINGOLA

i_clip = 3

# creare una concatenazione con tutte le clip con numpy stack


# Estrai i tracciati per la clip specificata (otteniamo una matrice 3D)
reduced_data = EEG_filtt[:, i_clip, :, :]

print(reduced_data.shape)
'''

# finestra fissata a 10 per tutto
selected_tw = 10





#selection_alg.forward_selection_eer(n_channels, reduced_data, selected_tw, fs)
#selection_alg.forward_selection_auc(n_channels, reduced_data, selected_tw, fs)
#selection_alg.backward_selection_eer(n_channels, reduced_data, selected_tw, fs)
#selection_alg.backward_selection_auc(n_channels, reduced_data, selected_tw, fs)


#martedi alle 16 nuovo incontro7
'''
import numpy as np

# Dimensioni della matrice a 4 dimensioni
soggetti = 4
clip = 2
canali = 1
campionamenti = 3

# Creazione di una matrice a 4 dimensioni di esempio
matrice_4D = np.random.rand(soggetti, clip, canali, campionamenti)

print(matrice_4D)

# Concatenazione delle clip
matrice_3D = np.concatenate(np.split(matrice_4D, clip, axis=1), axis=3)

# Stampa delle nuove dimensioni
print("Dimensioni della matrice a 3 dimensioni:", matrice_3D.shape)

print(matrice_3D)
'''


# CONCATENIAMO LE 18 CLIPS

matrice_3D = np.concatenate(np.split(EEG_filtt, n_clips, axis=1), axis=3)

print(matrice_3D.shape)

reduced_data = matrice_3D[:, 0, :, :]

print(reduced_data.shape)



#selection_alg.forward_selection_eer(n_channels, reduced_data, selected_tw, fs)
#selection_alg.forward_selection_auc(n_channels, reduced_data, selected_tw, fs)
#selection_alg.backward_selection_eer(n_channels, reduced_data, selected_tw, fs)
#selection_alg.backward_selection_auc(n_channels, reduced_data, selected_tw, fs)

# 15:30 iniziato
#20:15 analizzati 6 canali, inizia il 7

#2:00 tornato e ha finito non so quando...