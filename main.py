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

i_clip = 3

# creare una concatenazione con tutte le clip con numpy stack


# Estrai i tracciati per la clip specificata (otteniamo una matrice 3D)
reduced_data = EEG_filtt[:, i_clip, :, :]

# finestra fissata a 10 per tutto
selected_tw = 10

# Per "forward" sono limiti minimi perche i valori sono più alti partendo da un canale alla volta
# Per "backward" sono limiti massimi perche i valori sono più bassi partendo da tutti i canali meno uno
EER_limit = 0.1
AUC_limit = 0.1

channel_limit = 8

algorithm = 'forward' #or 'backward'


#selection_alg.forward_selection_eer(n_channels, reduced_data, selected_tw, fs)
selection_alg.forward_selection_auc(n_channels, reduced_data, selected_tw, fs)
#selection_alg.backward_selection_eer(n_channels, reduced_data, selected_tw, fs)
#selection_alg.backward_selection_auc(n_channels, reduced_data, selected_tw, fs)

#selected_channels, best_EER, best_AUC = selection_alg.apply_selection(algorithm, n_channels, reduced_data, selected_tw, fs, EER_limit, AUC_limit, channel_limit)

#print('Migliori canali: ', selected_channels, ' con EER: ', best_EER, ' e AUC: ', best_AUC)

selected_channels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
#selected_channels = [1, 2, 3, 4, 6, 8, 9, 12, 14]
#selected_channels = [1, 2, 3, 4, 6, 8, 9, 12]
#selected_channels = [1, 3, 4, 6, 8, 9, 12]
#selected_channels = [4, 6, 9, 12]
#selected_channels = [4, 6, 12]


#core.compute_total_performance(reduced_data, tw_range, fs, selected_channels)

#martedi alle 16 nuovo incontro7