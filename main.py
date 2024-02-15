import numpy as np
import core
from scipy.io import loadmat
import selection_alg
import time
import utils
import compare

start_time = time.time()

datapath = 'C:\\Users\\fabio\\Desktop\\ING INFORMATICA\\TESI\\DREAMER\\DREAMER_base\\dreamer_base.mat'


# Range di lunghezza delle finsetre temporali del segnale (determina la lunghezza delle epoche)
tw_range = np.arange(1, 12.5, 0.5)


# Frequenza di campionamento con cui sono stati presi i dati raw e finsestra di campionamento fissata
fs = 128
tw = 10


# Carica i dati (procedura necessaria perche è un file .mat)
data = loadmat(datapath)
EEG_filtt = data['my_base']


#compare.compare_all_clips(EEG_filtt, tw, fs)

#compare.compare_forward_with_list(EEG_filtt, tw, fs)


n_sbjs, n_clips, n_channels, n_samples = EEG_filtt.shape


# FORWARD SINGLE CLIP

i_clip = 3

reduced_data = EEG_filtt[:, i_clip, :, :]

#compare.compare_FOOOF_vs_PSD(reduced_data,tw,fs)

#selection_alg.backward_selection_eer(n_channels,reduced_data,tw,fs,'welch_3')

compare.compare_Forward_vs_Backward(reduced_data,tw,fs)

#compare.compare_all_FOOOF_features(reduced_data, tw, fs)

#selection_alg.forward_selection_eer(n_channels,reduced_data,tw,fs,'welch')

#EER, AUC = core.compute_EER_AUC_exp_off(reduced_data, tw, fs, [4, 6, 12, 9, 8, 1])
#print(EER)


'''
# OLD CODE: MATRICE CON CLIP SINGOLA

#i_clip = 5


for clip in range( n_clips ):

    # Estrai i tracciati per la clip specificata (otteniamo una matrice 3D)
    reduced_data = EEG_filtt[:, clip, :, :]

    print(reduced_data.shape)


    

    EER, AUC = core.compute_EER_AUC_exp(reduced_data, tw, fs, selected_channels=[1,4,6,9,12])

    print(f'Clip {clip + 1} -> EER: ', EER, '; AUC: ', AUC)
'''


'''
# CONCATENIAMO LE 18 CLIPS

matrice_3D = np.concatenate(np.split(EEG_filtt, n_clips, axis=1), axis=3)

print(matrice_3D.shape)

reduced_data = matrice_3D[:, 0, :, :]

print(reduced_data.shape)


selection_alg.forward_selection_eer(n_channels, reduced_data, tw, fs, string='welch')
'''



end_time = time.time()

elapsed_time = end_time - start_time

print("Secondi impiegati per l'operazione: ",elapsed_time)