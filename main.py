import numpy as np
import core
from scipy.io import loadmat
import selection_alg
import time
import utils
import compare
import dataset

start_time = time.time()


# Dataset DREAMER
EEG_filtt, fs = dataset.matrix_4d_DREAMER()
n_sbjs, n_clips, n_channels, n_samples = EEG_filtt.shape


# Dataset Physionet EEG 109
#matrix, fs = dataset.matrix_3d_109()
#n_sbjs, n_channels, n_samples = matrix.shape


tw = 10


#compare.compare_forward_with_list(EEG_filtt,tw,fs)

# MATRICE CLIP SINGOLA

i_clip = 3

reduced_data = EEG_filtt[:, i_clip, :, :]

#compare.compare_FOOOF_vs_PSD(reduced_data,tw,fs)


#selection_alg.forward_selection_auc(n_channels,matrix,tw,fs,'welch_2')

#compare.compare_Forward_vs_Backward(reduced_data,tw,fs)

#compare.compare_all_FOOOF_features(reduced_data, tw, fs)

#selection_alg.forward_selection_eer(n_channels,reduced_data,tw,fs,'welch_new')

compare.compare_new_PSD_features(reduced_data,tw,fs)

#EER, AUC = core.compute_EER_AUC_welch(reduced_data, tw, fs,[1,2,3,4,5,6,7,8,9,10,11,12,13,14],2,[(0,10),(10,20)])
#print(EER)



'''
# ITERAZIONE CLIPS


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


selection_alg.backward_selection_eer(n_channels, reduced_data, tw, fs, string='off')
'''

#compare.compare_vectors()


end_time = time.time()

elapsed_time = end_time - start_time

print("Secondi impiegati per l'operazione: ",elapsed_time)