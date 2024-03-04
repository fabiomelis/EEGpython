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
#EEG_filtt, fs = dataset.matrix_4d_DREAMER()
#n_sbjs, n_clips, n_channels, n_samples = EEG_filtt.shape


# Dataset Physionet EEG 109
matrix, fs = dataset.matrix_3d_109()
n_sbjs, n_channels, n_samples = matrix.shape


tw = 10


#compare.compare_vectors()



# MATRICE CLIP SINGOLA
#i_clip = 3
#reduced_data = EEG_filtt[:, i_clip, :, :]

#selection_alg.forward_selection_auc(n_channels,matrix,tw,fs,'welch_2')








# CONCATENIAMO LE 18 CLIPS
#matrice_3D = np.concatenate(np.split(EEG_filtt, n_clips, axis=1), axis=3)
#print(matrice_3D.shape)
#reduced_data = matrice_3D[:, 0, :, :]
#print(reduced_data.shape)

#selection_alg.forward_selection_eer(n_channels, reduced_data, tw, fs, string='welch_2')





end_time = time.time()

elapsed_time = end_time - start_time

print("Secondi impiegati per l'operazione: ",elapsed_time)