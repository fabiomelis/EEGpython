import selection_alg
import core
import numpy as np
import matplotlib.pyplot as plt
import utils
import os

def compare_all_clips(matrix_4D, tw, fs):

    n_sbjs, n_clips, n_channels, n_samples = matrix_4D.shape

    vettori = []
    stringhe_di_stampa = []

    for clip in range(n_clips):
        reduced_data = matrix_4D[:, clip, :, :]

        print('Clip analizzata: ', clip + 1)

        values, channel_list = selection_alg.forward_selection_eer(n_channels, reduced_data, tw, fs, 'welch_2')
        #values, channel_list = selection_alg.forward_selection_auc(n_channels, reduced_data, tw, fs, 'off')
        #values, channel_list = selection_alg.backward_selection_eer(n_channels, reduced_data, tw, fs, 'off')
        #values, channel_list = selection_alg.backward_selection_auc(n_channels, reduced_data, tw, fs, 'off')


        vettori.append(values)
        stringhe_di_stampa.append(f'Clip {clip + 1}: {channel_list}')

    # Trasponi la lista di vettori per ottenere una lista di colonne invece di righe
    vettori_trasposti = np.array(vettori).T

    print('\n'.join(stringhe_di_stampa))

    # Crea il grafico con tutti i vettori sovrapposti
    plt.plot(range(1, len(values) + 1), vettori_trasposti, marker='o')
    plt.xlabel('Numero di Canali')
    plt.ylabel('Valori di EER')
    plt.title('Valori di EER sovrapposti per ogni Clip')
    plt.xticks(range(1, len(values) + 1))
    plt.show()






def compare_forward_with_list(matrix_4D, tw, fs):

    n_sbjs, n_clips, n_channels, n_samples = matrix_4D.shape


    stringhe_di_stampa = []
    output_folder = r'C:\Users\fabio\Desktop\ING INFORMATICA\TESI\img compare\EXP OFF vs lista migliore EXP OFF'

    for clip in range(n_clips):
        reduced_data = matrix_4D[:, clip, :, :]
        vettori = []

        print('Clip analizzata: ', clip + 1)

        values, channel_list = selection_alg.forward_selection_eer(n_channels, reduced_data, tw, fs, 'off')

        vettori.append(values)
        stringhe_di_stampa.append(f'Clip {clip + 1}: {channel_list}')

        # Migliore lista EXP OFF
        list_channel = [6, 1, 2, 5, 14, 8, 13, 3, 9, 11, 10, 7, 12, 4]

        # Migliore lista Welch_2
        #list_channel = [6, 1, 2, 5, 14, 8, 13, 3, 9, 11, 10, 7, 12, 4]
        values, channel_list = core.compute_EER_fixed_list(list_channel, reduced_data, tw, fs, 'off')

        vettori.append(values)
        stringhe_di_stampa.append(f'Clip {clip + 1}: {channel_list}')

        # Trasponi la lista di vettori per ottenere una lista di colonne invece di righe
        vettori_trasposti = np.array(vettori).T

        print('\n'.join(stringhe_di_stampa))

        # Crea il grafico con tutti i vettori sovrapposti
        plt.plot(range(1, len(values) + 1), vettori_trasposti, marker='o')
        plt.xlabel('Numero di Canali')
        plt.ylabel('Valori di EER')
        plt.title('Valori di EER sovrapposti')
        plt.xticks(range(1, len(values) + 1))
        plt.legend(['Forward Selection', 'Lista Canali Fissati'])
        plt.savefig(os.path.join(output_folder, f'grafico_clip_{clip + 1}.png'))
        plt.close()



def compare_all_FOOOF_features(dataset, tw, fs):

    n_sbjs, n_channels, n_samples = dataset.shape

    vettori = []
    stringhe_di_stampa = []

    string = 'exp'
    values , channel_list = selection_alg.forward_selection_eer(n_channels, dataset, tw, fs, string)

    vettori.append(values)
    stringhe_di_stampa.append(f'FOOOF features = EXP: {channel_list}')

    string = 'off'
    values, channel_list = selection_alg.forward_selection_eer(n_channels, dataset, tw, fs, string)

    vettori.append(values)
    stringhe_di_stampa.append(f'FOOOF features = EXP, OFF: {channel_list}')

    string = 'freq'
    values, channel_list = selection_alg.forward_selection_eer(n_channels, dataset, tw, fs, string)

    vettori.append(values)
    stringhe_di_stampa.append(f'FOOOF features = EXP, OFF, FREQ: {channel_list}')


    # Trasponi la lista di vettori per ottenere una lista di colonne invece di righe
    vettori_trasposti = np.array(vettori).T

    print('\n'.join(stringhe_di_stampa))

    titolo = 'Valori di EER sovrapposti per Features estratte'
    etichette_personalizzate = ['EXP', 'EXP, OFF', 'EXP, OFF, FREQ']

    utils.plot_compare(values,vettori_trasposti,titolo,etichette_personalizzate)


def compare_all_PSD_features(dataset, tw, fs):

    n_sbjs, n_channels, n_samples = dataset.shape

    vettori = []
    stringhe_di_stampa = []

    string = 'welch_1'
    values , channel_list = selection_alg.forward_selection_eer(n_channels, dataset, tw, fs, string)

    vettori.append(values)
    stringhe_di_stampa.append(f'PSD features = [(1, 3)]: {channel_list}')

    string = 'welch_2'
    values, channel_list = selection_alg.forward_selection_eer(n_channels, dataset, tw, fs, string)

    vettori.append(values)
    stringhe_di_stampa.append(f'PSD features = [(1, 3), (4,7)]: {channel_list}')

    string = 'welch_3'
    values, channel_list = selection_alg.forward_selection_eer(n_channels, dataset, tw, fs, string)

    vettori.append(values)
    stringhe_di_stampa.append(f'PSD features = [(1, 3), (4,7), (8, 12)]: {channel_list}')

    string = 'welch_4'
    values, channel_list = selection_alg.forward_selection_eer(n_channels, dataset, tw, fs, string)

    vettori.append(values)
    stringhe_di_stampa.append(f'PSD features = [(1, 3), (4,7), (8, 12), (13,30)]: {channel_list}')

    string = 'welch_5'
    values, channel_list = selection_alg.forward_selection_eer(n_channels, dataset, tw, fs, string)

    vettori.append(values)
    stringhe_di_stampa.append(f'PSD features = [(1, 3), (4,7), (8, 12), (13,30) , (30, 50)]: {channel_list}')

    string = 'welch_alpha'
    values, channel_list = selection_alg.forward_selection_eer(n_channels, dataset, tw, fs, string)

    vettori.append(values)
    stringhe_di_stampa.append(f'PSD features = [(8, 12)]: {channel_list}')

    string = 'welch_beta'
    values, channel_list = selection_alg.forward_selection_eer(n_channels, dataset, tw, fs, string)

    vettori.append(values)
    stringhe_di_stampa.append(f'PSD features = [(13,30)]: {channel_list}')

    string = 'welch_alpha_beta'
    values, channel_list = selection_alg.forward_selection_eer(n_channels, dataset, tw, fs, string)

    vettori.append(values)
    stringhe_di_stampa.append(f'PSD features = [(8, 12), (13,30)]: {channel_list}')



    # Trasponi la lista di vettori per ottenere una lista di colonne invece di righe
    vettori_trasposti = np.array(vettori).T

    print('\n'.join(stringhe_di_stampa))
    print(vettori_trasposti)

    titolo = 'Valori di EER sovrapposti per Features estratte'
    etichette_personalizzate = ['[(1, 3)]', '[(1, 3), (4,7)]', '[(1, 3), (4,7), (8, 12)]',
                                '[(1, 3), (4,7), (8, 12), (13,30)', '[(1, 3), (4,7), (8, 12), (13,30) , (30, 50)]',
                                '(8, 12)]', '[(13,30)', '(8, 12), (13,30)]']


    utils.plot_compare(values,vettori_trasposti,titolo,etichette_personalizzate)



def compare_FOOOF_vs_PSD(dataset, tw, fs):

    n_sbjs, n_channels, n_samples = dataset.shape

    vettori = []
    stringhe_di_stampa = []

    string = 'off'
    values, channel_list = selection_alg.forward_selection_eer(n_channels, dataset, tw, fs, string)

    vettori.append(values)
    stringhe_di_stampa.append(f'FOOOF features = EXP, OFF: {channel_list}')

    string = 'welch_5'
    values, channel_list = selection_alg.forward_selection_eer(n_channels, dataset, tw, fs, string)

    vettori.append(values)
    stringhe_di_stampa.append(f'PSD features = [(1,3), (4,7), (8,12), (13,30), (30,50)]: {channel_list}')


    # Trasponi la lista di vettori per ottenere una lista di colonne invece di righe
    vettori_trasposti = np.array(vettori).T

    print('\n'.join(stringhe_di_stampa))
    print(vettori)

    titolo = 'Valori di EER sovrapposti per Features estratte'
    etichette = ['FOOOF', 'PSD']

    utils.plot_compare(values,vettori_trasposti,titolo,etichette)


def compare_Forward_vs_Backward(dataset, tw, fs):

    n_sbjs, n_channels, n_samples = dataset.shape

    vettori = []
    stringhe_di_stampa = []

    string = 'welch_2'
    values, channel_list = selection_alg.forward_selection_auc(n_channels, dataset, tw, fs, string)

    vettori.append(values)
    stringhe_di_stampa.append(f'Forward Selection: {channel_list}')

    string = 'welch_2'
    values, channel_list = selection_alg.backward_selection_auc(n_channels, dataset, tw, fs, string)

    vettori.append(values)
    stringhe_di_stampa.append(f'Backward Selection: {channel_list}')


    # Trasponi la lista di vettori per ottenere una lista di colonne invece di righe
    vettori_trasposti = np.array(vettori).T

    print('\n'.join(stringhe_di_stampa))


    etichette_personalizzate = ['Forward', 'Backward']
    titolo = 'Confronto prestazioni tra Forward e Backward Selection'


    utils.plot_compare(values,vettori_trasposti,titolo,etichette_personalizzate)


def compare_vectors():


    vettori = []
    stringhe_di_stampa = []

    values = [0.3189, 0.2963, 0.2896, 0.2886, 0.2904, 0.2894, 0.2871, 0.2867, 0.2884, 0.2890, 0.2978, 0.2987, 0.3154,
              0.3231]

    channel_list = [2, 14, 6, 13, 10, 5, 8, 3, 11, 9, 1, 4, 12, 7]

    vettori.append(values)
    stringhe_di_stampa.append(f'Forward Selection with FOOOF: {channel_list}')

    values = [0.3201871657754011, 0.29198635976129583, 0.2927613733240332, 0.2953964194373402, 0.29378826629466015, 0.2900100751763156, 0.29404014570254977, 0.2933523211656204, 0.2941273347283577, 0.29645237541656977, 0.3011605828101992, 0.30661474075796324, 0.31631209796171433, 0.3274141672479268]
    channel_list = [6, 12, 11, 5, 14, 2, 3, 13, 8, 10, 7, 9, 4, 1]

    vettori.append(values)
    stringhe_di_stampa.append(f'Forward Selection with PSD: {channel_list}')

    # Trasponi la lista di vettori per ottenere una lista di colonne invece di righe
    vettori_trasposti = np.array(vettori).T

    print('\n'.join(stringhe_di_stampa))

    titolo = 'Valori di EER sovrapposti per Features estratte'
    #titolo = 'Confronto prestazioni tra Forward e Backward Selection'
    #titolo = 'Confronto prestazioni tra Forward e Lista Fissata'
    etichette = ['FOOOF','PSD']

    utils.plot_compare(values,vettori_trasposti,titolo,etichette)


def compare_new_PSD_features(dataset, tw, fs):

    n_sbjs, n_channels, n_samples = dataset.shape

    vettori = []
    stringhe_di_stampa = []


    string = 'welch_2'
    values, channel_list = selection_alg.forward_selection_eer(n_channels, dataset, tw, fs, string)

    vettori.append(values)
    stringhe_di_stampa.append(f'PSD features = [(0, 10), (10,20)]: {channel_list}')

    string = 'welch_4'
    values, channel_list = selection_alg.forward_selection_eer(n_channels, dataset, tw, fs, string)

    vettori.append(values)
    stringhe_di_stampa.append(f'PSD features = [(1, 3), (4,7), (8, 12), (13,30)]: {channel_list}')

    string = 'welch_5'
    values, channel_list = selection_alg.forward_selection_eer(n_channels, dataset, tw, fs, string)

    vettori.append(values)
    stringhe_di_stampa.append(f'PSD features = [(1, 3), (4,7), (8, 12), (13,30) , (30, 50)]: {channel_list}')


    # Trasponi la lista di vettori per ottenere una lista di colonne invece di righe
    vettori_trasposti = np.array(vettori).T

    print('\n'.join(stringhe_di_stampa))
    print(vettori)

    titolo = 'Valori di EER sovrapposti per Features estratte'
    etichette_personalizzate = ['[(0, 10), (10,20)]','[(1, 3), (4,7), (8, 12), (13,30)]', '[(1, 3), (4,7), (8, 12), (13,30) , (30, 50)]']


    utils.plot_compare(values,vettori_trasposti,titolo,etichette_personalizzate)