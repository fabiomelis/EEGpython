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
    stringhe_di_stampa.append(f'PSD features = (0, 10): {channel_list}')

    string = 'welch_2'
    values, channel_list = selection_alg.forward_selection_eer(n_channels, dataset, tw, fs, string)

    vettori.append(values)
    stringhe_di_stampa.append(f'PSD features = [(0, 10), (10,20)]: {channel_list}')

    string = 'welch_3'
    values, channel_list = selection_alg.forward_selection_eer(n_channels, dataset, tw, fs, string)

    vettori.append(values)
    stringhe_di_stampa.append(f'PSD features = [(0, 10), (10,20), (20, 30)]: {channel_list}')

    string = 'welch_5'
    values, channel_list = selection_alg.forward_selection_eer(n_channels, dataset, tw, fs, string)

    vettori.append(values)
    stringhe_di_stampa.append(f'PSD features = [(0, 10), (10,20), (20, 30), (30, 40), (40, 50)]: {channel_list}')

    string = 'welch_alt'
    values, channel_list = selection_alg.forward_selection_eer(n_channels, dataset, tw, fs, string)

    vettori.append(values)
    stringhe_di_stampa.append(f'PSD features = [(0, 10), (20, 30), (40, 50)]: {channel_list}')

    # Trasponi la lista di vettori per ottenere una lista di colonne invece di righe
    vettori_trasposti = np.array(vettori).T

    print('\n'.join(stringhe_di_stampa))

    titolo = 'Valori di EER sovrapposti per Features estratte'
    etichette_personalizzate = ['[(0, 10)]', '[(0, 10), (10,20)]', '[(0, 10), (10,20), (20, 30)]',
                                '[(0, 10), (10,20), (20, 30), (30, 40), (40, 50)]', '[(0, 10), (20, 30), (40, 50)]']


    utils.plot_compare(values,vettori_trasposti,titolo,etichette_personalizzate)



def compare_FOOOF_vs_PSD(dataset, tw, fs):

    n_sbjs, n_channels, n_samples = dataset.shape

    vettori = []
    stringhe_di_stampa = []

    string = 'off'
    values, channel_list = selection_alg.forward_selection_eer(n_channels, dataset, tw, fs, string)

    vettori.append(values)
    stringhe_di_stampa.append(f'FOOOF features = EXP, OFF: {channel_list}')

    string = 'welch_2'
    values, channel_list = selection_alg.forward_selection_eer(n_channels, dataset, tw, fs, string)

    vettori.append(values)
    stringhe_di_stampa.append(f'PSD features = [(0,10), (10,20)]: {channel_list}')


    # Trasponi la lista di vettori per ottenere una lista di colonne invece di righe
    vettori_trasposti = np.array(vettori).T

    print('\n'.join(stringhe_di_stampa))

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

    values = [0.3322676896845695, 0.3094919786096257, 0.3084069596217934, 0.30235216616290783, 0.30265248391846855, 0.30183871967759435, 0.3043865767650934, 0.3081841432225064, 0.31413237231651553, 0.31498488723552664, 0.31853057428505, 0.3263969619468341, 0.3250019375339068, 0.33497054948461596]

    channel_list = [6, 12, 5, 13, 10, 14, 8, 11, 3, 2, 7, 9, 4, 1]

    vettori.append(values)
    stringhe_di_stampa.append(f'Forward Selection with [(0,10), (10,20)]: {channel_list}')

    values = [0.3322676896845695, 0.3094919786096257, 0.3084069596217934, 0.3062175463070604, 0.29745989304812837, 0.3040475083313958, 0.31461675579322634, 0.3140451832907076, 0.31498488723552664, 0.31301829032008055, 0.31853057428505, 0.3332073936293885, 0.3392524994187398, 0.33497054948461596]
    channel_list = [6, 12, 5, 8, 14, 3, 2, 11, 7, 13, 10, 1, 9, 4]

    vettori.append(values)
    stringhe_di_stampa.append(f'Fixed List with [(0,10), (10,20)]: {channel_list}')


    values = [0.3322676896845695, 0.34679919398589476, 0.3446485313492986, 0.34148066341160965, 0.3396303185305743, 0.33601681779431136, 0.3341180345656049, 0.3377218476323336, 0.34220723862667596, 0.33913624738432924, 0.33938812679221886, 0.33881655428970003, 0.3392524994187398, 0.33497054948461596]

    channel_list = [6, 1, 2, 5, 14, 8, 13, 3, 9, 11, 10, 7, 12, 4]

    vettori.append(values)
    stringhe_di_stampa.append(f'Fixed List with EXP OFF: {channel_list}')


    # Trasponi la lista di vettori per ottenere una lista di colonne invece di righe
    vettori_trasposti = np.array(vettori).T

    print('\n'.join(stringhe_di_stampa))

    titolo = 'Valori di EER sovrapposti per Features estratte'
    #titolo = 'Confronto prestazioni tra Forward e Backward Selection'
    #titolo = 'Confronto prestazioni tra Forward e Lista Fissata'
    etichette = ['Forward PSD','Fixed List PSD' , 'Fixed List FOOOF']

    utils.plot_compare(values,vettori_trasposti,titolo,etichette)


def compare_new_PSD_features(dataset, tw, fs):

    n_sbjs, n_channels, n_samples = dataset.shape

    vettori = []
    stringhe_di_stampa = []


    string = 'welch_2'
    values, channel_list = selection_alg.forward_selection_eer(n_channels, dataset, tw, fs, string)

    vettori.append(values)
    stringhe_di_stampa.append(f'PSD features = [(0, 10), (10,20)]: {channel_list}')

    string = 'welch_new'
    values, channel_list = selection_alg.forward_selection_eer(n_channels, dataset, tw, fs, string)

    vettori.append(values)
    stringhe_di_stampa.append(f'PSD features = [(1, 3), (4,7), (8, 12), (13,30) , (30, 50)]: {channel_list}')



    # Trasponi la lista di vettori per ottenere una lista di colonne invece di righe
    vettori_trasposti = np.array(vettori).T

    print('\n'.join(stringhe_di_stampa))

    titolo = 'Valori di EER sovrapposti per Features estratte'
    etichette_personalizzate = ['[(0, 10), (10,20)]','[(1, 3), (4,7), (8, 12), (13,30) , (30, 50)]']


    utils.plot_compare(values,vettori_trasposti,titolo,etichette_personalizzate)