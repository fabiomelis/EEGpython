import numpy as np
import utils
import performance
import fooof_features
import welch
import selection_alg
import matplotlib.pyplot as plt



def compute_features_welch(dataset, tw, fs, selected_channels, n_features, freq_intervals):

    n_sbjs, n_channels, n_samples = dataset.shape

    #n_sbjs = 10

    print(f'Processing window length {tw}')

    n_epochs = int(np.floor(n_samples / fs / tw))

    s_step = int(np.floor(n_samples / n_epochs))

    # Ad ogni epoca di ogni soggetto corrisponde un vettore di features composto da n_features per ogni canale
    features_lenght = len(selected_channels) * n_features

    # Definiamo la matrice contenente le features
    ps = np.zeros((n_sbjs, n_epochs, features_lenght))

    v_identity = np.ceil(np.arange(1, n_epochs * n_sbjs + 1) / n_epochs)
    print(f'Processing {n_epochs} epochs for {n_sbjs} subjects')


    for i_epoch in range(1, n_epochs + 1):
        ini_s = 1 + s_step * (i_epoch - 1)
        end_s = ini_s + s_step - 1
        tmp_data = dataset[:, :, ini_s:end_s]
        #print(f'Epoch {i_epoch} : Processing {n_sbjs} subjects')

        for i_sbj in range(1, n_sbjs + 1):

            features_vector = np.zeros(features_lenght)

            n_ch_sel = 1

            # for i_channel in range(1, n_channels + 1):
            for i_channel in selected_channels:

                psd_values, indices = welch.extract_area_from_intervals(tmp_data,i_sbj,i_channel,n_ch_sel,fs,n_features,freq_intervals)

                features_vector[indices] = psd_values

                n_ch_sel = n_ch_sel + 1

            # Ora inserisco il vettore delle features per ogni epoca di ogni soggetto all'interno della matrice delle features
            ps[i_sbj - 1, i_epoch - 1, :] = features_vector
            #print(f'sbj{i_sbj}, epoch{i_epoch}, features_vector: {features_vector}')

    ps = performance.reorg(ps)

    return ps, v_identity, n_epochs



def compute_features_FOOOF(dataset, tw, fs, selected_channels, n_features, string):

    n_sbjs, n_channels, n_samples = dataset.shape

    #n_sbjs = 10

    print(f'Processing window length {tw}')

    n_epochs = int(np.floor(n_samples / fs / tw))

    s_step = int(np.floor(n_samples / n_epochs))

    # Ad ogni epoca di ogni soggetto corrisponde un vettore di features composto da n_features per ogni canale
    features_lenght = len(selected_channels) * n_features

    # Definiamo la matrice contenente le features
    ps = np.zeros((n_sbjs, n_epochs, features_lenght))

    v_identity = np.ceil(np.arange(1, n_epochs * n_sbjs + 1) / n_epochs)
    print(f'Processing {n_epochs} epochs for {n_sbjs} subjects')


    for i_epoch in range(1, n_epochs + 1):
        ini_s = 1 + s_step * (i_epoch - 1)
        end_s = ini_s + s_step - 1
        tmp_data = dataset[:, :, ini_s:end_s]
        #print(f'Epoch {i_epoch} : Processing {n_sbjs} subjects')

        for i_sbj in range(1, n_sbjs + 1):

            features_vector = np.zeros(features_lenght)

            n_ch_sel = 1

            # for i_channel in range(1, n_channels + 1):
            for i_channel in selected_channels:

                if string == 'exp':
                    exp, indices = fooof_features.comp_aperiodic_exp(tmp_data,i_sbj,i_channel,n_ch_sel,fs,n_features)
                    features_vector[indices[0]] = exp
                elif string == 'off':
                    exp, off, indices = fooof_features.comp_aperiodic(tmp_data, i_sbj, i_channel, n_ch_sel, fs,n_features)
                    features_vector[indices[0]] = exp
                    features_vector[indices[1]] = off
                elif string == 'freq':
                    exp, off, freq, indices = fooof_features.comp_aperiodic_and_1st_freq(tmp_data, i_sbj, i_channel,n_ch_sel, fs, n_features)
                    features_vector[indices[0]] = exp
                    features_vector[indices[1]] = off
                    features_vector[indices[2]] = freq

                n_ch_sel = n_ch_sel + 1

            # Ora inserisco il vettore delle features per ogni epoca di ogni soggetto all'interno della matrice delle features
            ps[i_sbj - 1, i_epoch - 1, :] = features_vector

    ps = performance.reorg(ps)

    return ps, v_identity, n_epochs






def compute_EER_AUC_welch (dataset, tw, fs, selected_channels, n_features, freq_intervals):

    n_sbjs, n_channels, n_samples = dataset.shape

    #n_sbjs = 10

    ps, v_identity, n_epochs = compute_features_welch(dataset, tw, fs, selected_channels,n_features,freq_intervals)

    # print('Calcolo score')
    score_distanza, flag = performance.calcolo_score(ps, v_identity, n_epochs, n_sbjs)
    #print('Vettore degli score: ', score_distanza)

    # print('Calcolo FAR e FRR')
    FAR, FRR, vettore_soglia = performance.calcolo_FAR_FRR(score_distanza, flag)
    # print(FAR, FRR)

    #utils.far_frr(FAR,FRR,vettore_soglia)

    # print('Calcolo EER e AUC')
    EER, AUC = performance.calcolo_EER_AUC(FAR, FRR)

    return EER, AUC


def compute_EER_AUC_FOOOF(dataset, tw, fs, selected_channels, n_features, string):

    n_sbjs, n_channels, n_samples = dataset.shape

    #n_sbjs = 10

    ps, v_identity, n_epochs = compute_features_FOOOF(dataset, tw, fs, selected_channels, n_features, string)

    # print('Calcolo score')
    score_distanza, flag = performance.calcolo_score(ps, v_identity, n_epochs, n_sbjs)
    #print('Vettore degli score: ', score_distanza)

    # print('Calcolo FAR e FRR')
    FAR, FRR, vettore_soglia = performance.calcolo_FAR_FRR(score_distanza, flag)
    # print(FAR, FRR)

    # print('Calcolo EER e AUC')
    EER, AUC = performance.calcolo_EER_AUC(FAR, FRR)

    return EER, AUC


def compute_EER_fixed_list(list_channels, dataset, tw, fs, string):

    EER_values = []
    #channels_history = ""

    selected_channels = []

    for current_channel in list_channels:

        analyzed_channels = selected_channels + [current_channel]

        if string == 'exp':
            EER, AUC = compute_EER_AUC_FOOOF(dataset, tw, fs, analyzed_channels, 1, string)
        elif string == 'off':
            EER, AUC = compute_EER_AUC_FOOOF(dataset, tw, fs, analyzed_channels, 2, string)
        elif string == 'freq':
            EER, AUC = compute_EER_AUC_FOOOF(dataset, tw, fs, analyzed_channels, 3, string)
        elif string == 'welch_4':
            EER, AUC = compute_EER_AUC_welch(dataset, tw, fs, analyzed_channels, 4,[(1, 3), (4, 7), (8, 12), (13, 30)])
        elif string == 'welch_5':
            EER, AUC = compute_EER_AUC_welch(dataset, tw, fs, analyzed_channels, 5, [(1, 3), (4, 7), (8, 12), (13, 30), (30, 50)])

        EER_values.append(EER)

        #list_channels.remove(current_channel)
        selected_channels.append(current_channel)

        print(f'analyzed ch: {analyzed_channels}')


        #channels_str = f"{len(selected_channels)} canali: {', '.join(map(str, selected_channels))}, EER: {eer:.4f}\n"
        #channels_history += channels_str

        #print(channels_str)

    #print(channels_history)
    channel_list = ', '.join(map(str, selected_channels))

    print(f'channel list: {channel_list}')

    #utils.plot_EER_forward(EER_values)

    return EER_values, channel_list
