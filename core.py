import numpy as np
import utils
import performance
import fooof_features
import welch




def compute_features_matrix(dataset, tw, fs, selected_channels):

    n_sbjs, n_channels, n_samples = dataset.shape

    n_features = 20

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

                #exp, off, indices = fooof_features.comp_aperiodic(tmp_data, i_sbj, i_channel, n_ch_sel, fs, n_features)

                #exp, off, freq, indices = fooof_features.comp_aperiodic_and_1st_freq(tmp_data, i_sbj, i_channel, n_ch_sel, fs, n_features)

                #exp, indices = fooof_features.comp_aperiodic_exp(tmp_data,i_sbj,i_channel,n_ch_sel,fs,n_features)

                #pw1, pw2, indices = fooof_features.comp_peak_1st_2nd_pw(tmp_data,i_sbj,i_channel,n_ch_sel,fs,n_features)


                psd_values, indices = welch.extract_psd_features(tmp_data,i_sbj,i_channel,n_ch_sel,fs,n_features)

                features_vector[indices] = psd_values
                #features_vector[indices[0]] = exp
                #features_vector[indices[2]] = freq



                n_ch_sel = n_ch_sel + 1

            # Ora inserisco il vettore delle features per ogni epoca di ogni soggetto all'interno della matrice delle features
            ps[i_sbj - 1, i_epoch - 1, :] = features_vector

    ps = performance.reorg(ps)

    return ps, v_identity, n_epochs



def compute_total_performance(dataset, tw_range, fs, selected_channels):

    n_sbjs, n_channels, n_samples = dataset.shape




    n_cases = len(tw_range)
    EERtot = np.zeros(n_cases)
    AUCtot = np.zeros(n_cases)
    score_distanza_tot = []
    flag_tot = []

    i_tw = 0
    for tw in tw_range:
        i_tw += 1

        ps, v_identity, n_epochs = compute_features_matrix(dataset, tw, fs, selected_channels)

        #print('Calcolo score')
        score_distanza , flag  = performance.calcolo_score(ps, v_identity, n_epochs, n_sbjs)
        print('Vettore degli score: ', score_distanza)

        #print('Calcolo FAR e FRR')
        FAR, FRR, vettore_soglia = performance.calcolo_FAR_FRR(score_distanza, flag)
        #print(FAR, FRR)

        #print('Calcolo EER e AUC')
        EER, AUC = performance.calcolo_EER_AUC(FAR, FRR)

        print('EER e AUC: ', EER, AUC)

        #utils.plot_ROC_curve(FAR, FRR, AUC)

        EERtot[i_tw - 1] = EER
        AUCtot[i_tw - 1] = AUC
        score_distanza_tot.append(score_distanza)
        flag_tot.append(flag)

    # Plot EERtot e AUCtot
    utils.print_EER(tw_range, EERtot)
    utils.print_AUC(tw_range, AUCtot)

    # Calcola la media di AUCtot e EERtot
    mean_eer_tot = np.mean(EERtot)
    mean_auc_tot = np.mean(AUCtot)

    # Stampa i risultati
    print(f"Mean EERtot: {mean_eer_tot}")
    print(f"Mean AUCtot: {mean_auc_tot}")




def compute_EER_AUC (dataset, tw, fs, selected_channels):

    n_sbjs, n_channels, n_samples = dataset.shape

    ps, v_identity, n_epochs = compute_features_matrix(dataset, tw, fs, selected_channels)

    # print('Calcolo score')
    score_distanza, flag = performance.calcolo_score(ps, v_identity, n_epochs, n_sbjs)
    #print('Vettore degli score: ', score_distanza)

    # print('Calcolo FAR e FRR')
    FAR, FRR, vettore_soglia = performance.calcolo_FAR_FRR(score_distanza, flag)
    # print(FAR, FRR)

    # print('Calcolo EER e AUC')
    EER, AUC = performance.calcolo_EER_AUC(FAR, FRR)

    return EER, AUC



