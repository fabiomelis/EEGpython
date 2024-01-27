import core


def apply_selection(algorithm, n_channels, dataset, tw, fs, EER_required, AUC_required, channel_limit):

    if algorithm == 'forward':
        selected_channels , eer, auc = forward_selection(n_channels, dataset, tw, fs, EER_required, AUC_required, channel_limit)
    elif algorithm == 'backward':
        selected_channels , eer, auc = backward_selection(n_channels, dataset, tw, fs, EER_required, AUC_required, channel_limit)
    else:
        print("Errore nella scelta dell'algoritmo! Controlla i dati in ingresso.")

    return selected_channels, eer, auc



def forward_selection(n_channels, dataset, tw, fs, EER_required, AUC_required, channel_limit):


    best_eer = float('inf')
    best_auc = float('inf')

    selected_channels = []
    remaining_channels = list(range(1, n_channels + 1))

    while remaining_channels:
        best_channel, eer, auc = find_best_channel(selected_channels, remaining_channels, dataset, tw, fs)

        #if eer < best_eer:
        if eer < best_eer or auc < best_auc:
            best_eer = eer
            best_auc = auc
        else:
            break

        remaining_channels.remove(best_channel)
        selected_channels.append(best_channel)

        if (eer < EER_required or auc < AUC_required) or len(selected_channels) >= channel_limit:
            break

    return selected_channels, eer, auc



def backward_selection(n_channels, dataset, tw, fs, EER_required, AUC_required, channel_limit):


    #best_eer = float('inf')
    #best_auc = float('inf')

    selected_channels = list(range(1, n_channels + 1))

    while len(selected_channels) > 1:
        worst_channel, eer, auc = find_worst_channel(selected_channels, dataset, tw, fs)

        if eer < best_eer:
        #if eer < best_eer or auc < best_auc:
            best_eer = eer
            best_auc = auc
        else:
            break

        selected_channels.remove(worst_channel)

        print(f'Canali rimanenti: {selected_channels}')

        if (eer > EER_required or auc > AUC_required) or len(selected_channels) <= channel_limit:
            break

    return selected_channels, eer, auc



def find_best_channel(selected_channels, remaining_channels, dataset, tw, fs):
    best_channel = None
    best_eer = float('inf')
    best_auc = float('inf')

    for new_channel in remaining_channels:
        combined_channels = selected_channels + [new_channel]

        print('Canali analizzati: ', combined_channels, )

        EER, AUC = core.compute_EER_AUC(dataset, tw, fs, combined_channels)

        print('EER e AUC: ', EER, AUC)

        if EER < best_eer:
        #if EER < best_eer and AUC < best_auc:
            best_channel = new_channel
            best_eer = EER
            best_auc = AUC

    print(f'Miglior canale selezionato: {best_channel} con EER: {best_eer:.3f} e AUC: {best_auc:.3f}')

    return best_channel, best_eer, best_auc



def find_worst_channel(selected_channels, dataset, tw, fs):
    worst_channel = None
    best_eer = float('inf')
    best_auc = float('inf')

    for channel_to_remove in selected_channels:
        current_channels = []
        for channel in selected_channels:
            if channel != channel_to_remove:
                current_channels.append(channel)

        print('Canali analizzati: ', current_channels)

        EER, AUC = core.compute_EER_AUC(dataset, tw, fs, current_channels)

        print('EER e AUC: ', EER, AUC)

        if EER < best_eer or AUC < best_auc:
            worst_channel = channel_to_remove
            best_eer = EER
            best_auc = AUC

    print(f'Peggiore canale selezionato: {worst_channel} con EER: {best_eer:.3f} e AUC: {best_auc:.3f}')

    return worst_channel, best_eer, best_auc
