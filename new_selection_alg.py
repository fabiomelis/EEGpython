import core
import matplotlib.pyplot as plt


def apply_selection(algorithm, n_channels, dataset, tw, fs):

    if algorithm == 'forward':
        selected_channels , eer = forward_selection(n_channels, dataset, tw, fs)
    elif algorithm == 'backward':
        selected_channels , eer = backward_selection(n_channels, dataset, tw, fs)
    else:
        print("Errore nella scelta dell'algoritmo! Controlla i dati in ingresso.")

    return selected_channels, eer



def forward_selection(n_channels, dataset, tw, fs):

    EER_values = []
    channel_history = []

    selected_channels = []
    remaining_channels = list(range(1, n_channels + 1))

    while remaining_channels:
        best_channel, eer = find_best_channel(selected_channels, remaining_channels, dataset, tw, fs)

        EER_values.append(eer)
        channel_history.append(selected_channels.copy())

        remaining_channels.remove(best_channel)
        selected_channels.append(best_channel)

    plt.plot(channel_history, EER_values, marker='o')
    plt.xlabel('Canali Selezionati')
    plt.ylabel('Valore EER')
    plt.title('EER vs Canali Selezionati')
    plt.show()





def backward_selection(n_channels, dataset, tw, fs):


    selected_channels = list(range(1, n_channels + 1))

    while len(selected_channels) > 1:
        worst_channel, eer = find_worst_channel(selected_channels, dataset, tw, fs)

        if eer < best_eer:

            best_eer = eer

        else:
            break

        selected_channels.remove(worst_channel)

        print(f'Canali rimanenti: {selected_channels}')


    return selected_channels, eer



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

            best_channel = new_channel
            best_eer = EER


    print(f'Miglior canale selezionato: {best_channel} con EER: {best_eer:.3f} e AUC: {best_auc:.3f}')

    return best_channel, best_eer



def find_worst_channel(selected_channels, dataset, tw, fs):
    worst_channel = None
    best_eer = float('inf')


    for channel_to_remove in selected_channels:
        current_channels = []
        for channel in selected_channels:
            if channel != channel_to_remove:
                current_channels.append(channel)

        print('Canali analizzati: ', current_channels)

        EER, AUC = core.compute_EER_AUC(dataset, tw, fs, current_channels)

        print('EER e AUC: ', EER, AUC)

        if EER < best_eer:
            worst_channel = channel_to_remove
            best_eer = EER

    print(f'Peggiore canale selezionato: {worst_channel} con EER: {best_eer:.3f} e AUC: {best_auc:.3f}')

    return worst_channel, best_eer
