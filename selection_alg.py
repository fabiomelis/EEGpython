import core
import matplotlib.pyplot as plt
import utils

'''
def apply_selection(algorithm, n_channels, dataset, tw, fs):

    if algorithm == 'forward':
        selected_channels , eer = forward_selection(n_channels, dataset, tw, fs, string)
    elif algorithm == 'backward':
        selected_channels , eer = backward_selection(n_channels, dataset, tw, fs, string)
    else:
        print("Errore nella scelta dell'algoritmo! Controlla i dati in ingresso.")

    return selected_channels, eer
'''


def forward_selection_eer(n_channels, dataset, tw, fs, string):

    def find_best_channel_eer():
        best_channel = None
        best_eer = float('inf')
        best_auc = float('inf')

        for new_channel in remaining_channels:
            combined_channels = selected_channels + [new_channel]

            print('Canali analizzati: ', combined_channels, )

            if string == 'exp':
                EER, AUC = core.compute_EER_AUC_FOOOF(dataset, tw, fs, combined_channels,1, string)
            elif string == 'off':
                EER, AUC = core.compute_EER_AUC_FOOOF(dataset, tw, fs, combined_channels,2,string)
            elif string == 'freq':
                EER, AUC = core.compute_EER_AUC_FOOOF(dataset, tw, fs, combined_channels,3,string)
            elif string == 'welch_4':
                EER, AUC = core.compute_EER_AUC_welch(dataset, tw, fs, combined_channels,4,[(1, 3), (4,7), (8, 12), (13,30)])
            elif string == 'welch_5':
                EER, AUC = core.compute_EER_AUC_welch(dataset, tw, fs, combined_channels,5,[(1, 3), (4,7), (8, 12), (13,30) , (30, 50)])



            print('EER e AUC: ', EER, AUC)

            if EER < best_eer:
                best_channel = new_channel
                best_eer = EER

        print(f'Miglior canale selezionato: {best_channel} con EER: {best_eer:.4f} e AUC: {best_auc:.4f}')

        return best_channel, best_eer

    EER_values = []
    channels_history = ""

    selected_channels = []
    remaining_channels = list(range(1, n_channels + 1))

    while remaining_channels:
        best_channel, eer = find_best_channel_eer()

        EER_values.append(eer)

        remaining_channels.remove(best_channel)
        selected_channels.append(best_channel)

        channels_str = f"{len(selected_channels)} canali: {', '.join(map(str, selected_channels))}, EER: {eer:.4f}\n"
        channels_history += channels_str

        print(channels_str)

    print(channels_history)
    channel_list = ', '.join(map(str, selected_channels))

    utils.plot_EER(EER_values)

    print(EER_values)
    print(channel_list)

    return EER_values, channel_list



def forward_selection_auc(n_channels, dataset, tw, fs, string):

    def find_best_channel_auc():

        best_channel = None
        best_eer = float('inf')
        best_auc = float('inf')

        for new_channel in remaining_channels:
            combined_channels = selected_channels + [new_channel]

            print('Canali analizzati: ', combined_channels, )

            if string == 'exp':
                EER, AUC = core.compute_EER_AUC_FOOOF(dataset, tw, fs, combined_channels,1, string)
            elif string == 'off':
                EER, AUC = core.compute_EER_AUC_FOOOF(dataset, tw, fs, combined_channels,2,string)
            elif string == 'freq':
                EER, AUC = core.compute_EER_AUC_FOOOF(dataset, tw, fs, combined_channels,3,string)
            elif string == 'welch_4':
                EER, AUC = core.compute_EER_AUC_welch(dataset, tw, fs, combined_channels,4,[(1, 3), (4,7), (8, 12), (13,30)])
            elif string == 'welch_5':
                EER, AUC = core.compute_EER_AUC_welch(dataset, tw, fs, combined_channels,5,[(1, 3), (4,7), (8, 12), (13,30) , (30, 50)])


            print('EER e AUC: ', EER, AUC)

            if AUC < best_auc:
                best_channel = new_channel
                best_auc = AUC

        print(f'Miglior canale selezionato: {best_channel} con EER: {best_eer:.4f} e AUC: {best_auc:.4f}')

        return best_channel, best_auc

    AUC_values = []
    channels_history = ""

    selected_channels = []
    remaining_channels = list(range(1, n_channels + 1))

    while remaining_channels:
        best_channel, auc = find_best_channel_auc()

        AUC_values.append(auc)

        remaining_channels.remove(best_channel)
        selected_channels.append(best_channel)

        channels_str = f"{len(selected_channels)} canali: {', '.join(map(str, selected_channels))}, AUC: {auc:.4f}\n"
        channels_history += channels_str

        print(channels_str)

    print(channels_history)
    channel_list = ', '.join(map(str, selected_channels))

    utils.plot_AUC(AUC_values)

    print(AUC_values)
    print(channel_list)

    return AUC_values, channel_list


def backward_selection_eer(n_channels, dataset, tw, fs, string):

    EER_values = []
    channels_history = ""
    removed_channels = []

    selected_channels = list(range(1, n_channels + 1))

    if string == 'exp':
        EER, AUC = core.compute_EER_AUC_FOOOF(dataset, tw, fs, selected_channels, 1, string)
    elif string == 'off':
        EER, AUC = core.compute_EER_AUC_FOOOF(dataset, tw, fs, selected_channels, 2, string)
    elif string == 'freq':
        EER, AUC = core.compute_EER_AUC_FOOOF(dataset, tw, fs, selected_channels, 3, string)
    elif string == 'welch_4':
        EER, AUC = core.compute_EER_AUC_welch(dataset, tw, fs, selected_channels,4,[(1, 3), (4,7), (8, 12), (13,30)])
    elif string == 'welch_5':
        EER, AUC = core.compute_EER_AUC_welch(dataset, tw, fs, selected_channels,5,[(1, 3), (4,7), (8, 12), (13,30), (30,50)])

    channels_str = f"{len(selected_channels)} canali: {', '.join(map(str, selected_channels))}, AUC: {EER:.4f}\n"
    channels_history += channels_str

    print(channels_str)

    EER_values.append(EER)

    def find_worst_channel_eer():
        worst_channel = None
        best_eer = float('inf')
        best_auc = float('inf')

        for channel_to_remove in selected_channels:
            current_channels = []
            for channel in selected_channels:
                if channel != channel_to_remove:
                    current_channels.append(channel)

            print('Canali analizzati: ', current_channels)

            if string == 'exp':
                EER, AUC = core.compute_EER_AUC_FOOOF(dataset, tw, fs, current_channels, 1, string)
            elif string == 'off':
                EER, AUC = core.compute_EER_AUC_FOOOF(dataset, tw, fs, current_channels, 2, string)
            elif string == 'freq':
                EER, AUC = core.compute_EER_AUC_FOOOF(dataset, tw, fs, current_channels, 3, string)
            elif string == 'welch_4':
                EER, AUC = core.compute_EER_AUC_welch(dataset, tw, fs, current_channels, 4, [(1, 3), (4, 7), (8, 12), (13, 30)])
            elif string == 'welch_5':
                EER, AUC = core.compute_EER_AUC_welch(dataset, tw, fs, current_channels, 5, [(1, 3), (4, 7), (8, 12), (13, 30), (30, 50)])

            print('EER e AUC: ', EER, AUC)

            if EER < best_eer:
                worst_channel = channel_to_remove
                best_eer = EER

        print(f'Peggiore canale selezionato: {worst_channel} con EER: {best_eer:.4f} e AUC: {best_auc:.4f}')

        return worst_channel, best_eer

    while len(selected_channels) > 1:

        worst_channel, eer = find_worst_channel_eer()

        selected_channels.remove(worst_channel)
        EER_values.append(eer)
        removed_channels.append(worst_channel)

        print(f'Canali rimanenti: {selected_channels}')

        channels_str = f"{len(selected_channels)} canali: {', '.join(map(str, selected_channels))}, EER: {eer:.4f}\n"
        channels_history += channels_str

        print(channels_str)

    print(channels_history)

    removed_channels.append(selected_channels[0])

    print(f'Removed channels: {", ".join(map(str, removed_channels))}')

    utils.plot_EER(EER_values[::-1])

    print(EER_values[::-1])
    print(removed_channels[::-1])

    return EER_values[::-1], removed_channels[::-1]



def backward_selection_auc(n_channels, dataset, tw, fs, string):
    AUC_values = []
    channels_history = ""
    removed_channels = []

    selected_channels = list(range(1, n_channels + 1))

    if string == 'exp':
        EER, AUC = core.compute_EER_AUC_FOOOF(dataset, tw, fs, selected_channels, 1, string)
    elif string == 'off':
        EER, AUC = core.compute_EER_AUC_FOOOF(dataset, tw, fs, selected_channels, 2, string)
    elif string == 'freq':
        EER, AUC = core.compute_EER_AUC_FOOOF(dataset, tw, fs, selected_channels, 3, string)
    elif string == 'welch_4':
        EER, AUC = core.compute_EER_AUC_welch(dataset, tw, fs, selected_channels,4,[(1, 3), (4,7), (8, 12), (13,30)])
    elif string == 'welch_5':
        EER, AUC = core.compute_EER_AUC_welch(dataset, tw, fs, selected_channels,5,[(1, 3), (4,7), (8, 12), (13,30), (30,50)])

    channels_str = f"{len(selected_channels)} canali: {', '.join(map(str, selected_channels))}, AUC: {AUC:.4f}\n"
    channels_history += channels_str

    print(channels_str)

    AUC_values.append(AUC)

    def find_worst_channel_auc():
        worst_channel = None
        best_eer = float('inf')
        best_auc = float('inf')

        for channel_to_remove in selected_channels:
            current_channels = []
            for channel in selected_channels:
                if channel != channel_to_remove:
                    current_channels.append(channel)

            print('Canali analizzati: ', current_channels)

            if string == 'exp':
                EER, AUC = core.compute_EER_AUC_FOOOF(dataset, tw, fs, current_channels, 1, string)
            elif string == 'off':
                EER, AUC = core.compute_EER_AUC_FOOOF(dataset, tw, fs, current_channels, 2, string)
            elif string == 'freq':
                EER, AUC = core.compute_EER_AUC_FOOOF(dataset, tw, fs, current_channels, 3, string)
            elif string == 'welch_4':
                EER, AUC = core.compute_EER_AUC_welch(dataset, tw, fs, current_channels, 4,[(1, 3), (4, 7), (8, 12), (13, 30)])
            elif string == 'welch_5':
                EER, AUC = core.compute_EER_AUC_welch(dataset, tw, fs, current_channels, 5, [(1, 3), (4, 7), (8, 12), (13, 30), (30, 50)])

            print('EER e AUC: ', EER, AUC)

            if AUC < best_auc:
                worst_channel = channel_to_remove
                best_auc = AUC

        print(f'Peggiore canale selezionato: {worst_channel} con EER: {best_eer:.4f} e AUC: {best_auc:.4f}')

        return worst_channel, best_auc

    while len(selected_channels) > 1:
        worst_channel, auc = find_worst_channel_auc()

        selected_channels.remove(worst_channel)
        AUC_values.append(auc)
        removed_channels.append(worst_channel)

        print(f'Canali rimanenti: {selected_channels}')

        channels_str = f"{len(selected_channels)} canali: {', '.join(map(str, selected_channels))}, AUC: {auc:.4f}\n"
        channels_history += channels_str

        print(channels_str)

    print(channels_history)

    removed_channels.append(selected_channels[0])

    channel_list = ', '.join(map(str, reversed(removed_channels)))

    print(f'Removed channels: {channel_list}')

    #utils.plot_AUC(AUC_values[::-1])

    return AUC_values[::-1], channel_list

