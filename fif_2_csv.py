import mne
import pandas as pd
import numpy as np


ddef fif_to_csv(fif_file, csv_file):
    # Wczytaj dane z pliku .fif
    raw = mne.io.read_raw_fif(fif_file, preload=True)

    # Pobierz dane i czasy
    data, times = raw[:, :]  # Wczytuje wszystkie dane i czasy
    sfreq = raw.info['sfreq']
    # Pobierz anotacje
    annotations = raw.annotations
    channel_names = raw.info['ch_names']
    # Przygotuj dane do zapisania
    rows = []


    # Inicjalizuj zmienną do trzymania bieżącej anotacji
    current_annotation = None
    start_time = 0
    # Przechodzimy przez wszystkie anotacje
    number_of_sample = 0
    for onset, description in zip(annotations.onset, annotations.description):



        if description != 'END':
            current_annotation = description
            start_time = onset
        if description == 'END':
            number_of_sample += 1
            end_time = onset

            for i, time_point in enumerate(times):
                if start_time <= time_point < end_time:
                    row = {'Time': time_point, 'Sample' : number_of_sample, 'Annotation': current_annotation}
                    sample = int(time_point * sfreq)
                    for j, channel_data in enumerate(data):
                        row[channel_names[j]] = channel_data[sample]
                    rows.append(row)

    # Stwórz DataFrame
    df = pd.DataFrame(rows)
    df = df.drop(columns=['Accel_x', 'Accel_y', 'Accel_z', 'Digital', 'Sample'], inplace=False)


    # Zapisz DataFrame do pliku .csv
    df.to_csv(csv_file, index=False)