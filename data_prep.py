import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


from our_data_operations import createFrames, createOurDataset
from our_datasets import LargeDataset

def split_csv_train_test(df: pd.DataFrame, test_ratio: float = 0.2, seed: int = 123):
    """
    Randomly selects a contiguous chunk of rows from df as test set.
    The rest is the training set.
    """
    random.seed(seed)
    n_rows = len(df)
    test_size = int(n_rows * test_ratio)
    
    start_idx = random.randint(0, n_rows - test_size)
    end_idx = start_idx + test_size
    
    test_df = df.iloc[start_idx:end_idx].reset_index(drop=True)
    train_df = pd.concat([df.iloc[:start_idx], df.iloc[end_idx:]]).reset_index(drop=True)
    
    return train_df, test_df

def split_contiguous_chunk(df: pd.DataFrame, val_ratio: float = 0.1, seed: int = 123):
    random.seed(seed)
    n_rows = len(df)
    val_size = int(n_rows * val_ratio)
    start_idx = random.randint(0, n_rows - val_size)
    end_idx = start_idx + val_size
    val_df = df.iloc[start_idx:end_idx].reset_index(drop=True)
    train_df = pd.concat([df.iloc[:start_idx], df.iloc[end_idx:]]).reset_index(drop=True)
    return train_df, val_df

def df_to_dataset_random_chunk(csv_filepaths,
                               windowSize=128,
                               coverage=64,
                               test_ratio=0.2,
                               val_ratio=0.1,
                               augment_data=True,
                               amplitude_threshold=1000,
                               flat_channel_threshold=1e-8,
                               nan_threshold=0.9,
                               verbose=True):
    """
    Loads multiple EEG CSV files, splits them into train/val/test contiguous chunks,
    performs data augmentation, applies noise filtering, and returns PyTorch Datasets.
    
    Args:
        csv_filepaths: list of CSV file paths containing EEG data
        windowSize: number of samples per window (frame)
        coverage: overlap between consecutive windows
        test_ratio: portion of each file for testing
        val_ratio: portion of training set used for validation
        augment_data: whether to apply augmentation to training frames
        amplitude_threshold: threshold for detecting broken signals (µV)
        flat_channel_threshold: variance threshold for dead channels
        nan_threshold: max ratio of NaN/Inf values allowed
        verbose: print filtering logs
        
    Returns:
        train_dataset, val_dataset, test_dataset : LargeDataset objects
    """

    train_frames, train_labels = [], []
    val_frames, val_labels = [], []
    test_frames, test_labels = [], []

    for csv_file in csv_filepaths:
        df = pd.read_csv(csv_file)
        train_val_df, test_df = split_csv_train_test(df, test_ratio=test_ratio)
        train_df, val_df = split_contiguous_chunk(train_val_df, val_ratio=val_ratio)

        # --- Create frames from dataframes ---
        t_frames, t_labels = createFrames(train_df, windowSize, coverage)
        v_frames, v_labels = createFrames(val_df, windowSize, coverage)
        test_f, test_l = createFrames(test_df, windowSize, coverage)

        train_frames.extend(t_frames)
        train_labels.extend(t_labels)
        val_frames.extend(v_frames)
        val_labels.extend(v_labels)
        test_frames.extend(test_f)
        test_labels.extend(test_l)

        # --- Apply augmentation ONLY on training frames ---
        if augment_data:
            aug_frames, aug_labels = [], []
            for frame, label in zip(t_frames, t_labels):
                # Gaussian noise
                noise = np.random.normal(0, 0.01, frame.shape)
                aug_frames.append(frame + noise)
                aug_labels.append(label)

                # Time shifting
                shift = np.random.randint(-5, 6)
                aug_frames.append(np.roll(frame, shift, axis=0))
                aug_labels.append(label)

                # Amplitude scaling
                scale = np.random.uniform(0.9, 1.1)
                aug_frames.append(frame * scale)
                aug_labels.append(label)

                # Masking
                mask_len = np.random.randint(5, 15)
                start = np.random.randint(0, windowSize - mask_len)
                frame_mask = frame.copy()
                frame_mask[start:start+mask_len, :] = 0
                aug_frames.append(frame_mask)
                aug_labels.append(label)

                # Segment shuffling
                num_chunks = 4
                chunk_size = windowSize // num_chunks
                chunks = [frame[i*chunk_size:(i+1)*chunk_size, :] for i in range(num_chunks)]
                np.random.shuffle(chunks)
                aug_frames.append(np.vstack(chunks))
                aug_labels.append(label)

                # Channel dropout
                frame_dropout = frame.copy()
                drop_channel = np.random.randint(0, frame.shape[1])
                frame_dropout[:, drop_channel] = 0
                aug_frames.append(frame_dropout)
                aug_labels.append(label)

            train_frames.extend(aug_frames)
            train_labels.extend(aug_labels)

    # --- Build datasets ---
    train_dataset = LargeDataset(train_frames, train_labels)
    val_dataset = LargeDataset(val_frames, val_labels)
    test_dataset = LargeDataset(test_frames, test_labels)

    # --- Log distributions ---
    print("\n--- Dataset Summary ---")
    print("Train class distribution:", dict(Counter(train_labels)))
    print("Validation class distribution:", dict(Counter(val_labels)))
    print("Test class distribution:", dict(Counter(test_labels)))

    return train_dataset, val_dataset, test_dataset

def data_loading(fif_files: list, csv_files: list, l_freq: float = 6.0, h_freq: float = 40.0, notch_filter: list[float] = [50]):
    temp_dfs = []
    helper = []
    for i in range(len(fif_files)):
        helper.append(preprocess_eeg_data(dir_path = fif_files[i], l_freq = l_freq, h_freq = h_freq, notch_filter = notch_filter))
        fif_to_csv(helper[i], csv_files[i])
        df = pd.read_csv(csv_files[i])
        temp_dfs.append(df)
    
    data = pd.DataFrame()
    for df in temp_dfs:
        data = pd.concat([data, df], axis = 0, ignore_index=True)

    return data

def csv_loading(csv_files: list):
    temp_dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        temp_dfs.append(df)
    
    data = pd.DataFrame()
    for df in temp_dfs:
        data = pd.concat([data, df], axis=0, ignore_index=True)
    
    return data

def df_to_dataset(df: pd.DataFrame):
    '''
    Convert df to our dataset in order to use it later in pytorch
    '''
    
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]

    full_train_dataset = createOurDataset(train_df, 128, 108)
    full_test_dataset = createOurDataset(test_df, 128, 108)

    return full_train_dataset, full_test_dataset


import mne
from collections import Counter

def fif_to_csv(raw, csv_file):
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
    df = df.drop(columns=['Sample'], inplace=False)


    # Zapisz DataFrame do pliku .csv
    df.to_csv(csv_file, index=False)

def preprocess_eeg_data(
    dir_path: str,
    l_freq: float = 6.0,
    h_freq: float = 40.0,
    notch_filter: list[float] = [50]
) -> mne.io.Raw:
    """
    Preprocess EEG data for each block.
    
    Parameters:
    - dir_path (str): Directory path containing the EEG data file.
    - l_freq (float): Low cutoff frequency for band-pass filtering.
    - h_freq (float): High cutoff frequency for band-pass filtering.
    - notch_filter (list[float]): Frequencies for notch filtering.
    
    Returns:
    - raw (mne.io.Raw): Preprocessed raw EEG data.
    """
    raw = _load_data(dir_path)
    raw = _preprocess_raw_data(raw, l_freq, h_freq, notch_filter)
    
    return raw

def _load_data(file_path: str) -> mne.io.Raw:
    """
    Load raw EEG data.
    
    Parameters:
    - file_path (str): Path to the EEG data file.
    
    Returns:
    - mne.io.Raw: Loaded raw EEG data.
    """
    raw = mne.io.read_raw_fif(file_path)
    raw = raw.load_data()
    
    return raw

def _preprocess_raw_data(
    raw: mne.io.Raw, l_freq: float, h_freq: float, notch_filter: float
) -> mne.io.Raw:
    """
    Preprocess raw EEG data by applying band-pass and notch filters.
    
    Parameters:
    - raw (mne.io.Raw): Raw EEG data.
    - l_freq (float): Low cutoff frequency for band-pass filtering.
    - h_freq (float): High cutoff frequency for band-pass filtering.
    - notch_filter (float): Frequency for notch filtering.
    
    Returns:
    - mne.io.Raw: Preprocessed EEG data.
    """
    raw = raw.pick_types(eeg=True, stim=False, eog=False, exclude="bads") 
    raw.apply_function(lambda x: x * 10 ** -6)
    raw.filter(l_freq=l_freq, h_freq=h_freq) 
    raw.notch_filter(notch_filter)

    return raw

def decimation(df: pd.DataFrame, decimation_factor: int):
    df_decimated = df.copy(deep=True)
    df_decimated = df_decimated.iloc[::decimation_factor, :].reset_index(drop=True)

    return df_decimated

def sliding_window_averaging(df: pd.DataFrame, window_size: int):
    df_windowed = df.copy(deep=True)

    numeric_data = df.select_dtypes(include=[np.number])
    annotations = df.select_dtypes(exclude=[np.number])

    averaged_eeg = numeric_data.groupby(np.arange(len(numeric_data)) // window_size).mean()
    selected_annotations = annotations.groupby(np.arange(len(annotations)) // window_size).first()

    df_windowed = pd.concat([averaged_eeg.reset_index(drop=True), selected_annotations.reset_index(drop=True)], axis = 1)
    cols = df_windowed.columns.tolist()
    cols.insert(1, cols[-1])
    cols = cols[:-1]
    df_windowed = df_windowed[cols]

    return df_windowed