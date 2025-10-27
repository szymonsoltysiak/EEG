import os

data_path = '../datasets/mind_drive'

eeg_top_folders = os.listdir(data_path)
eeg_filepaths = []
for folder in eeg_top_folders:
    local_eeg_filepaths = os.listdir(os.path.join(data_path, folder))
    full_eeg_filepaths = [
        os.path.join(data_path, folder, file) 
        for file in local_eeg_filepaths 
        if file.endswith('.fif') and 'practice' not in file.lower()
    ]
    eeg_filepaths.extend(full_eeg_filepaths)

csv_folder = '../datasets/mind_drive/csv_files'
if not os.path.exists(csv_folder):
    os.makedirs(csv_folder)

csv_filepaths = [os.path.join(csv_folder, os.path.basename(f).replace('.fif', '.csv')) for f in eeg_filepaths]
eeg_filepaths, csv_filepaths