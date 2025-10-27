from collections import Counter

import pandas as pd
from torch.utils.data import Dataset

from our_datasets import LargeDataset


def createFrames(df: pd.DataFrame, windowSize: int, coverage: int ):
    coverage = min(windowSize, coverage)
    moveSize = windowSize - coverage
    labelMapping = {'tongue': 0, 'foot': 1, 'right': 2, 'left': 3}
    df['label'] = df['label'].replace(labelMapping)
    i = 0
    labels = df["label"].values.tolist()
    eegColumns = ['EEG-Fz', 'EEG-0', 'EEG-1', 'EEG-2', 'EEG-3', 'EEG-4', 'EEG-5', 'EEG-C3', 'EEG-6', 'EEG-Cz', 'EEG-7',
                  'EEG-C4', 'EEG-8', 'EEG-9', 'EEG-10', 'EEG-11', 'EEG-12', 'EEG-13', 'EEG-14', 'EEG-Pz', 'EEG-15',
                  'EEG-16']

    eegData = df[eegColumns].values.tolist()
    resultFrames = []
    resultLabels = []
    tmp = []
    print(len(eegData))
    while i + windowSize < len(eegData):
        tmp = (eegData[i:i + windowSize])
        tmpLabels = labels[i:i + windowSize]
        i += moveSize
        counter = Counter(tmpLabels)
        frameLabel = counter.most_common(1)[0][0]
        resultFrames.append(tmp)
        resultLabels.append(frameLabel)
    return resultFrames, resultLabels


def createDataset(df: pd.DataFrame, windowSize: int, coverage: int ) -> Dataset :
    frames, labels = createFrames(df, windowSize, coverage)
    return LargeDataset(frames, labels)
