from collections import Counter

import pandas as pd
from torch.utils.data import Dataset

from our_datasets import LargeDataset


def createFrames(df: pd.DataFrame, windowSize: int, coverage: int):
    from collections import Counter
    coverage = min(windowSize, coverage)
    moveSize = max(1, windowSize - coverage)  # avoid infinite loop
    
    labelMapping = {'Up': 0, 'Down': 1, 'Right': 2, 'Left': 3}
    df['Annotation'] = df['Annotation'].map(labelMapping).astype(int)
    
    labels = df["Annotation"].values.tolist()
    eegColumns = ['Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2','T3','T4','T5','T6','F7','F8']
    
    eegData = df[eegColumns].values.astype(float)  # shape: (num_samples, num_channels)
    eegData = (eegData - eegData.mean(axis=0)) / (eegData.std(axis=0) + 1e-6)
    
    resultFrames = []
    resultLabels = []
    i = 0
    while i + windowSize < len(eegData):
        tmp = eegData[i:i + windowSize]
        tmpLabels = labels[i:i + windowSize]
        i += moveSize
        frameLabel = Counter(tmpLabels).most_common(1)[0][0]
        resultFrames.append(tmp)
        resultLabels.append(frameLabel)
    
    return resultFrames, resultLabels


def createOurDataset(df: pd.DataFrame, windowSize: int, coverage: int ) -> Dataset :
    frames, labels = createFrames(df, windowSize, coverage)
    return LargeDataset(frames, labels)
