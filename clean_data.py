import numpy as np
import torch

def is_noisy_eeg(eeg_tensor, 
                 amplitude_threshold=1000,
                 flat_channel_threshold=1e-8,
                 nan_threshold=0.9):
    """
    Minimal check - only remove truly broken EEG data.
    
    Args:
        eeg_tensor: Single EEG sample (channels × samples or just samples)
        amplitude_threshold: Max acceptable amplitude (in µV)
        flat_channel_threshold: Variance threshold for dead channels
        nan_threshold: Max percentage of NaN/Inf values allowed (0-1)
    
    Returns:
        bool: True if broken, False if good
        str: Reason for rejection (if broken)
    """
    
    # Convert to numpy if tensor
    if isinstance(eeg_tensor, torch.Tensor):
        eeg = eeg_tensor.cpu().numpy()
    else:
        eeg = np.array(eeg_tensor)
    
    # 1. Check for too many NaN or Inf values
    invalid_count = np.isnan(eeg).sum() + np.isinf(eeg).sum()
    invalid_ratio = invalid_count / eeg.size
    if invalid_ratio > nan_threshold:
        return True, f"Too many NaN/Inf values ({invalid_ratio*100:.1f}%)"
    
    # Replace NaN/Inf for further checks
    eeg_clean = np.nan_to_num(eeg, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 2. Check for completely flat/dead channels (zero variance)
    if eeg_clean.ndim == 2:  # Multiple channels
        variances = np.var(eeg_clean, axis=1)
        # Only flag if ALL or almost all channels are dead
        dead_channels = np.sum(variances < flat_channel_threshold)
        if dead_channels == eeg_clean.shape[0]:  # All channels are dead
            return True, f"All channels are dead/flat"
    elif eeg_clean.ndim == 1:  # Single channel
        if np.var(eeg_clean) < flat_channel_threshold:
            return True, "Channel is completely flat/dead"
    
    # 3. Check for extreme amplitudes (only very extreme values)
    max_amplitude = np.max(np.abs(eeg_clean))
    if max_amplitude > amplitude_threshold:
        return True, f"Extreme amplitude detected ({max_amplitude:.1f} µV)"
    
    return False, "Good"


def filter_eeg_data(eeg_list, labels_list, 
                    amplitude_threshold=1000,
                    flat_channel_threshold=1e-8,
                    nan_threshold=0.9,
                    verbose=True):
    """
    Filter only truly broken EEG data and keep labels synchronized.
    
    Args:
        eeg_list: List of EEG tensors
        labels_list: List of corresponding labels
        amplitude_threshold: Max acceptable amplitude
        flat_channel_threshold: Variance threshold for dead channels
        nan_threshold: Max percentage of NaN/Inf values allowed
        verbose: Print filtering progress
    
    Returns:
        clean_eeg: List of good EEG samples
        clean_labels: List of corresponding labels
        removed_indices: Indices of removed samples
        removal_reasons: Reasons for each removal
    """
    
    clean_eeg = []
    clean_labels = []
    removed_indices = []
    removal_reasons = []
    
    for idx, (eeg, label) in enumerate(zip(eeg_list, labels_list)):
        is_broken, reason = is_noisy_eeg(
            eeg, 
            amplitude_threshold=amplitude_threshold,
            flat_channel_threshold=flat_channel_threshold,
            nan_threshold=nan_threshold
        )
        
        if is_broken:
            removed_indices.append(idx)
            removal_reasons.append(reason)
            if verbose:
                print(f"Removed sample {idx} (label: {label}) - Reason: {reason}")
        else:
            clean_eeg.append(eeg)
            clean_labels.append(label)
    
    if verbose:
        print(f"\n--- Filtering Summary ---")
        print(f"Total samples: {len(eeg_list)}")
        print(f"Removed: {len(removed_indices)}")
        print(f"Kept: {len(clean_eeg)}")
        if len(eeg_list) > 0:
            print(f"Removal rate: {len(removed_indices)/len(eeg_list)*100:.1f}%")
        
        if removal_reasons:
            print(f"\n--- Removal Reasons ---")
            from collections import Counter
            reason_counts = Counter(removal_reasons)
            for reason, count in reason_counts.most_common():
                print(f"  {reason}: {count}")
    
    return clean_eeg, clean_labels, removed_indices