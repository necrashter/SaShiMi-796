"""
Dataset creation & data preprocessing for WAV audio files.
"""
import torch
import glob
import os
from scipy.io import wavfile
from torch.utils.data import Dataset


class WavDataset(Dataset):
    """
    Represents a folder of WAV files.
    """
    
    def __init__(self, root_dir, transform=None):
        """
        Arguments:
            root_dir (string): Directory with all the WAV files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform

        self.wav_paths = sorted(glob.glob(os.path.join(root_dir, "*.wav")))
        wavs = [wavfile.read(path) for path in self.wav_paths]

        # Check if all sample rates are the same
        sample_rates = [i[0] for i in wavs]
        sample_rate = sample_rates[0]
        sample_rates = torch.Tensor(sample_rates)
        if not torch.all(sample_rate == sample_rates):
            raise ValueError("Not all WAV files have the same sample rate!")
        self.sample_rate = sample_rate

        self.wavs = [i[1] for i in wavs]
    
    def __len__(self):
        return len(self.wavs)

    def __getitem__(self, idx):
        audio = self.wavs[idx]
        
        if self.transform:
            audio = self.transform(audio)

        return audio
