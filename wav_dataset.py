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


class YoutubeMixTransform:
    """
    Transform class for autoregressive generation on Youtube Mix Dataset.
    """
    def __init__(self, device=None, sequence_length: int = 958400):
        
        """
        - device: Samples will be moved to this device after preprocessing.
        - sequence_length: Sample count. 958400 by default because some WAV files
          are a bit shorter than 1 minute.
        """
        self.device = device
        self.sequence_length = sequence_length

    def __call__(self, audio):
        """
        Returns:
        - x: Samples 0 to N, each sample is a float between -1 and +1.
        - y: Samples 1 to N+1, each sample is an integer between 0 to 255.
        """
        # 16 bit signed into float -1 to +1
        audio = (torch.from_numpy(audio) / 2**15)
        x = audio[:self.sequence_length]
        y = audio[1:1+self.sequence_length]
        # Make y labels from 0 to 255
        y = torch.clamp((y / 2.0) + 0.5, 0.0, 1.0)
        y = torch.mul(y, 255.0).to(torch.int32)
        if self.device is not None:
            x = x.to(self.device)
            y = y.to(self.device)
        return x, y
