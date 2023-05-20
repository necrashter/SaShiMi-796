"""
Dataset creation & data preprocessing for WAV audio files.
"""
import torch
import numpy as np
import glob
import os
from scipy.io import wavfile
from torch.utils.data import Dataset


def read_wavs(root_dir):
    """
    Read all WAV files from given directory.

    Returns:
    - sample_rate (int): Sample rate of WAV files.
    - wavs (list of numpy arrays): Contents of WAV files.

    Raises ValueError if not all WAV files have the same sample rate.
    """
    wav_paths = sorted(glob.glob(os.path.join(root_dir, "*.wav")))
    wavs = [wavfile.read(path) for path in wav_paths]

    # Check if all sample rates are the same
    sample_rates = [i[0] for i in wavs]
    sample_rate = sample_rates[0]
    sample_rates = torch.Tensor(sample_rates)
    if not torch.all(sample_rate == sample_rates):
        raise ValueError("Not all WAV files have the same sample rate!")
    sample_rate = sample_rate

    wavs = [i[1] for i in wavs]

    return sample_rate, wavs


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

        self.sample_rate, self.wavs = read_wavs(root_dir)
    
    def __len__(self):
        return len(self.wavs)

    def __getitem__(self, idx):
        audio = self.wavs[idx]
        
        if self.transform:
            audio = self.transform(audio)

        return audio


class YoutubeMixDataset(WavDataset):
    """
    Youtube Mix Dataset.
    """
    def __init__(self, root_dir, duration: float = 8.0, device=None):
        """
        Arguments:
            root_dir (string): Directory with all the WAV files.
            duration: Duration of each sample in seconds.
            device: Torch device to which the samples will be moved.
        """
        super().__init__(root_dir, YoutubeMixTransform(device))

        sequence_length = round(self.sample_rate * duration)

        wavs = []
        queue = self.wavs
        queue.reverse()
        head = queue.pop()
        while True:
            if head.shape[0] > sequence_length:
                wavs.append(head[:sequence_length+1])
                head = head[sequence_length:]
            elif queue:
                head = np.concatenate((head, queue.pop()))
            else:
                break

        self.wavs = wavs


class YoutubeMixTransform:
    """
    Transform class for autoregressive generation on Youtube Mix Dataset.
    """
    def __init__(self, device=None):
        
        """
        - device: Samples will be moved to this device after preprocessing.
        - sequence_length: Sample count. 958400 by default because some WAV files
          are a bit shorter than 1 minute.
        """
        self.device = device

    def __call__(self, audio):
        """
        Returns:
        - x: Samples 0 to N-1, each sample is a float between -1 and +1.
        - y: Samples 1 to N, each sample is an integer between 0 to 255.
        """
        # 16 bit signed into float -1 to +1
        audio = (torch.from_numpy(audio) / 2**15)
        x = audio[:-1]  # First samples
        y = audio[1:]   # Last samples
        # Make y labels from 0 to 255
        y = torch.clamp((y / 2.0) + 0.5, 0.0, 1.0)
        y = torch.mul(y, 255.0).to(torch.int64)
        if self.device is not None:
            x = x.to(self.device)
            y = y.to(self.device)
        return x, y
