"""
Dataset creation & data preprocessing for WAV audio files.
"""
import torch
import numpy as np
import glob
import os
from scipy.io import wavfile
from torch.utils.data import Dataset
from torchaudio.functional import mu_law_encoding


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

    if not wavs:
        raise ValueError("There are no WAV files in the given path!")

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
        - x: Samples 0 to N-1
        - y: Samples 1 to N

        Each sample is an integer between 0 to 255, encoded using mu-law.
        """
        audio = torch.from_numpy(audio)

        if self.device is not None:
            audio = audio.to(self.device)

        # 16 bit signed into float -1 to +1
        audio = audio / 2**15
        audio = mu_law_encoding(audio, 256).to(torch.int64)
        x = audio[:-1]  # First samples
        y = audio[1:]   # Last samples
        return x, y


class SC09(Dataset):
    """
    Represents a folder of WAV files from SC09 dataset.
    Each WAV file must be 1 second long with 16000 Hz bitrate.
    """
    def __init__(self, root_dir, device=None):
        """
        Arguments:
            root_dir (string): Directory with all the WAV files.
            device: Torch device to which the samples will be moved.
        """
        self.root_dir = root_dir
        self.wav_paths = sorted(glob.glob(os.path.join(root_dir, "*.wav")))
        self.device = device

        # Check the dataset
        print("WAV files in directory:", len(self.wav_paths))
        good = []
        for path in self.wav_paths:
            sample_rate, wav = wavfile.read(path)
            length = wav.shape[0]
            if sample_rate != 16000:
                raise ValueError(f"WAV file {path} doesn't have a sample rate of 16000 Hz. " +
                                 f"({sample_rate} Hz)")
            if length != 16000:
                good.append(False)
                continue
            good.append(True)

        num_bad = len([i for i in good if not i])
        if num_bad > 0:
            self.wav_paths = [path for path, condition in zip(self.wav_paths, good) if condition]
            print(num_bad, "WAV file(s) were discarded because they were not 1 second long.")

    def __len__(self):
        return len(self.wav_paths)

    def __getitem__(self, idx):
        path = self.wav_paths[idx]
        audio = wavfile.read(path)[1]
        # 16 bit signed into float -1 to +1
        audio = (torch.from_numpy(audio) / 2**15)
        audio = torch.nn.functional.pad(audio, (1, 0))

        if self.device is not None:
            audio = audio.to(self.device)

        audio = mu_law_encoding(audio, 256).to(torch.int64)
        x = audio[:-1]
        y = audio[1:]

        return x, y
