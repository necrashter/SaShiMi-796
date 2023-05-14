"""
Dataset creation & data preprocessing for WAV audio files.
"""
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
        self.wav_pths = glob.glob(os.path.join(root_dir, "*.wav"))
        self.wavs = [wavfile.read(path) for path in self.wav_pths]
    
    def __len__(self):
        return len(self.wav_pths)

    def __getitem__(self, idx):
        sampling_rate, audio = self.wavs[idx]
        
        if self.transform:
            audio = self.transform(audio)

        return audio
