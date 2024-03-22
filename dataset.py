import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
from utils import instantiate_from_config
import lightning.pytorch as pl

class AudioDataset(Dataset):
    def __init__(self, root_dir, max_audio_duration, sample_rate):
        super().__init__()
        
        self.root_dir = root_dir
        self.max_audio_duration = max_audio_duration
        self.sample_rate = sample_rate
        self.labels = sorted(os.listdir(self.root_dir))
        self.label_to_idx = {self.labels[i]: i for i in range(len(self.labels))}
        self.audio_filenames = self._get_audio_filenames_label()

    def __len__(self):
        return len(self.audio_filenames)

    def __getitem__(self, idx):
        audio_filename, audio_label = self.audio_filenames[idx]
        waveform = self._load_audio_and_transform(audio_filename)
        return waveform, torch.tensor(self.label_to_idx[audio_label])
    
    def _get_audio_filenames_label(self):
        audio_filenames = []
        for label in self.labels:
            label_dir = os.path.join(self.root_dir, label)
            for filename in os.listdir(label_dir):
                audio_filenames.append([os.path.join(label_dir, filename), label])
        return audio_filenames
    
    def _load_audio_and_transform(self, audio_filename):
        waveform, src_sr = torchaudio.load(audio_filename)
        resample = transforms.Resample(src_sr, self.sample_rate)
        waveform = resample(waveform)
        # stereo to mono
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        waveform = waveform.squeeze()
        src_length = waveform.size(-1)
        tgt_length = self.max_audio_duration * self.sample_rate
        if src_length > tgt_length:
            waveform = waveform[:tgt_length]
        elif src_length < tgt_length:
            pad_length = tgt_length - src_length
            waveform = F.pad(waveform, (0, pad_length), "constant", 0)
        return waveform
        
class MelSpectrogramDataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, val_dataset, batch_size, num_worker, sample_rate, n_fft=1024, n_mels=128):
        super().__init__()
        
        self.train_dataset_config = train_dataset
        self.val_dataset_config = val_dataset
        self.batch_size = batch_size
        self.num_worker = num_worker
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.n_mels = n_mels
        

    def setup(self, stage=None):
        self.train_dataset = instantiate_from_config(self.train_dataset_config)
        self.val_dataset = instantiate_from_config(self.val_dataset_config)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                          num_workers=self.num_worker, collate_fn=self.collate_fn, pin_memory=True, shuffle=True)
        
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, 
                          num_workers=self.num_worker, collate_fn=self.collate_fn, pin_memory=True, shuffle=False)
        

    def collate_fn(self, batch):
        waveforms, labels = zip(*batch)
        mel_transform = transforms.MelSpectrogram(
            sample_rate = self.sample_rate,
            n_fft = self.n_fft,
            n_mels = self.n_mels
        )
        mel_spectrograms = []
        for waveform in waveforms:
            mel_spectrogram = mel_transform(waveform)
            mel_spectrogram = mel_spectrogram.unsqueeze(0)
            mel_spectrograms.append(mel_spectrogram)
        return torch.stack(mel_spectrograms), torch.tensor(labels)
    