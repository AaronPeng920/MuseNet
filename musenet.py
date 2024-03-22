import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as transforms
import lightning.pytorch as pl
from modules import TimeFreqCoTransformer, ResBlock, Downsample, conv_nd, avg_pool_nd
from utils import instantiate_from_config
import torchaudio
from tqdm import tqdm
import os
import json

class MuseNet(nn.Module):
    def __init__(self, in_channels, model_channels, input_size, 
                 cls_count=3, ch_mults=[2, 2, 2, 2, 2], 
                 num_res_blocks=2, n_heads=8, d_head=32, 
                 dropout=0.0, dims=2):
        super().__init__()
        self.input_size = input_size
        self.ch_mults = ch_mults
        
        self.input_conv = conv_nd(dims, in_channels, model_channels, 3, padding=1)
        
        self.blocks = nn.ModuleList()
        ch = model_channels
        for level, mult in enumerate(ch_mults):
            layers = []
            for _ in range(num_res_blocks):
                layers.append(
                    ResBlock(channels=ch, dropout=dropout, out_channels=mult*model_channels)
                )
                ch = mult * model_channels
                layers.append(
                    TimeFreqCoTransformer(ch, n_heads=n_heads, d_head=d_head, dropout=dropout)
                )
            self.blocks.append(nn.Sequential(*layers))
            self.blocks.append(
                nn.Sequential(Downsample(ch, use_conv=True, dims=dims, out_channels=ch))
            )
            
        H, W = self._get_final_feature_size()
        self.cls_head = nn.Sequential(
            nn.Linear(model_channels * ch_mults[-1] * H * W, cls_count),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        x = self.input_conv(x)
        for idx, module in enumerate(self.blocks):
            x = module(x)
        x = torch.flatten(x, start_dim=1)
        pred = self.cls_head(x)
        return pred
    
    def _get_final_feature_size(self):
        H, W = self.input_size
        for _ in range(len(self.ch_mults)):
            H = (H - 1) // 2 + 1
            W = (W - 1) // 2 + 1
        return H, W 

class MuseNetModel(pl.LightningModule):
    def __init__(self, model, criterion, lr):
        super().__init__()
        
        self.model = instantiate_from_config(model)
        self.lr = lr
        if criterion == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss()
        elif criterion == 'mse':
            self.criterion = nn.MSELoss()
        self.save_hyperparameters()
        
    def training_step(self, batch, batch_idx):
        mel_spect, label = batch
        mel_spect.to(self.device)
        label.to(self.device)
        pred = self.model(mel_spect)
        loss = self.criterion(pred, label)
        self.log("train_loss", loss.data)
        return loss
    
    def validation_step(self, batch, batch_idx):
        mel_spect, label = batch
        mel_spect.to(self.device)
        label.to(self.device)
        pred = self.model(mel_spect)
        loss = self.criterion(pred, label)
        self.log("val_loss", loss.data)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        return optimizer

class MuseNetPipeline:
    def __init__(self, sample_rate, n_fft, n_mels, max_audio_duration, ckpt_path, max_batch_size, accuracy=1, save_dir='result/', device='cpu'):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.max_audio_duration = max_audio_duration
        self.accuracy = accuracy
        self.segment_length = self.max_audio_duration * self.sample_rate
        self.hop_length = self.accuracy * self.sample_rate
        model = MuseNetModel.load_from_checkpoint(ckpt_path, map_location='cpu')
        self.musenet = model.model
        self.musenet.eval()
        self.musenet.to(device)
        self.max_batch_size = max_batch_size
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.device = device
        
    def processing(self, audio_meta_data):
        start = 0
        length = audio_meta_data.size(-1)
        end = start + self.segment_length
        mel_transform = transforms.MelSpectrogram(
            sample_rate = self.sample_rate,
            n_fft = self.n_fft,
            n_mels = self.n_mels
        )
        
        mel_spectrograms = []
        info = {}
        start_time = 0
        end_time = start_time + self.accuracy
        idx = 0
        batch_mel_spectrograms = []
        while end < length:
            segment = audio_meta_data[start:end]
            mel_spectrogram = mel_transform(segment)
            mel_spectrogram = mel_spectrogram.unsqueeze(0)
            mel_spectrograms.append(mel_spectrogram)
            start = start + self.hop_length
            end = start + self.segment_length
            if end < length:
                info[idx] = {"start_time": "{}s".format(start_time), "end_time": "{}s".format(end_time), "label": -1, "logits": 0}
            else:
                info[idx] = {"start_time": "{}s".format(start_time), "end_time": "-", "label": -1, "logits": 0}
            start_time = start_time + self.accuracy
            end_time = start_time + self.accuracy
            idx += 1
            if (idx % self.max_batch_size == 0 and idx > 0) or (end >= length):
                mel_spectrograms_a_batch = torch.stack(mel_spectrograms, dim=0)
                batch_mel_spectrograms.append(mel_spectrograms_a_batch)
                mel_spectrograms = []
                
        return batch_mel_spectrograms, info
              
    def pipe(self, audio_filename):
        waveform = self._load_audio_and_transform(audio_filename)
        batch_mel_spectrograms, info = self.processing(waveform)
        with torch.no_grad():
            idx = 0
            for mel_spectrograms in tqdm(batch_mel_spectrograms, desc='Analysing using ' + self.device):
                mel_spectrograms = mel_spectrograms.to(self.device)
                pred = self.musenet(mel_spectrograms)
                pred_logits, pred_label = torch.max(pred, dim=-1)        # logits, label
                for (label, logits) in zip(pred_label, pred_logits):
                    info[idx]['label'] = label.item()
                    info[idx]['logits'] = logits.item()
                    idx += 1
                    
        json_str = json.dumps(info)
        basename = os.path.splitext(os.path.basename(audio_filename))[0]
        save_filename = os.path.join(self.save_dir, basename + '.json')
        with open(save_filename, 'w') as json_file:
            json_file.write(json_str)
        print("Analysis Json File Saved in", save_filename)
        
    def _load_audio_and_transform(self, audio_filename):
        waveform, src_sr = torchaudio.load(audio_filename)
        resample = transforms.Resample(src_sr, self.sample_rate)
        waveform = resample(waveform)
        # stereo to mono
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        waveform = waveform.squeeze()
        src_length = waveform.size(-1)
        segment_length = self.max_audio_duration * self.sample_rate
        hop_length = self.accuracy * self.sample_rate
        times = (src_length - segment_length) // hop_length + 1 if src_length >= segment_length else 0
        effective_length = (times - 1) * hop_length + segment_length if times >= 1 else src_length
        pad_length = (segment_length - (src_length - effective_length)) % segment_length
        waveform = F.pad(waveform, (0, pad_length), "constant", 0)  
        return waveform

if __name__ == '__main__':
    model = MuseNetPipeline(16000, 1024, 128, 5, '/home/app/project/MuseNet/logs/musenetlog/version_1/checkpoints/epoch=4-step=25.ckpt', 32, 1, device='cuda')
    audio_filename = "/home/app/datasets/music/gtzan/vocal_noslience/instrument_blues.00000.wav.reformatted.wav_10.wav.wav"
    model.pipe(audio_filename)