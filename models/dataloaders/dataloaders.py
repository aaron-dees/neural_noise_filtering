
import torch
import os
import librosa
from tqdm import tqdm
import torchaudio
import pandas as pd


class AudioDataset(torch.utils.data.Dataset):
    """AudioDataset class.

    Args:
    ----------
    dataset_path : str
        Path to directory containing the dataset. It must be .wav files.
    audio_size_samples : int
        Size of the training chunks (in samples)
    sampling_rate : int
        Sampling rate
    min_batch_size : int
        Minimum batch size (prevents training on very small batches)
    device : str
        Device (cuda or cpu)
    auto_control_params : list
        Which control parameters to compute automatically. Options: 'loudness', 'centroid'
    control_params_path : str
        If not None it will load control params from this path. Otherwise it will compute them automatically.
    """

    def __init__(self, dataset_path, audio_size_samples, sampling_rate, min_batch_size, device='cuda'):
        self.annotations = pd.read_csv(dataset_path)

        self.audio_size_samples = audio_size_samples
        self.sampling_rate = sampling_rate
        self.min_batch_size = min_batch_size
        self.device = device

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):

        audio_path = self.annotations.iloc[idx, 0]
        audio_data, _ = librosa.load(audio_path, sr=self.sampling_rate, mono=True)
        audio_data = torch.from_numpy(audio_data).to(self.device)
        x_audio = audio_data[..., :self.audio_size_samples]

        x_audio = self.normalise_audio(x_audio)
        
        return x_audio

    def normalise_audio(self, audio):
        audio = audio / torch.max(torch.abs(audio))          
        return audio


class AudioDataset_old(torch.utils.data.Dataset):
    """AudioDataset class.

    Args:
    ----------
    dataset_path : str
        Path to directory containing the dataset. It must be .wav files.
    audio_size_samples : int
        Size of the training chunks (in samples)
    sampling_rate : int
        Sampling rate
    min_batch_size : int
        Minimum batch size (prevents training on very small batches)
    device : str
        Device (cuda or cpu)
    auto_control_params : list
        Which control parameters to compute automatically. Options: 'loudness', 'centroid'
    control_params_path : str
        If not None it will load control params from this path. Otherwise it will compute them automatically.
    """

    def __init__(self, dataset_path, audio_size_samples, sampling_rate, min_batch_size, device='cuda'):
        self.audio_size_samples = audio_size_samples
        self.sampling_rate = sampling_rate
        self.min_batch_size = min_batch_size
        self.audio_data, repeats = self.get_audiofiles(dataset_path = dataset_path, sampling_rate = sampling_rate)
        self.audio_data = self.normalise_audio(self.audio_data)
        self.len_dataset = self.audio_data.shape[-1]//self.audio_size_samples

        # self.control_params = []
        # if control_params_path==None:
        #     #get automatic control_params
        #     if "loudness" in auto_control_params:
        #         loudness, max_loudness, min_loudness = compute_loudness(audio_data=self.audio_data, sampling_rate=sampling_rate, 
        #                                                         n_fft=128, hop_length=32, normalise=True)
        #         loudness = loudness.to(device)
        #         self.control_params.append(loudness)
        #     if "centroid" in auto_control_params:
        #         centroid, max_centroid, min_centroid = compute_centroid(audio_data=self.audio_data, sampling_rate=sampling_rate,
        #                                                                 n_fft=512, hop_length=128, normalise=True)
        #         centroid = centroid.to(device)
        #         self.control_params.append(centroid)
        # else:
        #     #user-defined control params
        #     self.control_params = load_control_params(path=control_params_path, device=device, repeats=repeats)
        # assert len(self.control_params) > 0, f"The model needs at least one control parameter, got: {len(self.control_params)}"

        self.audio_data = self.audio_data.to(device)

    def __len__(self):
        #prevents small batches if the dataset is small. 
        #Training examples change dynamically (chunks are different each train step)
        return max(self.len_dataset,self.min_batch_size)
    
    def __getitem__(self, idx):
        idx = idx % self.len_dataset
        audio_index = idx * self.audio_size_samples
        randomised_idx = torch.randint(-self.audio_size_samples//2, self.audio_size_samples//2, (1,))
        new_audio_index = audio_index+randomised_idx
        x_control_params = []
        #start slice in the negative numbers
        if new_audio_index < 0:
            x_audio = torch.cat([self.audio_data[...,new_audio_index:], self.audio_data[..., :new_audio_index+self.audio_size_samples]], dim=-1)
            # for i in range(len(self.control_params)):
            #     x_control_params.append(torch.cat([self.control_params[i][...,new_audio_index:], self.control_params[i][..., :new_audio_index+self.audio_size_samples]], dim=-1))
        #wrap around
        elif new_audio_index+self.audio_size_samples > self.audio_data.shape[-1]:
            x_audio = torch.cat([self.audio_data[..., new_audio_index:], self.audio_data[..., :self.audio_size_samples - (self.audio_data.shape[-1] - new_audio_index)]], dim=-1)
            # for i in range(len(self.control_params)):
            #     x_control_params.append(torch.cat([self.control_params[i][..., new_audio_index:], self.control_params[i][..., :self.audio_size_samples - (self.audio_data.shape[-1] - new_audio_index)]], dim=-1))
        #normal slicing
        else:
            x_audio = self.audio_data[..., new_audio_index:new_audio_index+self.audio_size_samples]
            # for i in range(len(self.control_params)):
            #     x_control_params.append(self.control_params[i][..., new_audio_index:new_audio_index+self.audio_size_samples])
        # return x_audio, x_control_params
        return x_audio

    def normalise_audio(self, audio):
        audio = audio / torch.max(torch.abs(audio))          
        return audio

    def load_audio(self, audio_path, sampling_rate):
        audio, _ = librosa.load(audio_path, sr=sampling_rate, mono=True)
        return audio

    def get_audiofiles(self, dataset_path, sampling_rate):
        audiofiles = []
        repeats = 1 #handle small audio files and their labelled data
        print(f'(Current working directory: {os.getcwd()})')
        print(f'Reading audio files from {dataset_path}...')
        for root, dirs, files in os.walk(dataset_path):
            for name in tqdm(files):
                if os.path.splitext(name)[-1] == '.wav':
                    audio = self.load_audio(audio_path = os.path.join(root, name), sampling_rate = sampling_rate)
                    audiofiles.append(torch.from_numpy(audio))
        # audiofiles = torch.hstack(audiofiles).unsqueeze(0)
        audiofiles = torch.hstack(audiofiles)
        #handle small audio files
        if audiofiles.shape[-1] < self.audio_size_samples:
            repeats = self.audio_size_samples//audiofiles.shape[-1]+1
            print(f'Audio files length ({audiofiles.shape[-1]}) is shorter than the desired audio size ({self.audio_size_samples}). Repeating audio files to match the desired audio size...')
            audiofiles = audiofiles.repeat(1, repeats)
        print(f'Done. Total audio files length: {audiofiles.shape[-1]} samples (~{(audiofiles.shape[-1]/self.sampling_rate):.6} seconds or ~{((audiofiles.shape[-1]/self.sampling_rate)/60):.4} minutes).')
        print(f'Number of training chunks (slices of size {self.audio_size_samples} samples): {audiofiles.shape[-1]//self.audio_size_samples}.')
        return audiofiles, repeats