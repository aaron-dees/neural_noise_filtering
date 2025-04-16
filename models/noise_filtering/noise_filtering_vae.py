import sys
sys.path.append('../')

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.nn import functional as F
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy import fft
import torch_dct as dct
import torchaudio
import librosa
import math
import cached_conv as cc

from utils.utilities import sample_from_distribution, generate_noise_grains
from utils.dsp_components import noise_filtering, mod_sigmoid, safe_log10, amp_to_impulse_response, fft_convolve, fft_convolve_ddsp, minimum_phase, frame
from models.noise_filtering.filterbank import FilterBank
from scripts.configs.noise_filtering_config import SAMPLE_RATE, DEVICE, NORMALIZE_OLA, HOP_SIZE_RATIO

def compute_magnitude_filters(filters):
    magnitude_filters = torch.fft.rfft(filters)
    magnitude_filters = torch.abs(magnitude_filters)
    return magnitude_filters

def check_power_of_2(x):
    return 2 ** int(math.log(x, 2)) == x

def get_next_power_of_2(x):
    return int(math.pow(2, math.ceil(math.log(x)/math.log(2))))

def pad_filters(filters, n_samples):
    for i in range(len(filters)):
        filters[i] = np.pad(filters[i], (n_samples-len(filters[i]),0))
    return torch.from_numpy(np.array(filters))

# Builds loopable noise bands, based on filter bank and white nose, see 'deterministic loopable noise bands in original paper'
def get_noise_bands(fb, min_noise_len, normalize):
    #build deterministic loopable noise bands
    if fb.max_filter_len > min_noise_len:
        noise_len = get_next_power_of_2(fb.max_filter_len)
    else:
        noise_len = min_noise_len
    filters = pad_filters(fb.filters, noise_len)
    magnitude_filters = compute_magnitude_filters(filters=filters)
    torch.manual_seed(42) #enforce deterministic noise
    phase_noise = torch.FloatTensor(magnitude_filters.shape[0], magnitude_filters.shape[-1]).uniform_(-math.pi, math.pi).to(magnitude_filters.device)
    phase_noise = torch.exp(1j*phase_noise)
    phase_noise[:,0] = 0
    phase_noise[:,-1] = 0
    magphase = magnitude_filters*phase_noise
    noise_bands = torch.fft.irfft(magphase)
    if normalize:
        noise_bands = (noise_bands / torch.max(noise_bands.abs())) 

    # test noise look by concatonating along x axis.
    # cat = torch.concat((noise_bands[10], noise_bands[10]))
    # cat = torch.concat((cat, noise_bands[10]))
    # print(cat.shape) 
    # torchaudio.save(f'/Users/adees/Code/neural_granular_synthesis/scripts/recon_audio/loopNoise.wav', noise_bands[10].to(torch.float32).unsqueeze(0).cpu(), 44100)
    # torchaudio.save(f'/Users/adees/Code/neural_granular_synthesis/scripts/recon_audio/loopNoiseCat.wav', cat.to(torch.float32).unsqueeze(0).cpu(), 44100)

    return noise_bands.unsqueeze(0).float(), noise_len

# MLP and GRU for v2
def mlp(in_size, hidden_size, n_layers):
    channels = [in_size] + (n_layers) * [hidden_size]
    net = []
    for i in range(n_layers):
        net.append(nn.Linear(channels[i], channels[i + 1]))
        net.append(nn.LayerNorm(channels[i + 1]))
        net.append(nn.LeakyReLU())
    return nn.Sequential(*net)

def gru(n_input, hidden_size):
    return nn.GRU(n_input * hidden_size, hidden_size, batch_first=True)

class linear_block(nn.Module):
    def __init__(self, in_size,out_size,norm="BN"):
        super(linear_block, self).__init__()
        if norm=="BN":
            self.block = nn.Sequential(nn.Linear(in_size,out_size),nn.BatchNorm1d(out_size),nn.LeakyReLU(0.2))
        if norm=="LN":
            self.block = nn.Sequential(nn.Linear(in_size,out_size),nn.LayerNorm(out_size),nn.LeakyReLU(0.2))
    def forward(self, x):
        return self.block(x)

#############
# v1
#   - single dense layer
#############

class SpectralEncoder_v1(nn.Module):

    def __init__(self,
                    n_grains,
                    z_dim = 128,
                    l_grain=2048,
                    h_dim = 512,
                    n_bands = 2048
                    ):
        super(SpectralEncoder_v1, self).__init__()


        self.n_grains = n_grains
        self.l_grain = l_grain
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_bands = n_bands


        self.flatten_size = int((l_grain//2)+1)
        self.encoder_linears = nn.Sequential(linear_block(self.flatten_size,h_dim))
        self.mu = nn.Linear(h_dim,z_dim)
        self.logvar = nn.Sequential(nn.Linear(h_dim,z_dim),nn.Hardtanh(min_val=-5.0, max_val=5.0)) # clipping to avoid numerical instabilities


    def encode(self, x):

        # The reshape is important for the KL Loss and trating each grains as a batch value,
        # This reshap can be performed here or simply before the KL loss calculation.
        in_size = x.shape


        mb_grains = x.reshape(x.shape[0]*self.n_grains,(self.l_grain//2)+1)

        # Linear layer
        h = self.encoder_linears(mb_grains)

        # h --> z
        # h of shape [bs*n_grains,z_dim]
        mu = self.mu(h)
        logvar = self.logvar(h)

        # z of shape [bs*n_grains,z_dim]
        z = sample_from_distribution(mu, logvar)

        return z, mu, logvar

    def forward(self, audio):

        z, mu, logvar = self.encode(audio)
        # z = self.encode(audio)

        return z, mu, logvar
        # return z

class SpectralEncoder_v2(nn.Module):

    def __init__(self,
                    z_dim = 128,
                    l_grain=2048,
                    h_dim = 512,
                    n_bands = 2048,
                    synth_window = 32,
                    n_mfcc = 30,
                    mfcc_n_fft = 512,
                    mfcc_n_mels = 128,
                    mfcc_hop_size = 128
                    ):
        super(SpectralEncoder_v2, self).__init__()


        self.l_grain = l_grain
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_bands = n_bands
        self.synth_window = synth_window

        # self.flatten_size = int((l_grain//2)+1)
        self.flatten_size = n_mfcc
        self.encoder_linears = nn.Sequential(linear_block(self.flatten_size,h_dim))
        self.mu = nn.Linear(h_dim,z_dim)
        self.logvar = nn.Sequential(nn.Linear(h_dim,z_dim),nn.Hardtanh(min_val=-5.0, max_val=5.0)) # clipping to avoid numerical instabilities

        self.mfcc = torchaudio.transforms.MFCC(
        sample_rate=SAMPLE_RATE,
            n_mfcc=n_mfcc,
            log_mels=True,
            melkwargs={
                "n_fft": mfcc_n_fft,
                "n_mels": mfcc_n_mels,
                "hop_length": mfcc_hop_size,
                # "hop_length": 32,
                "f_min": 20.0,
                "f_max": 8000.0
        }).to(DEVICE)

        #TODO check this is ok, I've switch from using 2D norm to 1D, since I want my n_frames to be dynamic.
        self.norm = nn.LayerNorm(self.flatten_size)
        # self.norm = nn.LayerNorm((self.n_grains, self.flatten_size))
        self.gru = nn.GRU(input_size=self.flatten_size, hidden_size=self.h_dim, batch_first=True)
        self.linear = nn.Linear(self.h_dim, self.z_dim)
        self.mu = nn.Linear(self.h_dim, self.z_dim)
        self.logvar = nn.Linear(self.h_dim, self.z_dim)



    def encode(self, x, noise_synth='filterbank'):

        # h = x.reshape(x.shape[0]*self.n_grains,(self.l_grain//2)+1)
        # mfccs = spectral_ops.compute_mfcc(
        #     audio,
        #     lo_hz=20.0,
        #     hi_hz=8000.0,
        #     fft_size=self.fft_size,
        #     mel_bins=128,
        #     mfcc_bins=30,
        #     overlap=self.overlap,
        #     pad_end=True)

        # # Normalize.
        # z = self.z_norm(mfccs[:, :, tf.newaxis, :])[:, :, 0, :]
        # # Run an RNN over the latents.
        # z = self.rnn(z)
        # # Bounce down to compressed z dimensions.
        # z = self.dense_out(z)

        # Based on the input signal size and the synth window, calculate the internal sample size, this is to account for upsampling later
        mfccs = self.mfcc(x).permute(0,2,1)

        # NOTE Keep all frames is using DDSP Noise synth, and crop is using noisefilterbank

        if(noise_synth == "ddsp"):
            h = self.norm(mfccs[:,:,:])
        else:
            in_size = x.shape[-1]
            n_frames = int(in_size // self.synth_window)
            h = self.norm(mfccs[:,:n_frames,:])

        h = self.gru(h)[0]
        mu = self.mu(h)
        logvar = self.logvar(h)

        # z of shape [bs*n_grains,z_dim]
        z = sample_from_distribution(mu, logvar)

        return z, mu, logvar

    def forward(self, audio, noise_synth="filterbank"):

        z, mu, logvar = self.encode(audio, noise_synth=noise_synth)
        # z = self.encode(audio)

        return z, mu, logvar
        # return z

class SpectralEncoder_v3(nn.Module):

    def __init__(self,
                    z_dim = 128,
                    l_grain=2048,
                    h_dim = 512,
                    n_bands = 2048,
                    synth_window = 32,
                    n_cc = 64,
                    cc_frame_size = 512,
                    cc_hop_size = 128,
                    
                    ):
        super(SpectralEncoder_v3, self).__init__()


        self.l_grain = l_grain
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_bands = n_bands
        self.synth_window = synth_window
        self.cc_frame_size = cc_frame_size
        self.cc_hop_size = cc_hop_size
        self.n_cc = n_cc

        self.flatten_size = int((self.cc_frame_size//2)+1)
        self.encoder_linears = nn.Sequential(linear_block(self.flatten_size,h_dim))
        self.mu = nn.Linear(h_dim,z_dim)
        self.logvar = nn.Sequential(nn.Linear(h_dim,z_dim),nn.Hardtanh(min_val=-5.0, max_val=5.0)) # clipping to avoid numerical instabilities


        #TODO check this is ok, I've switch from using 2D norm to 1D, since I want my n_frames to be dynamic.
        self.norm = nn.LayerNorm(self.flatten_size)
        # self.norm = nn.LayerNorm((self.n_grains, self.flatten_size))
        self.gru = nn.GRU(input_size=self.flatten_size, hidden_size=self.h_dim, batch_first=True)
        self.linear = nn.Linear(self.h_dim, self.z_dim)
        self.mu = nn.Linear(self.h_dim, self.z_dim)
        self.logvar = nn.Linear(self.h_dim, self.z_dim)



    def encode(self, x, noise_synth='filterbank'):

                        # ---------- Turn Waveform into grains ----------
        ola_window = torch.from_numpy(signal.hann(self.cc_frame_size,sym=False)).type(torch.float32).to(DEVICE)
        stft_audio = torch.stft(x, n_fft = self.cc_frame_size, hop_length = self.cc_hop_size, window=ola_window, center=True, return_complex=True, pad_mode="constant")
        

        # ---------- Turn Waveform into grains END ----------


        # # ---------- Get CCs, or MFCCs and invert ----------
        # CCs
        # print(torch.abs(stft_audio.sum()))
        grain_db = 20*safe_log10(torch.abs(stft_audio))
        # cepstral_coeff = dct.dct(torch.from_numpy(y_log_audio).permute(1,0))
        cepstral_coeff = dct.dct(grain_db.permute(0,2,1))
        cepstral_coeff[:,:,self.n_cc:] = 0
        inv_cep_coeffs = 10**(dct.idct(cepstral_coeff) / 20)
        
        # # ---------- Get CCs, or MFCCs and invert END ----------


        # NOTE Keep all frames is using DDSP Noise synth, and crop is using noisefilterbank

        if(noise_synth == "ddsp"):
            h = self.norm(inv_cep_coeffs[:,:,:])
        else:
            in_size = x.shape[-1]
            n_frames = int(in_size // self.synth_window)
            h = self.norm(inv_cep_coeffs[:,:n_frames,:])
        # h = h.unsqueeze(-2)
        # z = z.permute(2, 0, 1)
        h = self.gru(h)[0]
        # print(img)
        # h = h.reshape(h.shape[0]*n_frames,self.h_dim)
        # z = self.linear(z)
        mu = self.mu(h)
        logvar = self.logvar(h)

        # z of shape [bs*n_grains,z_dim]
        z = sample_from_distribution(mu, logvar)

        return z, mu, logvar

    def forward(self, audio, noise_synth="filterbank"):

        z, mu, logvar = self.encode(audio, noise_synth=noise_synth)
        # z = self.encode(audio)

        return z, mu, logvar
        # return z

# CNN encoder with Mel Spec / Inv MFCCs
class SpectralEncoder_v4(nn.Module):

    def __init__(self,
                    z_dim = 128,
                    l_grain=2048,
                    h_dim = 512,
                    n_bands = 2048,
                    synth_window = 32,
                    n_mfcc = 128,
                    mfcc_n_fft = 512,
                    mfcc_n_mels = 128,
                    mfcc_hop_size = 128,
                    n_mels = 128
                    ):
        super(SpectralEncoder_v4, self).__init__()


        self.l_grain = l_grain
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_bands = n_bands
        self.synth_window = synth_window
        self.hop_length = mfcc_hop_size
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels

        # self.mu = nn.Linear(h_dim,z_dim)
        # self.logvar = nn.Sequential(nn.Linear(h_dim,z_dim),nn.Hardtanh(min_val=-5.0, max_val=5.0)) # clipping to avoid numerical instabilities

        self.norm = nn.LayerNorm(self.n_mels)
        self.mu = nn.Linear(self.h_dim, self.z_dim)
        self.logvar = nn.Linear(self.h_dim, self.z_dim)

        # transform = T.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, win_length=1024, hop_length=block_size, 
                                #  f_min=20, f_max=8000, n_mels=128, center=True, normalized=True)
        self.mel_spec = torchaudio.transforms.MelSpectrogram(SAMPLE_RATE, n_fft=self.l_grain, hop_length=self.hop_length, norm='slaney', power=2, mel_scale='slaney', center=True, n_mels=self.n_mels)
        #look into using cached conv
        # self.cnn1 = nn.Conv1d(128, 32, 3, stride=1, padding=1)
        # self.cnn2 = nn.Conv1d(32, 8, 3, stride=1, padding=1)
        self.cnn1 = cc.Conv1d(128, 64, 3, stride=1, padding=(1,1))
        self.bn1 = nn.BatchNorm1d(64)
        self.act1 = nn.LeakyReLU(0.2)
        self.cnn2 = cc.Conv1d(64, 32, 3, stride=1, padding=(1,1))
        self.bn2 = nn.BatchNorm1d(32)
        self.act2 = nn.LeakyReLU(0.2)
        self.cnn3 = cc.Conv1d(32, 16, 3, stride=1, padding=(1,1))
        self.bn3 = nn.BatchNorm1d(16)
        self.act3 = nn.LeakyReLU(0.2)
        self.mu = nn.Linear(16, self.z_dim)
        self.logvar = nn.Linear(16, self.z_dim)



    def encode(self, x, noise_synth='filterbank'):

        #Input rep - MEL SPEC. MFCCs. MEL_ENERGY
        #CNN_Block {
            #CNN1D
            #RELU
            #BN
        # } X 3
        #Mu
        #logVar Layers


        mel_spec = self.mel_spec(x)

        # ----- INV MFCC CODE
        # mel_spec_db = 20*safe_log10(mel_spec)
        # mfcc = dct.dct(mel_spec_db.permute(0,2,1))
        # mfcc[:,:,self.n_mfcc:] = 0
        # inv_mfcc = 10**(dct.idct(mfcc) / 20).permute(0,2,1)
        # mel_spec = inv_mfcc
        # ----- INV MFCC CODE END

        if(noise_synth == "ddsp"):
            mel_spec = self.norm(mel_spec.permute(0,2,1)[:,:,:]).permute(0,2,1)
        else:
            in_size = x.shape[-1]
            n_frames = int(in_size // self.synth_window)
            mel_spec = self.norm(mel_spec.permute(0,2,1)[:,:n_frames,:]).permute(0,2,1)


        # print(S.shape)
        # print(S.sum())
        # print(mel_spec.shape)
        # print(mel_spec.sum())
        # plt.plot(mel_spec[0,0])
        # plt.plot(S[0,:,0])
        # plt.savefig("test.png")
        z = self.cnn1(mel_spec)
        z = self.bn1(z)
        z = self.act1(z)
        # z = F.relu(z)
        z = self.cnn2(z)
        z = self.bn2(z)
        z = self.act2(z)

        z = self.cnn3(z)
        z = self.bn3(z)
        z = self.act3(z)
        z = z.permute(0, 2, 1)


        mu = self.mu(z)
        logvar = self.logvar(z)

        z = sample_from_distribution(mu, logvar)

        return z, mu, logvar

    def forward(self, audio, noise_synth="filterbank"):

        z, mu, logvar = self.encode(audio, noise_synth=noise_synth)

        return z, mu, logvar

# Inv MFCC - GRU Architecture
class SpectralEncoder_v5(nn.Module):

    def __init__(self,
                    z_dim = 128,
                    l_grain=2048,
                    h_dim = 512,
                    n_bands = 2048,
                    synth_window = 32,
                    n_mfcc = 30,
                    mfcc_n_fft = 512,
                    mfcc_n_mels = 128,
                    mfcc_hop_size = 128,
                    n_mels = 128
                    ):
        super(SpectralEncoder_v5, self).__init__()


        self.l_grain = l_grain
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_bands = n_bands
        self.synth_window = synth_window
        self.hop_length = mfcc_hop_size
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels

        # self.flatten_size = int((l_grain//2)+1)
        # need to change this to number of mel bands.
        self.flatten_size = self.n_mels
        self.mu = nn.Linear(h_dim,z_dim)
        self.logvar = nn.Sequential(nn.Linear(h_dim,z_dim),nn.Hardtanh(min_val=-5.0, max_val=5.0)) # clipping to avoid numerical instabilities

        #TODO check this is ok, I've switch from using 2D norm to 1D, since I want my n_frames to be dynamic.
        # self.norm = nn.LayerNorm(128)
        self.norm = nn.LayerNorm(self.n_mels)
        # self.norm = nn.LayerNorm((self.n_grains, self.flatten_size))
        self.gru = nn.GRU(input_size=self.flatten_size, hidden_size=self.h_dim, batch_first=True)
        self.linear = nn.Linear(self.h_dim, self.z_dim)
        self.mu = nn.Linear(self.h_dim, self.z_dim)
        self.logvar = nn.Linear(self.h_dim, self.z_dim)

        # transform = T.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, win_length=1024, hop_length=block_size, 
                                #  f_min=20, f_max=8000, n_mels=128, center=True, normalized=True)
        self.mel_spec = torchaudio.transforms.MelSpectrogram(SAMPLE_RATE, n_fft=self.l_grain, hop_length=self.hop_length, norm='slaney', power=2, mel_scale='slaney', center=True, n_mels=self.n_mels)



    def encode(self, x, noise_synth='filterbank'):

        #Input rep - MEL SPEC. MFCCs. MEL_ENERGY
        #CNN_Block {
            #CNN1D
            #RELU
            #BN
        # } X 3
        #Mu
        #logVar Layers


        mel_spec = self.mel_spec(x)

        # ----- INV MFCC CODE
        mel_spec_db = 20*safe_log10(mel_spec)
        mfcc = dct.dct(mel_spec_db.permute(0,2,1))
        mfcc[:,:,self.n_mfcc:] = 0
        inv_mfcc = 10**(dct.idct(mfcc) / 20).permute(0,2,1)
        mel_spec = inv_mfcc
        # ----- INV MFCC CODE END

        if(noise_synth == "ddsp"):
            mel_spec = self.norm(mel_spec.permute(0,2,1)[:,:,:])
        else:
            in_size = x.shape[-1]
            n_frames = int(in_size // self.synth_window)
            mel_spec = self.norm(mel_spec.permute(0,2,1)[:,:n_frames,:])
        
        h = self.gru(mel_spec)[0]
        mu = self.mu(h)
        logvar = self.logvar(h)

        # z of shape [bs*n_grains,z_dim]
        z = sample_from_distribution(mu, logvar)

        return z, mu, logvar

    def forward(self, audio, noise_synth="filterbank"):

        z, mu, logvar = self.encode(audio, noise_synth=noise_synth)
        # z = self.encode(audio)

        return z, mu, logvar
        # return z

# Waveform decoder consists simply of dense layers.
class SpectralDecoder_v1(nn.Module):

    """
    Decoder.

    Constructor arguments: 
        use_z : (Bool), if True, Decoder will use z as input.
        mlp_units: 512
        mlp_layers: 3
        z_units: 16
        n_harmonics: 101
        n_freq: 65
        gru_units: 512
        bidirectional: False

    input(dict(f0, z(optional), l)) : a dict object which contains key-values below
        f0 : fundamental frequency for each frame. torch.tensor w/ shape(B, time)
        z : (optional) residual information. torch.tensor w/ shape(B, time, z_units)
        loudness : torch.tensor w/ shape(B, time)

        *note dimension of z is not specified in the paper.

    output : a dict object which contains key-values below
        f0 : same as input
        c : torch.tensor w/ shape(B, time, n_harmonics) which satisfies sum(c) == 1
        a : torch.tensor w/ shape(B, time) which satisfies a > 0
        H : noise filter in frequency domain. torch.tensor w/ shape(B, frame_num, filter_coeff_length)
    """

    def __init__(self,
                    z_dim,
                    l_grain = 2048,
                    n_linears = 3,
                    h_dim = 512,
                    n_band = 2048
                    ):
        super(SpectralDecoder_v1, self).__init__()

        self.l_grain = l_grain
        # self.filter_size = l_grain//2+1
        self.filter_size = n_band
        self.z_dim = z_dim
        self.h_dim = h_dim

        decoder_linears = [linear_block(self.z_dim,self.h_dim)]
        # decoder_linears += [linear_block(h_dim,h_dim) for i in range(1,n_linears)]
        decoder_linears += [nn.Linear(self.h_dim, self.filter_size)]
        self.decoder_linears = nn.Sequential(*decoder_linears)
        self.sigmoid = nn.Sigmoid()

    def decode(self, z, n_grains=None, ola_windows=None, ola_folder=None, ola_divisor=None):

        filter_coeffs = self.decoder_linears(z)

        # What does this do??
        filter_coeffs = mod_sigmoid(filter_coeffs)
        # filter_coeffs = self.sigmoid(filter_coeffs)

        # Reshape back into the batch and grains
        filter_coeffs = filter_coeffs.reshape(-1, self.n_grains, self.filter_size)
        filter_coeffs = filter_coeffs.permute(0,2,1)

        return filter_coeffs

    def forward(self, z, n_grains=None, ola_windows=None, ola_divisor=None):

        audio = self.decode(z, n_grains=n_grains, ola_windows=ola_windows, ola_divisor=ola_divisor)

        return audio


# Waveform decoder consists simply of dense layers.
class SpectralDecoder_v2(nn.Module):

    """
    Decoder.

    Constructor arguments: 
        use_z : (Bool), if True, Decoder will use z as input.
        mlp_units: 512
        mlp_layers: 3
        z_units: 16
        n_harmonics: 101
        n_freq: 65
        gru_units: 512
        bidirectional: False

    input(dict(f0, z(optional), l)) : a dict object which contains key-values below
        f0 : fundamental frequency for each frame. torch.tensor w/ shape(B, time)
        z : (optional) residual information. torch.tensor w/ shape(B, time, z_units)
        loudness : torch.tensor w/ shape(B, time)

        *note dimension of z is not specified in the paper.

    output : a dict object which contains key-values below
        f0 : same as input
        c : torch.tensor w/ shape(B, time, n_harmonics) which satisfies sum(c) == 1
        a : torch.tensor w/ shape(B, time) which satisfies a > 0
        H : noise filter in frequency domain. torch.tensor w/ shape(B, frame_num, filter_coeff_length)
    """

    def __init__(self,
                    z_dim,
                    l_grain = 2048,
                    n_linears = 3,
                    h_dim = 512,
                    n_band = 2048,
                    noise_synth = "filterbank"
                    ):
        super(SpectralDecoder_v2, self).__init__()

        self.l_grain = l_grain
        if noise_synth == "ddsp":
            self.filter_size = l_grain//2+1
        elif noise_synth == "filterbank":
            self.filter_size = n_band
        else:
            print("ERROR! Not a valid noise synth")
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.n_bands = n_band

        # original, trating each element in the latent as a control parameter
        # in_mlps = []
        # for i in range(n_control_params):
        # for i in range(self.z_dim):
        #     in_mlps.append(mlp(1, self.h_dim, 1))
        # self.in_mlps = nn.ModuleList(in_mlps)
        # self.gru = gru(self.z_dim, self.h_dim)
        # self.out_mlp = mlp(self.h_dim + n_control_params, self.h_dim, 3)

        self.in_mlps = mlp(self.z_dim, self.h_dim, 1)
        self.gru = gru(1, self.h_dim)
        self.out_mlp = mlp(self.h_dim, self.h_dim, 3)
        self.amplitudes_layer = nn.Linear(self.h_dim, self.filter_size)

    def decode(self, z, gru_state=None, n_grains=None, ola_windows=None, ola_folder=None, ola_divisor=None):

        if gru_state==None:
            # note num layer is 1 here
            gru_state=torch.zeros(1, z.shape[0], self.h_dim).to(DEVICE)

        hidden = []
        # z = z.reshape(-1, n_grains, z.shape[1])
        # for i in range(len(self.in_mlps)):
        #     hidden.append(self.in_mlps[i](z[:,:,i].unsqueeze(-1)))
        # print("In: ", z.shape)
        hidden.append(self.in_mlps(z))
        hidden = torch.cat(hidden, dim=-1)
        # print("Cat mlps: ", hidden.shape)
        hidden, gru_state = self.gru(hidden, gru_state)
        # print(state.shape)
        # print(hidden.shape)
        # for i in range(hidden.shape[1]):
        #     print(f"{hidden[:,i,:].sum():.30}")
        # print(f"{hidden[:,0,:].sum():.30}")
        # print("GRU: ", hidden.shape)
        # Why do we use the below??
        # for i in range(self.z_dim):
            # hidden = torch.cat([hidden, z[:,:,i].unsqueeze(-1)], dim=-1)
        # hidden = torch.cat([hidden, z], dim=-1)
        # print("Control Cat: ", hidden.shape)
        hidden = self.out_mlp(hidden)
        # print("Out mlp: ", hidden.shape)
        amplitudes = self.amplitudes_layer(hidden).permute(0,2,1)
        # print("Amp layer: ", amplitudes.shape)
        amplitudes = mod_sigmoid(amplitudes)

        # print(z.shape)
        # filter_coeffs = self.decoder_linears(z)

        # # What does this do??
        # filter_coeffs = mod_sigmoid(filter_coeffs)
        # # filter_coeffs = self.sigmoid(filter_coeffs)

        # # Reshape back into the batch and grains
        # filter_coeffs = filter_coeffs.reshape(-1, self.n_grains, self.filter_size)
        # filter_coeffs = filter_coeffs.permute(0,2,1)
        # print(filter_coeffs.shape)

        # print("Amp shape: ", amplitudes.shape)
        # print(f"Amps first: {amplitudes[:,:,:].sum():.40}")


        return amplitudes, gru_state

    def forward(self, z, gru_state=None, n_grains=None, ola_windows=None, ola_divisor=None):

        amplitudes, gru_state = self.decode(z, gru_state=gru_state, n_grains=n_grains, ola_windows=ola_windows, ola_divisor=ola_divisor)

        return amplitudes, gru_state

# Waveform decoder consists simply of dense layers.
class SpectralDecoder_v3(nn.Module):

    """
    Decoder.

    Constructor arguments: 
        use_z : (Bool), if True, Decoder will use z as input.
        mlp_units: 512
        mlp_layers: 3
        z_units: 16
        n_harmonics: 101
        n_freq: 65
        gru_units: 512
        bidirectional: False

    input(dict(f0, z(optional), l)) : a dict object which contains key-values below
        f0 : fundamental frequency for each frame. torch.tensor w/ shape(B, time)
        z : (optional) residual information. torch.tensor w/ shape(B, time, z_units)
        loudness : torch.tensor w/ shape(B, time)

        *note dimension of z is not specified in the paper.

    output : a dict object which contains key-values below
        f0 : same as input
        c : torch.tensor w/ shape(B, time, n_harmonics) which satisfies sum(c) == 1
        a : torch.tensor w/ shape(B, time) which satisfies a > 0
        H : noise filter in frequency domain. torch.tensor w/ shape(B, frame_num, filter_coeff_length)
    """

    def __init__(self,
                    z_dim,
                    l_grain = 2048,
                    n_linears = 3,
                    h_dim = 512,
                    n_band = 2048,
                    noise_synth = "filterbank"
                    ):
        super(SpectralDecoder_v3, self).__init__()

        self.l_grain = l_grain
        if noise_synth == "ddsp":
            self.filter_size = l_grain//2+1
        elif noise_synth == "filterbank":
            self.filter_size = n_band
        else:
            print("ERROR! Not a valid noise synth")
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.n_bands = n_band

        # original, trating each element in the latent as a control parameter
        # in_mlps = []
        # for i in range(n_control_params):
        # for i in range(self.z_dim):
        #     in_mlps.append(mlp(1, self.h_dim, 1))
        # self.in_mlps = nn.ModuleList(in_mlps)
        # self.gru = gru(self.z_dim, self.h_dim)
        # self.out_mlp = mlp(self.h_dim + n_control_params, self.h_dim, 3)

        self.in_mlps_1 = mlp(self.z_dim, self.h_dim, 1)
        self.in_mlps_2 = mlp(self.h_dim, self.h_dim, 1)
        self.in_mlps_3 = mlp(self.h_dim, self.h_dim, 1)
        self.gru = gru(1, self.h_dim)
        self.out_mlp = mlp(self.h_dim, self.h_dim, 3)
        self.amplitudes_layer = nn.Linear(self.h_dim, self.filter_size)

    def decode(self, z, gru_state=None, n_grains=None, ola_windows=None, ola_folder=None, ola_divisor=None):

        if gru_state==None:
            # note num layer is 1 here
            gru_state=torch.zeros(1, z.shape[0], self.h_dim).to(DEVICE)

        hidden = []
        # z = z.reshape(-1, n_grains, z.shape[1])
        # for i in range(len(self.in_mlps)):
        #     hidden.append(self.in_mlps[i](z[:,:,i].unsqueeze(-1)))
        # print("In: ", z.shape)
        h = self.in_mlps_1(z)
        h = self.in_mlps_2(h)
        h = self.in_mlps_3(h)
        hidden.append(h)
        hidden = torch.cat(hidden, dim=-1)
        # print("Cat mlps: ", hidden.shape)
        hidden, gru_state = self.gru(hidden, gru_state)
        # print(state.shape)
        # print(hidden.shape)
        # for i in range(hidden.shape[1]):
        #     print(f"{hidden[:,i,:].sum():.30}")
        # print(f"{hidden[:,0,:].sum():.30}")
        # print("GRU: ", hidden.shape)
        # Why do we use the below??
        # for i in range(self.z_dim):
            # hidden = torch.cat([hidden, z[:,:,i].unsqueeze(-1)], dim=-1)
        # hidden = torch.cat([hidden, z], dim=-1)
        # print("Control Cat: ", hidden.shape)
        hidden = self.out_mlp(hidden)
        # print("Out mlp: ", hidden.shape)
        amplitudes = self.amplitudes_layer(hidden).permute(0,2,1)
        # print("Amp layer: ", amplitudes.shape)
        amplitudes = mod_sigmoid(amplitudes)

        # print(z.shape)
        # filter_coeffs = self.decoder_linears(z)

        # # What does this do??
        # filter_coeffs = mod_sigmoid(filter_coeffs)
        # # filter_coeffs = self.sigmoid(filter_coeffs)

        # # Reshape back into the batch and grains
        # filter_coeffs = filter_coeffs.reshape(-1, self.n_grains, self.filter_size)
        # filter_coeffs = filter_coeffs.permute(0,2,1)
        # print(filter_coeffs.shape)

        # print("Amp shape: ", amplitudes.shape)
        # print(f"Amps first: {amplitudes[:,:,:].sum():.40}")


        return amplitudes, gru_state

    def forward(self, z, gru_state=None, n_grains=None, ola_windows=None, ola_divisor=None):

        amplitudes, gru_state = self.decode(z, gru_state=gru_state, n_grains=n_grains, ola_windows=ola_windows, ola_divisor=ola_divisor)

        return amplitudes, gru_state
    
class SpectralVAE_v1(nn.Module):

    def __init__(self,
                    l_grain=2048,                    
                    n_linears=3,
                    z_dim = 128,
                    h_dim=512,
                    n_band = 2048,
                    linear_min_f = 20, 
                    linear_max_f_cutoff_fs = 4,
                    fs = 44100,
                    filterbank_attenuation=50,
                    min_noise_len = 2**16,
                    normalize_noise_bands=True,
                    synth_window=32,
                    mfcc_hop_size = 32,
                    noise_synth = "filterbank"
                    ):
        super(SpectralVAE_v1, self).__init__()

        self.z_dim = z_dim
        self.l_grain = l_grain
        self.synth_window = synth_window
        self.mfcc_hop_size = mfcc_hop_size
        self.noise_synth = noise_synth

        fb  = FilterBank(n_filters_linear = n_band//2, n_filters_log = n_band//2, linear_min_f = linear_min_f, linear_max_f_cutoff_fs = linear_max_f_cutoff_fs,  fs = fs, attenuation = filterbank_attenuation)
        self.center_frequencies = fb.band_centers #store center frequencies for reference
        self.noise_bands, self.noise_len = get_noise_bands(fb=fb, min_noise_len=min_noise_len, normalize=normalize_noise_bands)

        # Encoder and decoder components
        self.Encoder = SpectralEncoder_v2(
                        l_grain = l_grain,
                        z_dim = z_dim,
                        h_dim = h_dim,
                        synth_window=synth_window,
                        # mfcc_hop_size = mfcc_hop_size,
                    )
        self.Decoder = SpectralDecoder_v3(
                        l_grain = l_grain,
                        z_dim = z_dim,
                        h_dim = h_dim,
                        n_linears = n_linears,
                        n_band=n_band,
                        noise_synth=noise_synth,

                    )
    def ddsp_noise_synth(self, amplitudes, fir_taps=-1):
        """ Apply predicted amplitudes to DDSP noise synthesizer
        
        """
        signal_length = self.mfcc_hop_size * (amplitudes.shape[-1]-1)
            
        amplitudes = amplitudes.permute(0,2,1)

        impulse = amp_to_impulse_response(amplitudes, fir_taps)
        #generate a noise signal of full length
        noise = torch.rand(
            impulse.shape[0],
            signal_length,
        ).to(impulse) * 2 - 1

        noise = fft_convolve_ddsp(noise, impulse).contiguous()

        return noise

    def ddsp_noise_synth_min_phase(self, amplitudes, fir_taps=-1):
        """ Apply predicted amplitudes to DDSP noise synthesizer

        """
        signal_length = self.mfcc_hop_size * (amplitudes.shape[-1]-1)

        amplitudes = amplitudes.permute(0,2,1)

        impulse = amp_to_impulse_response(amplitudes, fir_taps)


        #generate a noise signal of full length
        noise = torch.rand(
            impulse.shape[0],
            signal_length,
        ).to(impulse) * 2 - 1


        # -------  Look at adding this in!! -----------
        # TODO NEW
        frame_size = self.l_grain
        n_frames = impulse.shape[1]
        min_phase_spec, min_phase_fir = minimum_phase(impulse, frame_size)

        noise = frame(noise, frame_size, self.mfcc_hop_size, pad_end=True) 
        noise = noise.reshape(noise.shape[0],noise.shape[1], frame_size)
        # print(noise.shape)
        # print(min_phase_fir.shape)
        # print(img)

        audio = fft_convolve(noise, min_phase_fir)
        audio = audio.reshape(-1,n_frames,frame_size)

        #7 Apply same window as previously to newly filtered noise
        # DO we need this?
        ola_window = signal.hann(frame_size,sym=False)
        ola_windows = torch.from_numpy(ola_window).unsqueeze(0).repeat(n_frames,1).type(torch.float32).to(DEVICE)
        ola_windows[0,:frame_size//2] = ola_window[frame_size//2] # start of 1st grain is not windowed for preserving attacks
        ola_windows[-1,frame_size//2:] = ola_window[frame_size//2] # end of last grain is not wondowed to preserving decays
        ola_windows = ola_windows
        audio = audio*(ola_windows.unsqueeze(0).repeat(impulse.shape[0],1,1))
        # Apply simply normalisation
        audio = audio/torch.max(audio)

        #8 Overlap Add
        # Overlap add folder, folds and reshapes audio into target dimensions, so [bs, tar_l]
        # This is essentially folding and adding the overlapping grains back into original sample size, based on the given hop size.
        # Note that shape is changed here, so that input tensor is of chape [bs, channel X (kernel_size), L],
        # since kernel size is l_grain, this is needed in the second dimension.
        ola_folder = torch.nn.Fold((signal_length+frame_size,1),(frame_size,1),stride=(self.mfcc_hop_size,1))
        audio_sum = ola_folder(audio.permute(0,2,1)).squeeze().unsqueeze(0)

        # Normalise the energy values across the audio samples
        if NORMALIZE_OLA:
            # Normalises based on number of overlapping frames used in folding per point in time.
            unfolder = torch.nn.Unfold((frame_size,1),stride=(self.mfcc_hop_size,1))
            input_ones = torch.ones(1,1,signal_length+frame_size,1)
            ola_divisor = ola_folder(unfolder(input_ones)).squeeze()
            ola_divisor = ola_divisor.to(DEVICE)
            audio_sum = audio_sum/ola_divisor.unsqueeze(0).repeat(impulse.shape[0],1)
        else:
            audio_sum = audio_sum.unsqueeze(0)

        # print(audio_sum.shape)
        # print(img)
        # print(audio_sum.shape)
        audio_sum = audio_sum.squeeze()

        return audio_sum[:,:signal_length]
    
    def ddsp_noise_synth_original(self, amplitudes, fir_taps=-1):
        """ Apply predicted amplitudes to DDSP noise synthesizer

        """
        signal_length = self.mfcc_hop_size * (amplitudes.shape[-1]-1)
        # signal_length = amplitudes.shape[-1]*self.synth_window
        # signal_length = 24000

        amplitudes = amplitudes.permute(0,2,1)

        impulse = amp_to_impulse_response(amplitudes, fir_taps)

        # -------  Look at adding this in!! -----------
        # TODO NEW
        frame_size = self.l_grain
        n_frames = impulse.shape[1]

        # TESTING Noise filtering
        # Noise filtering (Maybe try using the new noise filtering function and compare to current method...)
        filter_window = torch.nn.Parameter(torch.fft.fftshift(torch.hann_window(frame_size)),requires_grad=False).to(DEVICE)
        audio = noise_filtering(amplitudes, filter_window, n_frames, frame_size, HOP_SIZE_RATIO)

        #7 Apply same window as previously to newly filtered noise
        # DO we need this?
        ola_window = signal.hann(frame_size,sym=False)
        ola_windows = torch.from_numpy(ola_window).unsqueeze(0).repeat(n_frames,1).type(torch.float32).to(DEVICE)
        ola_windows[0,:frame_size//2] = ola_window[frame_size//2] # start of 1st grain is not windowed for preserving attacks
        ola_windows[-1,frame_size//2:] = ola_window[frame_size//2] # end of last grain is not wondowed to preserving decays
        ola_windows = ola_windows
        audio = audio*(ola_windows.unsqueeze(0).repeat(impulse.shape[0],1,1))
        # Apply simply normalisation
        audio = audio/torch.max(audio)

        #8 Overlap Add
        # Overlap add folder, folds and reshapes audio into target dimensions, so [bs, tar_l]
        # This is essentially folding and adding the overlapping grains back into original sample size, based on the given hop size.
        # Note that shape is changed here, so that input tensor is of chape [bs, channel X (kernel_size), L],
        # since kernel size is l_grain, this is needed in the second dimension.
        ola_folder = torch.nn.Fold((signal_length+frame_size,1),(frame_size,1),stride=(self.mfcc_hop_size,1))
        audio_sum = ola_folder(audio.permute(0,2,1)).squeeze().unsqueeze(0)

        # Normalise the energy values across the audio samples
        if NORMALIZE_OLA:
            # Normalises based on number of overlapping frames used in folding per point in time.
            unfolder = torch.nn.Unfold((frame_size,1),stride=(self.mfcc_hop_size,1))
            input_ones = torch.ones(1,1,signal_length+frame_size,1)
            ola_divisor = ola_folder(unfolder(input_ones)).squeeze()
            ola_divisor = ola_divisor.to(DEVICE)
            audio_sum = audio_sum/ola_divisor.unsqueeze(0).repeat(impulse.shape[0],1)
        else:
            audio_sum = audio_sum.unsqueeze(0)

        # print(audio_sum.shape)
        # print(img)
        # print(audio_sum.shape)
        # audio_sum = audio_sum.squeeze()
        # print(audio_sum.shape)
        return audio_sum[:,:signal_length]

    def synth_batch(self, amplitudes, noise_index=0):
        """Apply the predicted amplitudes to the noise bands.
        Args:
        ----------
        amplitudes : torch.Tensor
            Predicted amplitudes

        Returns:
        ----------  
        signal : torch.Tensor
            Output audio signal
        """
        noise_index = noise_index * self.synth_window
        #synth in noise_len frames to fit longer sequences on GPU memory
        frame_len = int(self.noise_len/self.synth_window)
        n_frames = math.ceil(amplitudes.shape[-1]/frame_len)
        self.noise_bands = self.noise_bands.to(amplitudes.device)
        #avoid overfitting to noise values
        self.noise_bands = torch.roll(self.noise_bands, shifts=int(torch.randint(low=0, high=self.noise_bands.shape[-1], size=(1,))), dims=-1)
        signal_len = amplitudes.shape[-1]*self.synth_window
        # signal_len = 65536
        # print(amplitudes.shape)
        # print(signal_len)
        # print(img)
        #smaller amp len than noise_len
        # if amplitudes.shape[-1]/frame_len <= 1:
        if amplitudes.shape[-1]/frame_len < 1:
            # print(amplitudes.shape)
            # scale_factor = 65536 / amplitudes.shape[-1]
            # upsampled_amplitudes = F.interpolate(amplitudes, scale_factor=scale_factor, mode='linear')
            # upsampled_amplitudes = F.interpolate(amplitudes, scale_factor=self.synth_window, mode='linear')
            upsampled_amplitudes = F.interpolate(amplitudes, scale_factor=self.synth_window, mode='nearest')
            # print(f"Upsampled amps: {upsampled_amplitudes.shape}")
            # print(f"Up Amps Sum: {upsampled_amplitudes[:,:,:32].sum():.16}")
            signal = (self.noise_bands[..., noise_index:noise_index+signal_len]*upsampled_amplitudes).sum(1, keepdim=True)
        else:
            for i in range(n_frames):
                if i == 0:
                    upsampled_amplitudes = F.interpolate(amplitudes[..., :frame_len], scale_factor=self.synth_window, mode='linear')
                    signal = (self.noise_bands*upsampled_amplitudes).sum(1, keepdim=True)
                #last iteration
                elif i == (n_frames-1):
                    upsampled_amplitudes = F.interpolate(amplitudes[..., i*frame_len:], scale_factor=self.synth_window, mode='linear')
                    signal = torch.cat([signal, (self.noise_bands[...,:upsampled_amplitudes.shape[-1]]*upsampled_amplitudes).sum(1, keepdim=True)], dim=-1)
                else:
                    upsampled_amplitudes = F.interpolate(amplitudes[..., i*frame_len:(i+1)*frame_len], scale_factor=self.synth_window, mode='linear')
                    signal = torch.cat([signal, (self.noise_bands*upsampled_amplitudes).sum(1, keepdim=True)], dim=-1)

        # Remove the extra dimension.
        return signal.reshape(signal.shape[0], signal.shape[2])
        # return signal 

        # Number of convolutional layers
    def encode(self, x, noise_synth="filterbank"):

        # x ---> z
        z, mu, log_variance = self.Encoder(x, noise_synth=noise_synth);
    
        return {"z":z, "mu":mu, "logvar":log_variance} 

    def decode(self, z, gru_state=None, noise_index=0, noise_synth="filterbank", fir_taps=-1):

        amplitudes, gru_state = self.Decoder(z, gru_state=gru_state)
        if(noise_synth=="ddsp"):
            signal = self.ddsp_noise_synth(amplitudes=amplitudes)
            # signal = self.ddsp_noise_synth_min_phase(amplitudes=amplitudes, fir_taps=fir_taps)
            # signal = self.ddsp_noise_synth_original(amplitudes=amplitudes)
            # signal = self.ddsp_noise_synth_original(amplitudes=amplitudes, fir_taps=fir_taps)
        else:
            signal = self.synth_batch(amplitudes=amplitudes,noise_index=noise_index)
            
        return {"audio":signal, "gru_state":gru_state}

    def forward(self, x, sampling=True, noise_synth="filterbank"):

        # x ---> z
        
        z, mu, log_variance = self.Encoder(x, noise_synth=noise_synth);

        # z = z + 0.1

        # z ---> x_hat
        # Note in paper they also have option passing mu into the decoder and not z
        if sampling:
            amplitudes, _ = self.Decoder(z)
            if(noise_synth=="ddsp"):
                signal = self.ddsp_noise_synth(amplitudes=amplitudes)
                # signal = self.ddsp_noise_synth_min_phase(amplitudes=amplitudes)
                # signal = self.ddsp_noise_synth_original(amplitudes=amplitudes)
            else:
                signal = self.synth_batch(amplitudes=amplitudes)

        else:
            amplitudes, _ = self.Decoder(mu)
            if(noise_synth=="ddsp"):
                signal = self.ddsp_noise_synth(amplitudes=amplitudes)
                # signal = self.ddsp_noise_synth_min_phase(amplitudes=amplitudes)
                # signal = self.ddsp_noise_synth_original(amplitudes=amplitudes)
            else:
                signal = self.synth_batch(amplitudes=amplitudes)

        return signal, z, mu, log_variance
