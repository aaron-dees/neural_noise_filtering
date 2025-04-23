import torch
import matplotlib.pyplot as plt
import os
import seaborn as sns
import soundfile as sf
from sklearn.decomposition import PCA
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
import torchaudio
import librosa
from scipy import fft, signal
import torch_dct as dct
# from dsp_components import amp_to_impulse_response_w_phase
import utils.dsp_components as dsp
import heapq

# TODO Should really pass these as variables to be safer
from scripts.configs.noise_filtering_config import NORMALIZE_OLA, RECONSTRUCTION_SAVE_DIR, SAMPLE_RATE, NOISE_SYNTH
# from scripts.configs.hyper_parameters_waveform import NORMALIZE_OLA, RECONSTRUCTION_SAVE_DIR, SAMPLE_RATE


# Sample from a gaussian distribution
def sample_from_distribution(mu, log_variance):

    # point = mu + sigma*sample(N(0,1))
    
    std = torch.exp(log_variance * 0.5)
    # epsilon = torch.normal(torch.zeros(shape).to(device), torch.ones(shape).to(device))
    epsilon = torch.randn_like(std)
    sampled_point = mu + std * epsilon

    return sampled_point

# Show the latent space
def show_latent_space(latent_representations, sample_labels):
    plt.figure(figsize=(10,10))
    plt.scatter(latent_representations[:, 0],
        latent_representations[:, 1],
        cmap="rainbow",
        c = sample_labels,
        alpha = 0.5,
        s = 2)
    plt.colorbar
    plt.savefig("laetnt_rep.png") 

def show_image_comparisons(images, x_hat):

    fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(25,4))
            
    # input images on top row, reconstructions on bottom
    for images, row in zip([images, x_hat], axes):
        for img, ax in zip(images, row):
            ax.imshow(np.squeeze(img), cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.savefig("comparisons.png")

def plot_latents(train_latents,train_labels, classes,export_dir):
    print(train_labels)
    if os.path.exists(export_dir) is False:
        os.makedirs(export_dir)
    n_grains = train_latents.shape[1]
    z_dim = train_latents.shape[2]
    train_latents = train_latents.view(-1,z_dim).numpy()
    train_labels = train_labels.unsqueeze(-1).repeat(1,n_grains).view(-1).numpy().astype(str)
    for i,c in enumerate(classes):
        train_labels[np.where(train_labels==str(i))] = c
    pca = PCA(n_components=2)
    pca.fit(train_latents)
    train_latents = pca.transform(train_latents)
    print(f'PCA Shape: {train_latents.shape}')
    # TODO: shuffle samples for better plotting
    sns.scatterplot(x=train_latents[:,0], y=train_latents[:,1], hue=train_labels, s=4)
    plt.plot(train_latents[:,0], train_latents[:,1], color="green", linestyle='dashed', marker='o')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(export_dir,"latent_scatter_trainset.pdf"))
    plt.close("all")

# Compute the latens
def compute_latents(w_model, dataloader, batch_size, device):
    tmploader = torch.utils.data.DataLoader(dataloader.dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    dataset_latents = []
    # dataset_labels = []
    for waveform in tmploader:
        with torch.no_grad():
            waveform = waveform.to(device)

            # ---------- Run Model ----------
            mu = w_model.encode(waveform, noise_synth=NOISE_SYNTH)["mu"].cpu()
            dataset_latents.append(mu)
            # dataset_labels.append(labels)
    dataset_latents = torch.cat(dataset_latents,0)
    # dataset_labels = torch.cat(dataset_labels,0)
    # labels not so important now, but will be in future
    # print("--- Exported dataset sizes:\t",dataset_latents.shape,dataset_labels.shape)
    print("--- Exported dataset sizes:\t",dataset_latents.shape)
    return dataset_latents
    # return dataset_latents,dataset_labels

# Export the latents
def export_latents(w_model, train_dataloader, val_dataloader, batch_size, device):
    train_latents = compute_latents(w_model,train_dataloader, batch_size, device)
    test_latents= compute_latents(w_model,val_dataloader, batch_size, device)
    return train_latents,test_latents

# Safe log for cases where x is very close to zero
def safe_log(x, eps=1e-7):
    return torch.log(x + eps)

def init_beta(max_steps,tar_beta,beta_steps=1000, warmup_perc=0.1):
    # if continue_training:
    #     beta = tar_beta
    #     print("\n*** setting fixed beta of ",beta)
    # else:
    # warmup wihtout increasing beta
    warmup_start = int(warmup_perc*max_steps)
    # set beta steps to only increase of half of max steps
    beta_step_size = int(max_steps/2/beta_steps)
    beta_step_val = tar_beta/beta_steps
    beta = 0
    print("--- Initialising Beta, from 0 to ", tar_beta)
    print("")
    print('--- Beta: {}'.format(beta),
            '\tWarmup Start: {}'.format(warmup_start),
            '\tStep Size: {}'.format(beta_step_size),
            '\tStep Val: {:.5f}'.format(beta_step_val))
        
    return beta, beta_step_val, beta_step_size, warmup_start

def export_embedding_to_audio_reconstructions(l_model,w_model,batch, export_dir, sr, device, hop_size, tar_l, hop_size_ratio,trainset=False):
    if os.path.exists(export_dir) is False:
        os.makedirs(export_dir)

    mse_loss = nn.MSELoss()
    with torch.no_grad():
        z,conds = batch
        z,conds = z.to(device),conds.to(device)
        # forward through latent embedding
        z_hat, e, mu, log_variance = l_model(z,conds, sampling=False)
        
        rec_loss = mse_loss(z_hat,z) # we train with a deterministic output
        print("Latent Reconstruction Loss: ", rec_loss)

        # reshape as minibatch of individual grains of shape [bs*n_grains,z_dim]
        z,z_hat = z.reshape(-1,w_model.z_dim),z_hat.reshape(-1,w_model.z_dim)
        # export reconstruction by pretrained waveform model and by embedding + waveform models
        x,x_hat = w_model.decode(z),w_model.decode(z_hat)
        x = x['audio']
        x_hat = x_hat['audio']

        # Need to be put back together
        # ---------- Run Model END ----------

        # ---------- Noise Filtering ----------

        # Reshape for noise filtering - TODO Look if this is necesary
        x = x.reshape(x.shape[0]*x.shape[1],x.shape[2])
        x_hat = x_hat.reshape(x_hat.shape[0]*x_hat.shape[1],x_hat.shape[2])

        # Noise filtering (Maybe try using the new noise filtering function and compare to current method...)
        filter_window = nn.Parameter(torch.fft.fftshift(torch.hann_window(w_model.l_grain)),requires_grad=False).to(device)
        audio = dsp.noise_filtering(x, filter_window, w_model.n_grains, w_model.l_grain, hop_size_ratio)
        audio_hat = dsp.noise_filtering(x_hat, filter_window, w_model.n_grains, w_model.l_grain, hop_size_ratio)
        # ---------- Noise Filtering END ----------

        # ---------- Concatonate Grains ----------

        # Check if number of grains wanted is entered, else use the original
        if w_model.n_grains is None:
            audio = audio.reshape(-1, w_model.n_grains, w_model.l_grain)
            audio_hat = audio_hat.reshape(-1, w_model.n_grains, w_model.l_grain)
        else:
            audio = audio.reshape(-1,w_model.n_grains,w_model.l_grain)
            audio_hat = audio_hat.reshape(-1,w_model.n_grains,w_model.l_grain)
        bs = audio.shape[0]

        ola_window = signal.hann(w_model.l_grain,sym=False)
        ola_windows = torch.from_numpy(ola_window).unsqueeze(0).repeat(w_model.n_grains,1).type(torch.float32)
        ola_windows[0,:w_model.l_grain//2] = ola_window[w_model.l_grain//2] # start of 1st grain is not windowed for preserving attacks
        ola_windows[-1,w_model.l_grain//2:] = ola_window[w_model.l_grain//2] # end of last grain is not wondowed to preserving decays
        ola_windows = nn.Parameter(ola_windows,requires_grad=False).to(device)

        # Check if an overlapp add window has been passed, if not use that used in encoding.
        audio = audio*(ola_windows.unsqueeze(0).repeat(bs,1,1))
        audio_hat = audio_hat*(ola_windows.unsqueeze(0).repeat(bs,1,1))

        # Folder
        # Folds input tensor into shape [bs, channels, tar_l, 1], using a kernel size of l_grain, and stride of hop_size
        # can see doc here, https://pytorch.org/docs/stable/generated/torch.nn.Fold.html
        ola_folder = nn.Fold((tar_l,1),(w_model.l_grain,1),stride=(hop_size,1))
        # Overlap add folder, folds and reshapes audio into target dimensions, so [bs, tar_l]
        # This is essentially folding and adding the overlapping grains back into original sample size, based on the given hop size.
        # Note that shape is changed here, so that input tensor is of chape [bs, channel X (kernel_size), L],
        # since kernel size is l_grain, this is needed in the second dimension.
        audio_sum = ola_folder(audio.permute(0,2,1)).squeeze()
        audio_hat_sum = ola_folder(audio_hat.permute(0,2,1)).squeeze()

        # Normalise the energy values across the audio samples
        if NORMALIZE_OLA:
            unfolder = nn.Unfold((w_model.l_grain,1),stride=(hop_size,1))
            input_ones = torch.ones(1,1,tar_l,1)
            ola_divisor = ola_folder(unfolder(input_ones)).squeeze()
            ola_divisor = nn.Parameter(ola_divisor,requires_grad=False).to(device)
            audio_sum = audio_sum/ola_divisor.unsqueeze(0).repeat(bs,1)
            audio_hat_sum = audio_hat_sum/ola_divisor.unsqueeze(0).repeat(bs,1)

        #audio_export = torch.cat((audio,audio_hat),-1).cpu().numpy()
        for i in range(audio_hat_sum.shape[0]):
            if trainset:
                sf.write(os.path.join(export_dir,"embedding_to_audio_train_reconstruction_orig_"+str(i)+".wav"),audio_sum[i,:], sr)
                sf.write(os.path.join(export_dir,"embedding_to_audio_train_reconstruction_hat_"+str(i)+".wav"),audio_hat_sum[i,:], sr)
                sf.write(os.path.join(export_dir,"waveformmodel_audio/embedding_to_audio_train_reconstruction_orig_"+str(i)+".wav"),audio_sum[i,:], sr)
                sf.write(os.path.join(export_dir,"latentmodel_audio/embedding_to_audio_train_reconstruction_hat_"+str(i)+".wav"),audio_hat_sum[i,:], sr)
            else:
                sf.write(os.path.join(export_dir,"embedding_to_audio_test_reconstruction_orig_"+str(i)+".wav"),audio_sum[i,:], sr)
                sf.write(os.path.join(export_dir,"embedding_to_audio_test_reconstruction_hat_"+str(i)+".wav"),audio_hat_sum[i,:], sr)
                sf.write(os.path.join(export_dir,"waveformmodel_audio/embedding_to_audio_test_reconstruction_orig_"+str(i)+".wav"),audio_sum[i,:], sr)
                sf.write(os.path.join(export_dir,"latentmodel_audio/embedding_to_audio_test_reconstruction_hat_"+str(i)+".wav"),audio_hat_sum[i,:], sr)

def latent_to_audio(latent ,latent_hat,w_model, export_dir, sr, device, hop_size, tar_l, hop_size_ratio,trainset=False):
    if os.path.exists(export_dir) is False:
        os.makedirs(export_dir)

    with torch.no_grad():

        latent, latent_hat = latent.to(device), latent_hat.to(device) 

        x = w_model.decode(latent)
        x_hat = w_model.decode(latent_hat)
        x = x['audio']
        x_hat = x_hat['audio']

        # Need to be put back together
        # ---------- Run Model END ----------

        # ---------- Noise Filtering ----------

        # Reshape for noise filtering - TODO Look if this is necesary
        x = x.reshape(x.shape[0]*x.shape[1],x.shape[2])
        x_hat = x_hat.reshape(x_hat.shape[0]*x_hat.shape[1],x_hat.shape[2])

        # Noise filtering (Maybe try using the new noise filtering function and compare to current method...)
        filter_window = nn.Parameter(torch.fft.fftshift(torch.hann_window(w_model.l_grain)),requires_grad=False).to(device)
        audio = dsp.noise_filtering(x, filter_window, w_model.n_grains, w_model.l_grain, hop_size_ratio)
        audio_hat = dsp.noise_filtering(x_hat, filter_window, w_model.n_grains, w_model.l_grain, hop_size_ratio)
        # ---------- Noise Filtering END ----------

        # ---------- Concatonate Grains ----------

        # Check if number of grains wanted is entered, else use the original
        if w_model.n_grains is None:
            audio = audio.reshape(-1, w_model.n_grains, w_model.l_grain)
            audio_hat = audio_hat.reshape(-1, w_model.n_grains, w_model.l_grain)
        else:
            audio = audio.reshape(-1,w_model.n_grains,w_model.l_grain)
            audio_hat = audio_hat.reshape(-1,w_model.n_grains,w_model.l_grain)
        bs = audio.shape[0]

        ola_window = signal.hann(w_model.l_grain,sym=False)
        ola_windows = torch.from_numpy(ola_window).unsqueeze(0).repeat(w_model.n_grains,1).type(torch.float32)
        ola_windows[0,:w_model.l_grain//2] = ola_window[w_model.l_grain//2] # start of 1st grain is not windowed for preserving attacks
        ola_windows[-1,w_model.l_grain//2:] = ola_window[w_model.l_grain//2] # end of last grain is not wondowed to preserving decays
        ola_windows = nn.Parameter(ola_windows,requires_grad=False).to(device)

        # Check if an overlapp add window has been passed, if not use that used in encoding.
        audio = audio*(ola_windows.unsqueeze(0).repeat(bs,1,1))
        audio_hat = audio_hat*(ola_windows.unsqueeze(0).repeat(bs,1,1))

        # Folder
        # Folds input tensor into shape [bs, channels, tar_l, 1], using a kernel size of l_grain, and stride of hop_size
        # can see doc here, https://pytorch.org/docs/stable/generated/torch.nn.Fold.html
        ola_folder = nn.Fold((tar_l,1),(w_model.l_grain,1),stride=(hop_size,1))
        # Overlap add folder, folds and reshapes audio into target dimensions, so [bs, tar_l]
        # This is essentially folding and adding the overlapping grains back into original sample size, based on the given hop size.
        # Note that shape is changed here, so that input tensor is of chape [bs, channel X (kernel_size), L],
        # since kernel size is l_grain, this is needed in the second dimension.
        audio_sum = ola_folder(audio.permute(0,2,1)).squeeze()
        audio_hat_sum = ola_folder(audio_hat.permute(0,2,1)).squeeze()

        # Normalise the energy values across the audio samples
        if NORMALIZE_OLA:
            unfolder = nn.Unfold((w_model.l_grain,1),stride=(hop_size,1))
            input_ones = torch.ones(1,1,tar_l,1)
            ola_divisor = ola_folder(unfolder(input_ones)).squeeze()
            ola_divisor = nn.Parameter(ola_divisor,requires_grad=False).to(device)
            audio_sum = audio_sum/ola_divisor.unsqueeze(0).repeat(bs,1)
            audio_hat_sum = audio_hat_sum/ola_divisor.unsqueeze(0).repeat(bs,1)

        #audio_export = torch.cat((audio,audio_hat),-1).cpu().numpy()
        for i in range(audio_hat_sum.shape[0]):
            if trainset:
                sf.write(os.path.join(export_dir,"embedding_to_audio_train_reconstruction_orig_"+str(i)+".wav"),audio_sum[i,:].cpu().numpy(), sr)
                sf.write(os.path.join(export_dir,"embedding_to_audio_train_reconstruction_hat_"+str(i)+".wav"),audio_hat_sum[i,:].cpu().numpy(), sr)
                sf.write(os.path.join(export_dir,"waveformmodel_audio/embedding_to_audio_train_reconstruction_orig_"+str(i)+".wav"),audio_sum[i,:].cpu().numpy(), sr)
                sf.write(os.path.join(export_dir,"latentmodel_audio/embedding_to_audio_train_reconstruction_hat_"+str(i)+".wav"),audio_hat_sum[i,:].cpu().numpy(), sr)
            else:
                sf.write(os.path.join(export_dir,"embedding_to_audio_test_reconstruction_orig_"+str(i)+".wav"),audio_sum[i,:].cpu().numpy(), sr)
                sf.write(os.path.join(export_dir,"embedding_to_audio_test_reconstruction_hat_"+str(i)+".wav"),audio_hat_sum[i,:].cpu().numpy(), sr)
                sf.write(os.path.join(export_dir,"waveformmodel_audio/embedding_to_audio_test_reconstruction_orig_"+str(i)+".wav"),audio_sum[i,:].cpu().numpy(), sr)
                sf.write(os.path.join(export_dir,"latentmodel_audio/embedding_to_audio_test_reconstruction_hat_"+str(i)+".wav"),audio_hat_sum[i,:].cpu().numpy(), sr)

def export_random_samples(l_model,w_model,export_dir, z_dim, e_dim, sr, classes, device, tar_l, hop_size, hop_size_ratio, n_samples=10,temperature=1.):
    if os.path.exists(export_dir) is False:
        os.makedirs(export_dir)
    with torch.no_grad():
        for i,cl in enumerate(classes):
            rand_e = torch.randn((n_samples, e_dim)).to(device)
            rand_e = rand_e*temperature
            conds = torch.zeros(n_samples).to(device).long()+i
            z_hat = l_model.decode(rand_e,conds).reshape(-1, z_dim)
            # x_hat = w_model.decode(z_hat).view(-1).cpu().numpy()
            # x_hat = w_model.decode(z_hat).view(-1)
            x_hat = w_model.decode(z_hat)

            x_hat = x_hat['audio']

            # Need to be put back together
            # ---------- Run Model END ----------

            # ---------- Noise Filtering ----------

            # Reshape for noise filtering - TODO Look if this is necesary
            x_hat = x_hat.reshape(x_hat.shape[0]*x_hat.shape[1],x_hat.shape[2])

            # Noise filtering (Maybe try using the new noise filtering function and compare to current method...)
            filter_window = nn.Parameter(torch.fft.fftshift(torch.hann_window(w_model.l_grain)),requires_grad=False).to(device)
            audio_hat = dsp.noise_filtering(x_hat, filter_window, w_model.n_grains, w_model.l_grain, hop_size_ratio)
            # ---------- Noise Filtering END ----------

            # ---------- Concatonate Grains ----------

            # Check if number of grains wanted is entered, else use the original
            if w_model.n_grains is None:
                audio_hat = audio_hat.reshape(-1, w_model.n_grains, w_model.l_grain)
            else:
                audio_hat = audio_hat.reshape(-1,w_model.n_grains,w_model.l_grain)
            bs = audio_hat.shape[0]

            ola_window = signal.hann(w_model.l_grain,sym=False)
            ola_windows = torch.from_numpy(ola_window).unsqueeze(0).repeat(w_model.n_grains,1).type(torch.float32)
            ola_windows[0,:w_model.l_grain//2] = ola_window[w_model.l_grain//2] # start of 1st grain is not windowed for preserving attacks
            ola_windows[-1,w_model.l_grain//2:] = ola_window[w_model.l_grain//2] # end of last grain is not wondowed to preserving decays
            ola_windows = nn.Parameter(ola_windows,requires_grad=False).to(device)

            # Check if an overlapp add window has been passed, if not use that used in encoding.
            audio_hat = audio_hat*(ola_windows.unsqueeze(0).repeat(bs,1,1))

            # Folder
            # Folds input tensor into shape [bs, channels, tar_l, 1], using a kernel size of l_grain, and stride of hop_size
            # can see doc here, https://pytorch.org/docs/stable/generated/torch.nn.Fold.html
            ola_folder = nn.Fold((tar_l,1),(w_model.l_grain,1),stride=(hop_size,1))
            # Overlap add folder, folds and reshapes audio into target dimensions, so [bs, tar_l]
            # This is essentially folding and adding the overlapping grains back into original sample size, based on the given hop size.
            # Note that shape is changed here, so that input tensor is of chape [bs, channel X (kernel_size), L],
            # since kernel size is l_grain, this is needed in the second dimension.
            audio_hat_sum = ola_folder(audio_hat.permute(0,2,1)).squeeze()

            # Normalise the energy values across the audio samples
            if NORMALIZE_OLA:
                unfolder = nn.Unfold((w_model.l_grain,1),stride=(hop_size,1))
                input_ones = torch.ones(1,1,tar_l,1)
                ola_divisor = ola_folder(unfolder(input_ones)).squeeze()
                ola_divisor = nn.Parameter(ola_divisor,requires_grad=False).to(device)
                audio_hat_sum = audio_hat_sum/ola_divisor.unsqueeze(0).repeat(bs,1)

            for i in range(audio_hat_sum.shape[0]):
                sf.write(os.path.join(export_dir,f"random_samples_"+str(i)+".wav"),audio_hat_sum[i,:], sr)

def generate_noise_grains(batch_size, n_grains, l_grain, dtype, device, hop_ratio=0.25):

    tar_l = int(((n_grains+3)/4)*l_grain)

    noise = torch.rand(batch_size, tar_l, dtype=dtype, device=device)*2-1
    # TEST using a slightly differently scaled noise
    # noise = torch.rand(batch_size, tar_l, dtype=dtype, device=device)*0.6-0.3

    hop_size = int(hop_ratio*l_grain)

    new_noise = noise[:, 0:l_grain].unsqueeze(1)
    for i in range(1, n_grains):  
        starting_point = i*hop_size
        ending_point = starting_point+l_grain
        tmp_noise = noise[:, starting_point:ending_point].unsqueeze(1)
        new_noise = torch.cat((new_noise, tmp_noise), dim = 1)

    return new_noise

# TODO Try generating all the niose and splitting it into the grains after using (hop_ratio * ((l_grain/2) + 1 )), or something like that?.
def generate_noise_grains_freq(batch_size, n_grains, l_grain, dtype, device, hop_ratio=0.25):

    # tar_l = int((((n_grains+3)/4)*l_grain) / 2) + 1

    # noise_fft = torch.rand(batch_size, tar_l, dtype=dtype, device=device)*2-1
    # noise_fft = torch.rand(batch_size, tar_l, dtype=torch.cfloat, device=device)*2-1
    # print(noise_fft.real.min())
    # print(noise_fft.real.max())
    # plt.plot(noise_fft[0,513:1026])
    # # plt.plot(torch.rand(513))
    # # plt.plot(inv_cepstral_coeff[7])
    # # plt.plot(noise[7])
    # plt.savefig("test_freq.png")
    # print("Done printing")

    # print("Noise shape: ", torch.fft.fft(noise).shape)

    # noise = torch.fft.irfft(noise_fft)

    # print(torch.fft.rfft(noise).real.min())
    # print(torch.fft.rfft(noise).real.max())

    # plt.plot(torch.fft.rfft(noise)[0,513:1026])
    # # plt.plot(torch.rand(513))
    # # plt.plot(inv_cepstral_coeff[7])
    # plt.plot(noise[0, 513:1026])
    # plt.savefig("test_freq.png")
    # print("Done printing")

    # print("Noise Shape: ", noise.shape)

    # TEST using a slightly differently scaled noise
    # noise = torch.rand(batch_size, tar_l, dtype=dtype, device=device)*0.6-0.3

    hop_size = int(hop_ratio*l_grain)

    # new_noise = noise[:, 0:l_grain].unsqueeze(1)
    # print(new_noise.shape)
    new_noise_fft = (torch.rand(batch_size, int(l_grain/2)+1, dtype=torch.cfloat, device=device)*2-1).unsqueeze(1)
    # print(new_noise_fft.shape)
    for i in range(1, n_grains):  
        # starting_point = i*hop_size
        # ending_point = starting_point+l_grain
        # tmp_noise = noise[:, starting_point:ending_point].unsqueeze(1)
        # new_noise = torch.cat((new_noise, tmp_noise), dim = 1)
        tmp_fft = (torch.rand(batch_size, int(l_grain/2)+1, dtype=torch.cfloat, device=device)*2-1).unsqueeze(1)
        new_noise_fft = torch.cat((new_noise_fft, tmp_fft), dim=1)

    new_noise = torch.fft.irfft(new_noise_fft)
    print(new_noise.shape)

    # print(new_noise.shape)
    # print(new_noise_fft.shape)
    # # plt.plot(torch.fft.rfft(noise)[0,513:1026])
    # # plt.plot(torch.rand(513))
    # plt.plot(new_noise_fft[0,7])
    # # plt.plot(noise[0, 513:1026])
    # plt.savefig("test_freq.png")
    # print("Done printing")

    # print(torch.fft.rfft(new_noise).real.min())
    # print(torch.fft.rfft(new_noise).real.max())
    
    return new_noise

def generate_noise_grains_wNorm(batch_size, n_grains, l_grain, dtype, device, hop_ratio=0.25):

    tar_l = int(((n_grains+3)/4)*l_grain)

    noise = torch.rand(batch_size, tar_l, dtype=dtype, device=device)*2-1
    # TEST using a slightly differently scaled noise
    # noise = torch.rand(batch_size, tar_l, dtype=dtype, device=device)*0.6-0.3

    # Normalise the fft of the signal (only normalising the real part)
    

    # print("White Noise Max (freq): ", noise_fft.real.max())
    # print("White Noise Min (freq): ", noise_fft.real.min())
    # noise = torch.fft.irfft(noise_fft)
    # print("White Noise Max (freq): ", torch.fft.rfft(noise).real.max())
    # print("White Noise Min (freq): ", torch.fft.rfft(noise).real.min())
    # print(torch.fft.rfft(new_noise).shape)

    hop_size = int(hop_ratio*l_grain)

    new_noise = noise[:, 0:l_grain].unsqueeze(1)
    for i in range(1, n_grains):  
        starting_point = i*hop_size
        ending_point = starting_point+l_grain
        tmp_noise = noise[:, starting_point:ending_point].unsqueeze(1)
        new_noise = torch.cat((new_noise, tmp_noise), dim = 1)

    # Normalise the grains
    noise_fft = torch.fft.rfft(new_noise)
    noise_fft = noise_fft - noise_fft.real.min(dim=2, keepdim=True).values 
    noise_fft = (noise_fft / noise_fft.real.max(dim=2, keepdim=True).values) * 2 - 1

    new_noise = torch.fft.irfft(noise_fft)

    return new_noise

def generate_noise_grains_stft(batch_size, tar_l, dtype, device, hop_size):

    noise = torch.rand(batch_size, tar_l, dtype=dtype, device=device)*2-1

    noise_stft = librosa.stft(noise.cpu().numpy(), hop_length=hop_size)
    noise_stft = torch.from_numpy(noise_stft)

    return noise_stft

def print_spectral_shape(waveform, learnt_spec_shape, hop_size, l_grain):

    print("-----Saving Spectral Shape-----")

    slice_kernel = torch.eye(l_grain).unsqueeze(1)
    mb_grains = F.conv1d(waveform.unsqueeze(0).unsqueeze(0).cpu(), slice_kernel,stride=hop_size,groups=1,bias=None)
    mb_grains = mb_grains.permute(0,2,1).squeeze()

    grain_fft = fft.rfft(mb_grains.cpu().numpy())

    grain_db = 20*np.log10(np.abs(grain_fft))

    plt.plot(grain_db[0])

    # Note transposing for librosa
    cepstral_coeff = fft.dct(grain_db)

    cepstral_coeff[:, 128:] = 0

    inv_cepstral_coeff = fft.idct(cepstral_coeff)
    plt.plot(inv_cepstral_coeff[0])

    plt.plot(learnt_spec_shape[0])

    plt.savefig("spectral_shape.png")

def filter_spectral_shape(waveform, hop_size, l_grain, n_grains, tar_l):

    print("-----Noise filtering Spectral Shape-----")
    # Set BS equal to 1
    bs = 1

    slice_kernel = torch.eye(l_grain).unsqueeze(1)
    mb_grains = F.conv1d(waveform.unsqueeze(0).unsqueeze(0).cpu(), slice_kernel,stride=hop_size,groups=1,bias=None)
    mb_grains = mb_grains.permute(0,2,1)

    ola_window = signal.hann(l_grain,sym=False)
    ola_windows = torch.from_numpy(ola_window).unsqueeze(0).repeat(n_grains,1).type(torch.float32)
    ola_windows[0,:l_grain//2] = ola_window[l_grain//2] # start of 1st grain is not windowed for preserving attacks
    ola_windows[-1,l_grain//2:] = ola_window[l_grain//2] # end of last grain is not wondowed to preserving decays
    ola_windows = torch.nn.Parameter(ola_windows,requires_grad=False)
    mb_grains = mb_grains*(ola_windows.unsqueeze(0).repeat(bs,1,1))


    grain_fft = fft.rfft(mb_grains.cpu().numpy())
    grain_fft_torch = torch.fft.rfft(mb_grains)
    # grain_db = 20*np.log10(np.abs(grain_fft))
    grain_db = 20*np.log10(np.abs(grain_fft))
    grain_db_torch = 20*torch.log10(torch.abs(grain_fft_torch))

    # Note transposing for librosa
    cepstral_coeff = fft.dct(grain_db)
    cepstral_coeff_torch = dct.dct(grain_db_torch)

    cepstral_coeff[:, :, 128:] = 0
    cepstral_coeff_torch[:, :, 128:] = 0

    # Use torch and dct for now
    cepstral_coeff = cepstral_coeff_torch

    # inv_cepstral_coeff = 10**(fft.idct(cepstral_coeff) / 20)
    inv_cepstral_coeff = 10**(dct.idct(cepstral_coeff) / 20)

    # filter_ir = dsp.amp_to_impulse_response_w_phase(torch.from_numpy(inv_cepstral_coeff), l_grain)
    filter_ir = dsp.amp_to_impulse_response(inv_cepstral_coeff, l_grain)

    noise = generate_noise_grains(bs, n_grains, l_grain, filter_ir.dtype, filter_ir.device, hop_ratio=0.25)
    noise = noise.reshape(bs*n_grains, l_grain)

    audio = dsp.fft_convolve_no_pad(noise, filter_ir)

    # Check if number of grains wanted is entered, else use the original
    audio = audio.reshape(-1,n_grains,l_grain)
    # audio = inv_grain_fft.reshape(-1,n_grains,l_grain)

    # Check if an overlapp add window has been passed, if not use that used in encoding.

    ola_window = signal.hann(l_grain,sym=False)
    ola_windows = torch.from_numpy(ola_window).unsqueeze(0).repeat(n_grains,1).type(torch.float32)
    ola_windows[0,:l_grain//2] = ola_window[l_grain//2] # start of 1st grain is not windowed for preserving attacks
    ola_windows[-1,l_grain//2:] = ola_window[l_grain//2] # end of last grain is not wondowed to preserving decays
    ola_windows = ola_windows
    audio = audio*(ola_windows.unsqueeze(0).repeat(bs,1,1))

    audio = audio/torch.max(audio)


    # Overlap add folder, folds and reshapes audio into target dimensions, so [bs, tar_l]
    # This is essentially folding and adding the overlapping grains back into original sample size, based on the given hop size.
    # Note that shape is changed here, so that input tensor is of chape [bs, channel X (kernel_size), L],
    # since kernel size is l_grain, this is needed in the second dimension.
    ola_folder = torch.nn.Fold((tar_l,1),(l_grain,1),stride=(hop_size,1))
    audio_sum = ola_folder(audio.permute(0,2,1)).squeeze()

    # Normalise the energy values across the audio samples
    if NORMALIZE_OLA:
        # Normalises based on number of overlapping grains used in folding per point in time.
        unfolder = torch.nn.Unfold((l_grain,1),stride=(hop_size,1))
        input_ones = torch.ones(1,1,tar_l,1)
        ola_divisor = ola_folder(unfolder(input_ones)).squeeze()
        ola_divisor = ola_divisor
        audio_sum = audio_sum/ola_divisor.unsqueeze(0).repeat(bs,1)

    #for testing
    for i in range(audio_sum.shape[0]):
        torchaudio.save(f'{RECONSTRUCTION_SAVE_DIR}/cc_filtering_{i}.wav', audio_sum[i].unsqueeze(0).cpu(), SAMPLE_RATE)

def convert_audio(wav: torch.Tensor, sr: int, target_sr: int, target_channels: int):
    assert wav.dim() >= 2, "Audio tensor must have at least 2 dimensions"
    assert wav.shape[-2] in [1, 2], "Audio must be mono or stereo."
    *shape, channels, length = wav.shape
    if target_channels == 1:
        wav = wav.mean(-2, keepdim=True)
    elif target_channels == 2:
        wav = wav.expand(*shape, target_channels, length)
    elif channels == 1:
        wav = wav.expand(target_channels, -1)
    else:
        raise RuntimeError(f"Impossible to convert from {channels} to {target_channels}")
    wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    return wav

# def pghi(spectrogram, tgrad, fgrad, a, M, L, tol=10):
#     """"Implementation of "A noniterativemethod for reconstruction of phase from STFT magnitude". by Prusa, Z., Balazs, P., and Sondergaard, P. Published in IEEE/ACM Transactions on Audio, Speech and LanguageProcessing, 25(5):1154–1164 on 2017. 
#     a = hop size
#     M = fft window size
#     L = signal length
#     tol = tolerance under the max value of the spectrogram
#     """
#     abstol = -20
#     done_mask = np.zeros_like(spectrogram)
#     phase = np.zeros_like(spectrogram)
#     max_val = np.amax(spectrogram[done_mask == 0])
#     max_pos = np.where(spectrogram==max_val)
       
#     if max_val <= abstol:  #Avoid integrating the phase for the spectogram of a silent signal
#         print('Empty spectrogram')
#         return phase

#     M2 = spectrogram.shape[0]
#     N = spectrogram.shape[1]
#     b =  L / M  
    
#     sampToRadConst =  2.0 * np.pi / L # Rescale the derivs to rad with step 1 in both directions
#     tgradw = a * tgrad * sampToRadConst
#     fgradw = - b * ( fgrad + np.arange(spectrogram.shape[1]) * a ) * sampToRadConst # also convert relative to freqinv convention
                 
#     magnitude_heap = []
#     done_mask[spectrogram < max_val-tol] = 3 # Do not integrate over silence

#     while np.any([done_mask==0]):
#         max_val = np.amax(spectrogram[done_mask == 0]) # Find new maximum value to start integration
#         max_pos = np.where(spectrogram==max_val)
#         heapq.heappush(magnitude_heap, (-max_val, max_pos))
#         done_mask[max_pos] = 1

#         while len(magnitude_heap)>0: # Integrate around maximum value until reaching silence
#             max_val, max_pos = heapq.heappop(magnitude_heap)
            
#             col = max_pos[0]
#             row = max_pos[1]
            
#             #Spread to 4 direct neighbors
#             N_pos = col+1, row
#             S_pos = col-1, row
#             E_pos = col, row+1
#             W_pos = col, row-1

#             if max_pos[0] < M2-1 and not done_mask[N_pos]:
#                 phase[N_pos] = phase[max_pos] + (fgradw[max_pos] + fgradw[N_pos])/2
#                 done_mask[N_pos] = 2
#                 heapq.heappush(magnitude_heap, (-spectrogram[N_pos], N_pos))

#             if max_pos[0] > 0 and not done_mask[S_pos]:
#                 phase[S_pos] = phase[max_pos] - (fgradw[max_pos] + fgradw[S_pos])/2
#                 done_mask[S_pos] = 2
#                 heapq.heappush(magnitude_heap, (-spectrogram[S_pos], S_pos))

#             if max_pos[1] < N-1 and not done_mask[E_pos]:
#                 phase[E_pos] = phase[max_pos] + (tgradw[max_pos] + tgradw[E_pos])/2
#                 done_mask[E_pos] = 2
#                 heapq.heappush(magnitude_heap, (-spectrogram[E_pos], E_pos))

#             if max_pos[1] > 0 and not done_mask[W_pos]:
#                 phase[W_pos] = phase[max_pos] - (tgradw[max_pos] + tgradw[W_pos])/2
#                 done_mask[W_pos] = 2
#                 heapq.heappush(magnitude_heap, (-spectrogram[W_pos], W_pos))
#     return phase

        
