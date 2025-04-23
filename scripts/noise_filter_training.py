import sys
sys.path.append('../')

# Internal Imports
from models.noise_filtering.noise_filtering_vae import SpectralVAE_v1
from models.dataloaders.dataloaders import AudioDataset, AudioDataset_old
from models.dataloaders.customAudioDataset import CustomAudioDataset, collate_fn
from models.loss_functions.loss_functions import compute_kld, spectral_distances, envelope_distance, calc_reconstruction_loss
from scripts.configs.noise_filtering_config import *
from utils.utilities import export_latents, init_beta
from utils.dsp_components import safe_log10

# External Imports
import torch
import torchaudio
from torch.autograd import Variable
from scipy import signal
import time
import wandb
import numpy as np
from datetime import datetime
import torch_dct as dct
import librosa
from frechet_audio_distance import FrechetAudioDistance
import matplotlib.pyplot as plt
import librosa.display

print("--- Device: ", DEVICE)

# torch.manual_seed(0)

# start a new wandb run to track this script
if WANDB:
    wandb.login(key='31e9e9ed4e2efc0f50b1e6ffc9c1e6efae114bd2')
    wandb.init(
        # set the wandb project where this run will be logged
        project="filterbank_model",
        # name= f"run_{datetime.now()}",
        name= f"removed_reshape_addedtoKLLossCalc",
    
        # track hyperparameters and run metadata
        config={
        "learning_rate": LEARNING_RATE,
        "architecture": "v1",
        "dataset": "Full_Seawaves_UrbanSound8k",
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "latent size": LATENT_SIZE,
        "target_beta": TARGET_BETA,
        "beta_steps": BETA_STEPS,
        "beta_warmup_start": BETA_WARMUP_START_PERC,
        "hidden_dim": H_DIM,
        "grain_length": GRAIN_LENGTH,
        "hop_size_ratio": HOP_SIZE_RATIO,
        "num_ccs": NUM_CC,
        "num_mels": NUM_MELS,
        "sample_rate": SAMPLE_RATE,
        }
    )

# Evaluation metric
# TODO Do i need to resample audio before saving to 16kHz?
frechet = FrechetAudioDistance(
    model_name="vggish",
    # Do I need to resample these?
    sample_rate=16000,
    use_pca=False, 
    use_activation=False,
    verbose=False
)

if __name__ == "__main__":


    # audio_dataset = AudioDataset(dataset_path=AUDIO_PATHS, audio_size_samples=AUDIO_SAMPLE_SIZE, min_batch_size=BATCH_SIZE, sampling_rate=SAMPLE_RATE, device=DEVICE)
    # n_samples = len(audio_dataset)
    # audio_dataset = AudioDataset_old(dataset_path=AUDIO_DIR, audio_size_samples=AUDIO_SAMPLE_SIZE, min_batch_size=BATCH_SIZE, sampling_rate=SAMPLE_RATE, device=DEVICE)
    # train_dataset,test_dataset = torch.utils.data.random_split(audio_dataset, [n_train, n_samples-n_train])
    # train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    # val_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    custom_audio_dataset = CustomAudioDataset(csv_path=AUDIO_PATHS, sample_rate=SAMPLE_RATE, channels=1, tensor_cut=AUDIO_SAMPLE_SIZE) 
    n_samples = len(custom_audio_dataset)
    n_train = int(n_samples*TRAIN_SPLIT)
    trainset, testset = torch.utils.data.random_split(custom_audio_dataset, [n_train, n_samples-n_train])
    train_dataloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=BATCH_SIZE,
        shuffle=True, collate_fn=collate_fn,
        pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(
        testset,
        batch_size=BATCH_SIZE,
        shuffle=False, collate_fn=collate_fn,
        pin_memory=True)
        
    print("-----Dataset Loaded-----")
    # TODO Make a test dataloader

    hop_size = int(GRAIN_LENGTH * HOP_SIZE_RATIO)
    l_grain = GRAIN_LENGTH


    model = SpectralVAE_v1(l_grain=GRAIN_LENGTH, h_dim=H_DIM, z_dim=LATENT_SIZE, synth_window=hop_size, mfcc_hop_size=hop_size, n_band=2048, noise_synth=NOISE_SYNTH)
    model.to(DEVICE)

    if TRAIN:

        print("-------- Training Mode --------")

        ###########
        # Training
        ########### 

        optimizer = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)
        # lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.25, total_iters=4000)
        # decayRate = 0.99
        # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)

        start_epoch = 0
        accum_iter = 0
        # Calculate the max number of steps based on the number of epochs, number_of_epochs * batches_in_single_epoch
        max_steps = EPOCHS * len(train_dataloader)
        beta, beta_step_val, beta_step_size, warmup_start = init_beta(max_steps, TARGET_BETA, BETA_STEPS, BETA_WARMUP_START_PERC)

        if LOAD_CHECKPOINT:
            checkpoint = torch.load(CHECKPOINT_LOAD_PATH)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            train_loss = checkpoint['loss']
            # accum_iter = checkpoint['accum_iter']
            beta = checkpoint['beta']
            beta_step_val = checkpoint['beta_step_val']
            beta_step_size = checkpoint['beta_step_size']
            warmup_start = checkpoint['warmup_start']


            print("----- Checkpoint File Loaded -----")
            print(f'Epoch: {start_epoch}')
            print(f'Loss: {train_loss}')

        # Set spectral distances
        spec_dist = spectral_distances(sr=SAMPLE_RATE, device=DEVICE)

        for epoch in range(start_epoch, EPOCHS):

            start = time.time()

            # Turn gradient trackin on for training loop
            model.train()

            ###############
            # Training loop 
            ###############
            running_train_loss = 0.0
            running_kl_loss = 0.0
            running_spec_loss = 0.0

            # for data in train_dataloader:
            for waveform in train_dataloader:

                # set the beta for weighting the KL Divergence
                if (accum_iter+1)%beta_step_size==0:
                    if accum_iter<warmup_start:
                        beta = 0
                    elif beta<TARGET_BETA:
                        beta += beta_step_val
                        beta = np.min([beta,TARGET_BETA])
                    else:
                        beta = TARGET_BETA

                waveform = Variable(waveform).to(DEVICE)                      
                optimizer.zero_grad()                 

                # ---------- Run Model ----------
                x_hat, z, mu, log_variance = model(waveform, noise_synth=NOISE_SYNTH)

                # Reshape mu and log_variance for KLD calculation
                mu = mu.reshape(mu.shape[0]*mu.shape[1],mu.shape[2])
                log_variance = mu.reshape(log_variance.shape[0]*log_variance.shape[1], log_variance.shape[2])

                # ---------- Run Model END ----------

                # Reconstruction Loss
                spec_loss = spec_dist(x_hat, waveform)

                # Regularisation Loss
                if beta > 0:
                    kld_loss = compute_kld(mu, log_variance) * beta
                else:
                    kld_loss = 0.0

                loss = kld_loss + spec_loss

                # Compute gradients and update weights
                loss.backward()                    
                optimizer.step() 

                # Accumulate loss for reporting
                running_train_loss += loss
                running_kl_loss += kld_loss
                running_spec_loss += spec_loss

                accum_iter+=1

            # Decay the learning rate
            # lr_scheduler.step()
            # new_lr = optimizer.param_groups[0]["lr"]
                
            # get avg training statistics 
            train_loss = running_train_loss/len(train_dataloader) 
            kl_loss = running_kl_loss/len(train_dataloader)
            train_spec_loss = running_spec_loss/len(train_dataloader)
            
            #################
            # Validation loop
            #################
            running_val_loss = 0.0
            running_kl_val_loss = 0.0
            running_spec_val_loss = 0.0
            running_multi_spec_loss = 0.0

            model.eval()


            with torch.no_grad():
                for waveform in val_dataloader:

                    waveform = waveform.to(DEVICE)
                    
                    # ---------- Run Model ----------
                    x_hat, z, mu, log_variance = model(waveform, noise_synth=NOISE_SYNTH)
                    mu = mu.reshape(mu.shape[0]*mu.shape[1],mu.shape[2])
                    log_variance = mu.reshape(log_variance.shape[0]*log_variance.shape[1], log_variance.shape[2])
                    # ---------- Run Model END ----------

                    spec_loss = spec_dist(x_hat, waveform)

                    if beta > 0:
                        kld_loss = compute_kld(mu, log_variance) * beta
                    else:
                        kld_loss = 0.0

                    loss = kld_loss + spec_loss 

                    running_val_loss += loss
                    running_kl_val_loss += kld_loss
                    running_spec_val_loss += spec_loss
                
                # Get avg stats
                val_loss = running_val_loss/len(val_dataloader)
                kl_val_loss = running_kl_val_loss/len(val_dataloader)
                spec_val_loss = running_spec_val_loss/len(val_dataloader)

            end = time.time()


            # wandb logging
            if WANDB:
                wandb.log({"kl_loss": kl_loss, "spec_loss": train_spec_loss, "loss": train_loss, "kl_val_loss": kl_val_loss, "spec_val_loss": spec_val_loss, "val_loss": val_loss, "beta": beta})

            print('Epoch: {}'.format(epoch+1),
            '\tStep: {}'.format(accum_iter+1),
            '\t KL Loss: {:.5f}'.format(kl_loss),
            '\tTraining Loss: {:.4f}'.format(train_loss),
            '\tValidation Loss: {:.4f}'.format(val_loss),
            '\tTime: {:.2f}s'.format(end-start))

            if SAVE_CHECKPOINT:
                if (epoch+1) % CHECKPOINT_REGULAIRTY == 0:
                    torch.save({
                        'epoch': epoch+1,
                        'accum_iter': accum_iter,
                        'beta': beta,
                        'beta_step_val': beta_step_val,
                        'beta_step_size': beta_step_size,
                        'warmup_start': warmup_start,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': train_loss,
                       }, f"{SAVE_DIR}/waveform_vae_{DEVICE}_{EPOCHS}epochs_{BATCH_SIZE}batch_{beta}beta_{epoch+1}epoch_{datetime.now()}.pt")
                    # Save as latest also
                    torch.save({
                        'epoch': epoch+1,
                        'accum_iter': accum_iter,
                        'beta': beta,
                        'beta_step_val': beta_step_val,
                        'beta_step_size': beta_step_size,
                        'warmup_start': warmup_start,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': train_loss,
                        }, f"{SAVE_DIR}/waveform_vae_latest_latentTest.pt")
                    

            if SAVE_RECONSTRUCTIONS:
                if (epoch+1) % RECON_REGULAIRTY == 0:

                    # Get data using test dataset
                    with torch.no_grad():

                        dataiter = iter(val_dataloader)
                        waveform = next(dataiter)
                        waveform = waveform.to(DEVICE)

                        # ---------- Run Model ----------
                        x_hat, z, mu, log_variance = model(waveform, noise_synth=NOISE_SYNTH)
                        # ---------- Run Model END ----------

                        spec_loss = spec_dist(x_hat, waveform)

                        for i, recon_signal in enumerate(x_hat):
                            print("Saving ", i)
                            torchaudio.save(f'{RECONSTRUCTION_SAVE_DIR}/fake_audio/CC_recon.wav', recon_signal.unsqueeze(0).cpu(), SAMPLE_RATE)
                            torchaudio.save(f"{RECONSTRUCTION_SAVE_DIR}/real_audio/CC_{i}.wav", waveform[i].unsqueeze(0).cpu(), SAMPLE_RATE)

                        fad_score = frechet.score(f'{RECONSTRUCTION_SAVE_DIR}/real_audio', f'{RECONSTRUCTION_SAVE_DIR}/fake_audio', dtype="float32")

                        print('Test Spec Loss: {}'.format(spec_loss),
                            '\tTest FAD Score: {}'.format(fad_score))

                        if WANDB:
                            wandb.log({"test_spec_loss": spec_loss, "test_fad_score": fad_score})

    else:

        print("-------- Inference Mode --------")

        ###########
        # Inference
        ########### 
        audio, sr_orig = librosa.load(AUDIO_DIR+"/loud_waves.wav", sr=SAMPLE_RATE, mono=True)
        full_audio = torch.from_numpy(audio).unsqueeze(0)

        seed = 0
        torch.manual_seed(seed)


        # with torch.no_grad():
        if LOAD_CHECKPOINT:
            checkpoint = torch.load(CHECKPOINT_LOAD_PATH, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'], strict = False)

        # Put model in eval mode
        model.to(DEVICE)
        model.eval()

        with torch.no_grad():

            # Lets get batch of test images
            dataiter = iter(val_dataloader)
            waveforms = next(dataiter)
            waveforms = waveforms.to(DEVICE)

            # full audio test
            waveforms = full_audio

            # ---------- Run Model ----------
            x_hat, z, mu, log_variance = model(waveforms, noise_synth=NOISE_SYNTH)
            recon_audio = x_hat
            # ---------- Run Model END ----------

            # Save Spectrogram
            audio = librosa.resample(waveforms.cpu().squeeze().detach().numpy(), orig_sr=SAMPLE_RATE, target_sr=22050)
            S, phase = librosa.magphase(librosa.stft(y=audio))
            fig, ax = plt.subplots()
            fig.set_figheight(4)
            fig.set_figwidth(6)
            librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max)[:,:],
                                 y_axis='log', x_axis='time', ax=ax)
            ax.set(title='log Power spectrogram')
            plt.savefig("spec.png")

            spec_dist = spectral_distances(sr=SAMPLE_RATE, device=DEVICE)
            spec_loss = spec_dist(recon_audio, waveforms[:,:recon_audio.shape[1]])
            print("Spectral Loss: ", spec_loss)

            if SAVE_RECONSTRUCTIONS:
                for i, signal in enumerate(recon_audio):
                    print("Saving ", i)
                    print("Loss: ", spec_loss)
                    torchaudio.save(f'{RECONSTRUCTION_SAVE_DIR}/fake_audio/CC_recon_{i}.wav', signal.unsqueeze(0).cpu(), SAMPLE_RATE)
                    torchaudio.save(f"{RECONSTRUCTION_SAVE_DIR}/real_audio/CC_{i}.wav", waveforms[i].unsqueeze(0).cpu(), SAMPLE_RATE)

                    fad_score = frechet.score(f'{RECONSTRUCTION_SAVE_DIR}/real_audio', f'{RECONSTRUCTION_SAVE_DIR}/fake_audio', dtype="float32")
                    print("FAD Score: ", fad_score)