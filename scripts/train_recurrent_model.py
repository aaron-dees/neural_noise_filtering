import sys
sys.path.append('../')

from models.noise_filtering.noise_filtering_vae import SpectralVAE_v1
from models.temporal_models.recurrent_model import RNN_v1, RNN_v2
from models.dataloaders.customAudioDataset import  CustomAudioDataset, collate_fn
from models.dataloaders.latent_dataloaders import make_latent_dataloaders
from scripts.configs.temporal_config import *
# from scripts.configs.hyper_parameters_waveform import LATENT_SIZE, AUDIO_DIR, SAMPLE_RATE, NORMALIZE_OLA, POSTPROC_KER_SIZE, POSTPROC_CHANNELS, HOP_SIZE_RATIO, GRAIN_LENGTH, TARGET_LENGTH, HIGH_PASS_FREQ
from scripts.configs.noise_filtering_config import LATENT_SIZE, AUDIO_PATHS, SAMPLE_RATE, HOP_SIZE_RATIO, GRAIN_LENGTH, TRAIN_SPLIT, AUDIO_SAMPLE_SIZE, H_DIM, NOISE_SYNTH
from utils.utilities import plot_latents, export_latents, latent_to_audio 


import torch
import torch.nn as nn
import torchaudio
from torch.autograd import Variable
import pickle
import time
import wandb
import numpy as np
from datetime import datetime

def create_dataset(dataset, lookback):
    """Transform a time series into a prediction dataset
    
    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    feature = dataset[:, 0:0+lookback,:].unsqueeze(1)
    target = dataset [:, 0+1:0+lookback+1,:].unsqueeze(1)
    for i in range(1, dataset.shape[1]-lookback):
        feature = torch.cat((feature, dataset[:,i:i+lookback,:].unsqueeze(1)),1)
        target = torch.cat((target, dataset[:,i+1:i+lookback+1,:].unsqueeze(1)),1)

    return feature, target

from frechet_audio_distance import FrechetAudioDistance

# to use `vggish`
frechet = FrechetAudioDistance(
    model_name="vggish",
    sample_rate=16000,
    use_pca=False, 
    use_activation=False,
    verbose=False
)

# start a new wandb run to track this script
if WANDB:
    wandb.login(key='31e9e9ed4e2efc0f50b1e6ffc9c1e6efae114bd2')
    wandb.init(
        # set the wandb project where this run will be logged
        project=WANDB_PROJECT,
        name= f"{WANDB_NAME}_{datetime.now()}",
    
        # track hyperparameters and run metadata
        config={
        "learning_rate": LEARNING_RATE,
        "architecture": "LSTM+Linear",
        "dataset": "UrbanSound8K",
        "epochs": EPOCHS,
        "grain_length": GRAIN_LENGTH,
        "latent_size": LATENT_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "rnn_layers": NO_RNN_LAYERS,
        "lookback": LOOKBACK
        }
    )

if __name__ == "__main__":


    print("-------- Load model and exporting Latents --------")
    hop_size = int(GRAIN_LENGTH * HOP_SIZE_RATIO)
    l_grain = GRAIN_LENGTH

    # train_dataloader,val_dataloader,dataset,tar_l,n_grains,l_grain,hop_size,classes = make_audio_dataloaders(data_dir=AUDIO_DIR,classes=["sea_waves"],sr=SAMPLE_RATE,silent_reject=[0.2,0.2],amplitude_norm=False,batch_size=BATCH_SIZE,hop_ratio=HOP_SIZE_RATIO, tar_l=TARGET_LENGTH,l_grain=GRAIN_LENGTH,high_pass_freq=HIGH_PASS_FREQ,num_workers=0)
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

    # Test dataloader
    # test_set = torch.utils.data.Subset(dataset, range(0,TEST_SIZE))
    # test_dataloader = torch.utils.data.DataLoader(test_set, batch_size = TEST_SIZE, shuffle=False, num_workers=0)
    # test_dataloader, _, _, _, _, _, _, _ = make_audio_dataloaders(data_dir=TEST_AUDIO_DIR,classes=["sea_waves"],sr=SAMPLE_RATE,silent_reject=[0.2,0.2],amplitude_norm=False,batch_size=TEST_SIZE,hop_ratio=HOP_SIZE_RATIO, tar_l=TARGET_LENGTH,l_grain=GRAIN_LENGTH,high_pass_freq=HIGH_PASS_FREQ,num_workers=0)

    # w_model = SpectralVAE_v1(n_grains=n_grains, l_grain=l_grain, h_dim=H_DIM, z_dim=LATENT_SIZE)
    w_model = SpectralVAE_v1(l_grain=GRAIN_LENGTH, h_dim=H_DIM, z_dim=LATENT_SIZE, synth_window=hop_size, mfcc_hop_size=hop_size, n_band=2048, noise_synth=NOISE_SYNTH)
    # w_model = SpectralVAE_v2(n_grains=n_grains, l_grain=l_grain, h_dim=[2048, 1024, 512], z_dim=LATENT_SIZE)
    # w_model = SpectralVAE_v3(n_grains=n_grains, l_grain=l_grain, h_dim=[2048, 1024, 512], z_dim=LATENT_SIZE, channels = 32, kernel_size = 3, stride = 2)
    
    if LOAD_WAVEFORM_CHECKPOINT:
        checkpoint = torch.load(WAVEFORM_CHECKPOINT_LOAD_PATH, map_location=DEVICE)
        w_model.load_state_dict(checkpoint['model_state_dict'])

    w_model.to(DEVICE)
    w_model.eval()

    print("--- Exporting latents")

    # train_latents,train_labels,val_latents,val_labels = export_latents(w_model,train_dataloader,val_dataloader, DEVICE)
    # # train_latents,train_labels,val_latents,val_labels = export_latents(w_model,test_dataloader,test_dataloader, DEVICE)
    # test_latents,test_labels,_,_ = export_latents(w_model,test_dataloader,test_dataloader, DEVICE)
    # train_latents,train_labels,val_latents,val_labels = export_latents(w_model,train_dataloader,val_dataloader, l_grain, n_grains, hop_size, BATCH_SIZE,DEVICE)
    train_latents,val_latents = export_latents(w_model,train_dataloader,val_dataloader, BATCH_SIZE,DEVICE)
    # test_latents,test_labels, _, _ = export_latents(w_model,test_dataloader,test_dataloader, l_grain, n_grains, hop_size, TEST_SIZE, DEVICE)

    print("--- Creating dataset ---")
    # print(train_latents.shape)
    # print(val_latents.shape)
    # print(test_latents.shape)

    # train_latents = train_latents[:128, :, :]


    # lookback = 150
    X_train, y_train = create_dataset(train_latents, lookback=LOOKBACK)
    val_X_train, val_y_train = create_dataset(val_latents, lookback=LOOKBACK)
    # X_test, y_test = create_dataset(test_latents, lookback=LOOKBACK)

    X_train = X_train.reshape(-1, X_train.shape[2], X_train.shape[3])
    y_train = y_train.reshape(-1, y_train.shape[2], y_train.shape[3])
    val_X_train = val_X_train.reshape(-1, val_X_train.shape[2], val_X_train.shape[3])
    val_y_train = val_y_train.reshape(-1, val_y_train.shape[2], val_y_train.shape[3])
    # X_test = X_test.reshape(-1, X_test.shape[2], X_test.shape[3])
    # y_test = y_test.reshape(-1, y_test.shape[2], y_test.shape[3])

    # l_model = RNN_v1(LATENT_SIZE, HIDDEN_SIZE, LATENT_SIZE, NO_RNN_LAYERS)
    l_model = RNN_v2(LATENT_SIZE, HIDDEN_SIZE, LATENT_SIZE, NO_RNN_LAYERS)

    optimizer = torch.optim.Adam(l_model.parameters(), lr=LEARNING_RATE)
    # lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.25, total_iters=3*EPOCHS/4)

    loss_fn = nn.MSELoss()
    # Note the batch size here 
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, y_train), shuffle=True, batch_size=BATCH_SIZE)
    val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(val_X_train, val_y_train), shuffle=True, batch_size=BATCH_SIZE)
    
    if TRAIN:

        if LOAD_LATENT_CHECKPOINT:
            checkpoint = torch.load(LATENT_CHECKPOINT_LOAD_PATH, map_location=DEVICE)
            l_model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            train_loss = checkpoint['loss']
            accum_iter = checkpoint['accum_iter']
            beta = checkpoint['beta']
            beta_step_val = checkpoint['beta_step_val']
            beta_step_size = checkpoint['beta_step_size']
            warmup_start = checkpoint['warmup_start']
            print("----- Checkpoint File Loaded -----")
            print(f'Epoch: {start_epoch}')
            print(f'Loss: {train_loss}')

        for epoch in range(EPOCHS):
            l_model.train()
            running_rmse = 0.0;
            for X_batch, y_batch in loader:
                y_pred = l_model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                running_rmse += (loss.detach().numpy())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # # Decay the learning rate
            # lr_scheduler.step()
            # new_lr = optimizer.param_groups[0]["lr"]
            # Validation
            if epoch % EVAL_EVERY != 0:
                continue
            l_model.eval()
            running_val_rmse = 0.0
            with torch.no_grad():
                for val_X_batch, val_y_batch in val_loader:
                    y_pred = l_model(val_X_train)
                    running_val_rmse += (loss_fn(y_pred, val_y_train))

            # Run the test case - NOTE, this should be only over newly generated samples
            with torch.no_grad():
                y_pred = l_model(X_test)
                test_rmse = (loss_fn(y_pred, y_test))

            # Take the first sequence that is fed to the model
            recon_latent = X_test[0,:,:]
            # Add the next 10 samples on
            tmp = X_test[0,:,:].unsqueeze(0)
            for i in range(0, y_pred.shape[0]):
                tmp = l_model(tmp)
                recon_latent = torch.cat((recon_latent, tmp[0,-1,:].unsqueeze(0)), dim=0)
                # recon_latent = torch.cat((recon_latent, y_pred[i,-1,:].unsqueeze(0)), dim=0)
            
            sampled_seq_loss = loss_fn(recon_latent[LOOKBACK:, :] ,test_latents[0, LOOKBACK:, :])

            train_rmse = running_rmse/len(loader)
            val_rmse = running_val_rmse/len(val_loader)

            print("Epoch %d: train RMSE %.4f, validation RMSE %.4f" % (epoch, train_rmse, val_rmse))

            # wandb logging
            if WANDB:
                wandb.log({"train_RMSE": train_rmse, "val_RMSE": val_rmse, "test_RMSE": sampled_seq_loss})

            # Early stopping Criteria
            if sampled_seq_loss < 0.0001 and EARLY_STOPPING:
                print("Stopped Early as test RMSE %.4f < 0.0001" % (sampled_seq_loss))
                torch.save({
                        'epoch': epoch,
                        'model_state_dict': l_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': train_rmse,
                        }, f"{SAVE_MODEL_DIR}/latent_vae_latest_earlyStop.pt")
                break


            if SAVE_CHECKPOINT:
                if (epoch) % CHECKPOINT_REGULAIRTY == 0:
                    torch.save({
                        'epoch': epoch,
                        # 'warmup_start': warmup_start,
                        'model_state_dict': l_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': train_rmse,
                        }, f"{SAVE_MODEL_DIR}/latent_vae_{DEVICE}_{EPOCHS}epochs_{BATCH_SIZE}batch_{epoch}epoch_{datetime.now()}.pt")
                    # Save as latest also
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': l_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': train_rmse,
                        }, f"{SAVE_MODEL_DIR}/latent_vae_latest.pt")

        # Save after final epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': l_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_rmse,
            }, f"{SAVE_MODEL_DIR}/latent_vae_latest.pt")
            

    if EXPORT_AUDIO_RECON:

        print("-------- Exporting Audio Reconstructions --------")

        if LOAD_LATENT_CHECKPOINT:
            checkpoint = torch.load(LATENT_CHECKPOINT_LOAD_PATH, map_location=DEVICE)
            l_model.load_state_dict(checkpoint['model_state_dict'])

        # Maybe start with 10 samles, and try reconstruct the full latent
        print(n_grains)
        print(X_test.shape)

        l_model.eval()
        with torch.no_grad():
            y_pred = l_model(X_test)

        # Take the first sequence that is fed to the model
        recon_latent = X_test[0,:,:]
        # Add the next 10 samples on
        tmp = X_test[0,:,:].unsqueeze(0)
        print("Out hsape: ", tmp.shape)
        print(y_pred.shape)
        for i in range(0, y_pred.shape[0]):
            tmp = l_model(tmp)
            recon_latent = torch.cat((recon_latent, tmp[0,-1,:].unsqueeze(0)), dim=0)
            # recon_latent = torch.cat((recon_latent, y_pred[i,-1,:].unsqueeze(0)), dim=0)

        pred_sample_loss = loss_fn(recon_latent[LOOKBACK:, :], test_latents[0, LOOKBACK:, :])


        print("Total Loss: ", pred_sample_loss)
        z = test_latents.reshape(-1,w_model.z_dim)
        z_hat = recon_latent.reshape(-1,w_model.z_dim)

        latent_to_audio(z, z_hat, w_model, EXPORT_AUDIO_DIR, SAMPLE_RATE, DEVICE, hop_size, tar_l, HOP_SIZE_RATIO, trainset=True)






    #     dataiter = iter(test_dataloader)
    #     waveforms, labels = next(dataiter)
    #     waveforms = waveforms.to(DEVICE)
    #     for i, signal in enumerate(waveforms):
    #         torchaudio.save(f"{EXPORT_AUDIO_DIR}/real_audio/CC_{i}.wav", waveforms[i].unsqueeze(0).cpu(), SAMPLE_RATE)

    #     for batch in test_latentloader:
    #         export_embedding_to_audio_reconstructions(l_model, w_model, batch, EXPORT_AUDIO_DIR, SAMPLE_RATE, DEVICE, hop_size, tar_l, HOP_SIZE_RATIO, trainset=True)
    #         fad_score_real = frechet.score(f'{EXPORT_AUDIO_DIR}/real_audio', f'{EXPORT_AUDIO_DIR}/latentmodel_audio', dtype="float32")
    #         fad_score_wavemodel = frechet.score(f'{EXPORT_AUDIO_DIR}/waveformmodel_audio', f'{EXPORT_AUDIO_DIR}/latentmodel_audio', dtype="float32")
    #         fad_score_waveform_vs_real = frechet.score(f'{EXPORT_AUDIO_DIR}/waveformmodel_audio', f'{EXPORT_AUDIO_DIR}/real_audio', dtype="float32")
    #         print("FAD Score waveform model vs real: ", fad_score_waveform_vs_real)
    #         print("FAD Score latent model vs real: ", fad_score_real)
    #         print("FAD Score latent model vs wavemodel: ", fad_score_wavemodel)
    #         # break
        
    #     print("-------- Exporting Audio Reconstructions DONE --------")


    #     print("-------- Exporting Random Latent Audio Reconstructions --------")

    #     export_random_samples(l_model,w_model, EXPORT_RANDOM_LATENT_AUDIO_DIR, LATENT_SIZE, TEMPORAL_LATENT_SIZE,SAMPLE_RATE, ["SeaWaves"], DEVICE, tar_l, hop_size, HOP_SIZE_RATIO, n_samples=10)

    #     print("-------- Exporting Random Latent Audio Reconstructions Done --------")

    # #     model.to(DEVICE)
    # #     model.eval()

    #     # train_latents,train_labels,test_latents,test_labels = export_latents(model,test_dataloader,test_dataloader)
    # #     # train_latents,train_labels,test_latents,test_labels = export_latents(model,train_dataloader,val_dataloader)
        
    # #     print("-------- Done Exporting Latents --------")

