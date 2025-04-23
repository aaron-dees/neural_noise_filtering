import torch

# if torch.backends.mps.is_available():
#     DEVICE = torch.device("mps:0")
# else:
DEVICE = torch.device("cpu") 

# having issues with padding on Apple GPU
DATALOADER_DEVICE = torch.device('cpu')

# Hyper Parameters
BATCH_SIZE = 1
TEST_SIZE = 1
EPOCHS = 2000
LEARNING_RATE = 0.001
# LATENT_SIZE = 128
HIDDEN_SIZE = 512
TEMPORAL_LATENT_SIZE = 128
ENV_DIST = 0
KERNEL_SIZE = 3
NO_LINEAR_LAYERS = 2
RNN_TYPE = "LSTM"
NO_RNN_LAYERS = 1
# BETA_PARAMS
# Number of warmup iterations before increasing beta
BETA_WARMUP_START_PERC = 0.1
TARGET_BETA = 0.001
# number of warmup steps over half max_steps
# BETA_STEPS = 50
BETA_STEPS = 1

LOOKBACK = 150

# # Mode and directories
WANDB = False
TRAIN = True
# EXPORT_LATENTS = True
SAVE_CHECKPOINT = True
SAVE_MODEL_DIR = f'/Users/adees/Code/neural_granular_synthesis/models/saved_models/checkpoints'
CHECKPOINT_REGULAIRTY = 100
EVAL_EVERY = 50
LOAD_WAVEFORM_CHECKPOINT = True
LOAD_LATENT_CHECKPOINT = False
EXPORT_AUDIO_RECON = True
EXPORT_AUDIO_DIR = '/Users/adees/Code/neural_granular_synthesis/scripts/audio_tests/reconstructions/latent_reconstruections'
EXPORT_RANDOM_LATENT_AUDIO_DIR = '/Users/adees/Code/neural_granular_synthesis/scripts/audio_tests/reconstructions/latent_reconstruections/random'
# VIEW_LATENT = False
# SAVE_RECONSTRUCTIONS = False
# COMPARE_ENERGY = False
# RECONSTRUCTION_SAVE_DIR = "/Users/adees/Code/neural_granular_synthesis/datasets/fsdd/reconstructions/one_sec"
WAVEFORM_CHECKPOINT_LOAD_PATH = f"/Users/adees/Code/saved_models/paperModels/all/filterbank_model_v2enc_v3dec_128.pt"
LATENT_CHECKPOINT_LOAD_PATH = f"/Users/adees/Code/neural_granular_synthesis/models/saved_models/latent/latent_temporal_150steps_1sample.pt"
# LATENT_CHECKPOINT_LOAD_PATH = f"/Users/adees/Code/neural_granular_synthesis/models/saved_models/latent/latent_v1_fullDataset_500.pt"
# LATENT_CHECKPOINT_LOAD_PATH = f"/Users/adees/Code/neural_granular_synthesis/models/saved_models/latent/latent_v1_epch20_single.pt"
# CHECKPOINT_LOAD_PATH = f"/Users/adees/Code/neural_granular_synthesis/models/saved_models/checkpoints/waveform_vae_cpu_{EPOCHS}epochs_{64}batch_{BETA}beta_{ENV_DIST}envdist_latest.pt"
