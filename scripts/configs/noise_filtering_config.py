import torch

DEVICE = torch.device("cpu") 

print("DEVICE: ", DEVICE)
# if torch.cuda.is_available():
#     DEVICE = torch.device("cuda:0")
# else:
#     DEVICE = torch.device("cpu") 

# Hyper Parameters
BATCH_SIZE = 16
TRAIN_SPLIT=0.8
EPOCHS = 40000
LEARNING_RATE = 0.001
LATENT_SIZE = 16
H_DIM = 128

# Regularisation Params
# Number of warmup iterations before increasing beta
BETA_WARMUP_START_PERC = 0.1
TARGET_BETA = 0.0001
# number of warmup steps over half max_steps
BETA_STEPS = 250

# Noise Synth
NOISE_SYNTH = "filterbank"
# NOISE_SYNTH = "ddsp"

# Audio Processing Parameters
TEST_AUDIO_DIR = "/Users/adees/Code/neural_granular_synthesis/datasets/OceanWaves/samples/1secs/full"
AUDIO_DIR = "/Users/adees/Code/neural_granular_synthesis/datasets/OceanWaves/samples/1secs/full"
AUDIO_PATHS = "./data_csv.csv"

NUM_MELS = 64
SAMPLE_RATE = 24000
NUM_CC = 64
# MONO = True
NORMALIZE_OLA = True

# 0.88
# Grain Params
# AUDIO_SAMPLE_SIZE = 24064
AUDIO_SAMPLE_SIZE = 23936 
HOP_SIZE_RATIO = 0.25
GRAIN_LENGTH = 512

# Mode and directories
WANDB = False
TRAIN = True
EXPORT_LATENTS = False
SAVE_CHECKPOINT = False
CHECKPOINT_REGULAIRTY = 10
RECON_REGULAIRTY = 10
LOAD_CHECKPOINT = False
VIEW_LATENT = False
SAVE_RECONSTRUCTIONS = True
COMPARE_ENERGY = False
SEQ_TEST = False
SAVE_DIR = f"/Users/adees/Code/neural_granular_synthesis/models/saved_models/checkpoints"
RECONSTRUCTION_SAVE_DIR = "/Users/adees/Code/neural_granular_synthesis/scripts/recon_audio"

# Best model noisebandnet: v2 encoder, v3 decoder, 128 hidden units.
CHECKPOINT_LOAD_PATH = f"/Users/adees/Code/saved_models/paperModels/all/filterbank_model_v2enc_v3dec_128.pt"