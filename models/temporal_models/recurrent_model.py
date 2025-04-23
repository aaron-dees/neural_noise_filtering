import sys
sys.path.append('../')

from scripts.configs.noise_filtering_config import BATCH_SIZE, DEVICE, LATENT_SIZE


import torch
import torch.nn as nn
from torch.nn import functional as F

    
class RnnBlock(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, n_layers, *args, **kwargs):
		"""
		config net type 0 
		This is the baseline model and the simplest.
		A stack of RNN layers with dense layers at input and output

		Outputs - audio: logits to be fed into a softmax or cross entropy loss
				  parameters: Predicts a mean and variance for a normal distribution that can be sampled using sample_normal.

		input_size: if input layer - combined size of generated+conditional vectors, 
					for one-hot audio=mu-law channels + cond vector size
		hidden_size: no. of hidden nodes for each GRU layer
		output_size: size of output, normally equal to no. of generated parameters
					 for audio normally=256 for 8-bit mu-law
		n_layers: no of stacked GRU layers
		"""
		super(RnnBlock, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.n_layers = n_layers

		self.i2d = nn.Linear(input_size, hidden_size)
		self.gru = nn.GRU(hidden_size, hidden_size, n_layers, batch_first=True)
		self.h2o = nn.Linear(hidden_size, output_size)
	
	def forward(self, input, hidden, batch_size=1, **kwargs):
		
		h1 = F.relu(self.i2d(input)) 
		h_out, hidden = self.gru(h1.view(batch_size,1,-1),hidden)  #hidden shape = (n_layers*n_directions,batch_size,hidden_size)
		output = self.h2o(h_out.view(batch_size,-1))
		
		return output, hidden

	# initialize hiddens for each minibatch
	def init_hidden(self,batch_size=1):
		if self.plstm:
			return torch.zeros(batch_size, self.hidden_size, dtype=torch.float)
		else:
			return torch.zeros(self.n_layers, batch_size, self.hidden_size, dtype=torch.float)

# ------------
# LATENT AUTO-ENCODER v1
# ------------

class RNN_v1(nn.Module):
    def __init__(self,
				 input_size,
				 hidden_size,
				 output_size,
				 rnn_layers):
        super().__init__()
		
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.rnn_layers = rnn_layers

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.rnn_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

class RNN_v2(nn.Module):
    def __init__(self,
				 input_size,
				 hidden_size,
				 output_size,
				 rnn_layers):
        super().__init__()
		
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.rnn_layers = rnn_layers

        self.linear1 = nn.Linear(self.input_size, self.hidden_size)
        self.lstm = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=self.rnn_layers, batch_first=True)
        self.linear2 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        x = self.linear1(x)
        x, _ = self.lstm(x)
        x = self.linear2(x)
        return x
    
# class RNN_v1(nn.Module):

#     def __init__(self,
# 				 input_size,
# 				 hidden_size,
# 				 output_size,
# 				 n_layers,
#                 ):
#         super(RNN_v1, self).__init__()

#         self.net = RnnBlock(input_size, hidden_size, output_size, n_layers)

#     def forward(self, latents):
	
#         outputs, hidden = self.net(input,hidden,input.shape[0])

#         return outputs, hidden
	
#     def _get_init_hidden(self, batch_size):

#         hidden = self.net.init_hidden(batch_size)
		
#         return hidden
		
