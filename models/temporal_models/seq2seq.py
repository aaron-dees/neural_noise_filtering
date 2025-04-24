import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        outputs, (hidden, cell) = self.lstm(x)
        return hidden, cell
    
class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, num_layers=1):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(output_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden, cell):
        # x: [batch, 1, output_dim] – one time step
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        prediction = self.fc(output)  # [batch, 1, output_dim]
        return prediction, hidden, cell
    
class Seq2Seq(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_rnn_layers, device):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=n_rnn_layers)
        self.decoder = Decoder(output_dim=output_dim, hidden_dim=hidden_dim, num_layers=n_rnn_layers)
        self.device = device

    def forward(self, src, target_len, teacher_forcing_ratio=0.5, tgt=None):
        batch_size, _, output_dim = src.size()
        hidden, cell = self.encoder(src)

        # First input to the decoder is usually a zero vector or a start token
        # decoder_input = torch.zeros(batch_size, 1, output_dim).to(self.device)
        decoder_input = src[:,-1,:].unsqueeze(1)

        outputs = []
        for t in range(target_len):
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs.append(output)

            # Decide whether to use teacher forcing
            if tgt is not None and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = tgt[:, t:t+1, :]  # Use the true value
            else:
                decoder_input = output  # Use the predicted value

        return torch.cat(outputs, dim=1)  # [batch, target_len, output_dim]