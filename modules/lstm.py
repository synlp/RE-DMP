import torch
from torch import nn

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first = True):
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            bidirectional=True
        )

    def rand_init_hidden(self, batch_size, device):
        return (torch.zeros(2 * self.rnn_layers, batch_size, self.hidden_dim).to(device),
                torch.zeros(2 * self.rnn_layers, batch_size, self.hidden_dim).to(device))

    def forward(self, input):
        batch_size = input.shape[0]
        hidden = self.rand_init_hidden(batch_size, input.device)
        output, hidden = self.lstm(input, hidden)
        return output.contiguous().view(-1, self.hidden_size * 2)

