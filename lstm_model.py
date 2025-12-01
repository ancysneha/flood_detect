import torch
import torch.nn as nn

class LSTMForecast(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, output_size=1):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, seq):
        lstm_out, _ = self.lstm(seq)
        final = lstm_out[:, -1, :]
        return torch.sigmoid(self.fc(final))
