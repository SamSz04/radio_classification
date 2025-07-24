import torch.nn as nn

class LSTMGRU(nn.Module):
    def __init__(self, output_dim, input_dim=2):
        super(LSTMGRU, self).__init__()

        self.lstm = nn.LSTM(input_dim, 128, 2, batch_first=True)

        self.gru = nn.GRU(128, 128, 2, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x.transpose(1,2))
        gru_out, _ = self.gru(lstm_out)
        out = gru_out[:, -1, :]
        out = self.fc(out)
        return out
