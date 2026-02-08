import torch
import torch.nn as nn


class Attention(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (batch, seq_len, hidden_dim)
        weights = torch.softmax(self.attn(x), dim=1)
        context = torch.sum(weights * x, dim=1)
        return context


class MultitaskLSTMCNNAttention(nn.Module):

    def __init__(self, input_dim=3, cnn_channels=32, lstm_hidden=64):
        super().__init__()

        # Temporal CNN
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # LSTM
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            batch_first=True
        )

        # Attention
        self.attention = Attention(lstm_hidden)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )


    # x: (batch, seq_len, features)
    def forward(self, x):    

        # CNN expects (batch, channels, seq_len)
        x = x.transpose(1, 2)
        x = self.cnn(x)

        # Back to (batch, seq_len, features)
        x = x.transpose(1, 2)

        # LSTM
        x, _ = self.lstm(x)

        # Attention
        shared = self.attention(x)

        # Heads
        class_logits = self.classifier(shared)
        intensity = self.regressor(shared).squeeze(-1)

        return class_logits, intensity
