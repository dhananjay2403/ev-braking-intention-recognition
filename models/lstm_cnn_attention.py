'''
    This model takes a (75 × 3) time-series window and predicts one of 3 braking intentions.

    Why each block exists:

    1. CNN (Conv1D)
        - Extracts local braking patterns
        - Finds short-term signal changes
        - Reduces noise before LSTM

    2. LSTM
        - Captures temporal evolution
        - Learns braking buildup vs sudden braking

    3. Attention
        - Learns which time steps matter most
        - Focuses on braking onset moments
        - Improves interpretability & accuracy

    4. Fully Connected 
        - Maps learned features → intention class 
        - You should be able to explain this without hesitation.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    """
    Simple attention mechanism over time dimension.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_outputs):
        """
        lstm_outputs: (batch_size, time_steps, hidden_dim)
        """
        # Compute attention scores
        scores = self.attention(lstm_outputs)  # (batch, time, 1)
        weights = torch.softmax(scores, dim = 1)

        # Weighted sum of LSTM outputs
        context = torch.sum(weights * lstm_outputs, dim = 1)
        return context


# CNN + LSTM + Attention model for braking intention recognition.
class LSTMCNNAttention(nn.Module):

    def __init__(self, num_features = 3, num_classes = 3):

        super().__init__()

        # CNN block (local feature extraction) 
        self.conv1 = nn.Conv1d(
            in_channels = num_features,
            out_channels = 32,
            kernel_size = 3,
            padding = 1
        )
        self.bn1 = nn.BatchNorm1d(32)

        # LSTM block (temporal modeling) 
        self.lstm = nn.LSTM(
            input_size = 32,
            hidden_size = 64,
            num_layers = 1,
            batch_first = True
        )

        # Attention block 
        self.attention = AttentionLayer(hidden_dim = 64)

        # Fully connected layers 
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, num_classes)


    def forward(self, x):
        """
        x shape: (batch_size, time_steps, num_features)
        """

        # CNN expects (batch, channels, time)
        x = x.permute(0, 2, 1)

        # CNN forward
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        # Back to (batch, time, features)
        x = x.permute(0, 2, 1)

        # LSTM forward
        lstm_out, _ = self.lstm(x)

        # Attention
        context = self.attention(lstm_out)

        # Fully connected layers
        x = F.relu(self.fc1(context))
        output = self.fc2(x)

        return output


if __name__ == "__main__":

    model = LSTMCNNAttention()
    dummy_input = torch.randn(2, 75, 3)  # batch_size = 2, time_steps = 75, num_features = 3
    output = model(dummy_input)
    print("Output shape:", output.shape)
    print("Output:", output)