import sys
import torch
import torch.nn as nn
from src.frontends import get_frontend

# Constants
NUM_COEFFICIENTS = 384

# Define BLSTM Layer
class BLSTMLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        if output_dim % 2 != 0:
            print(f"Output_dim of BLSTMLayer is {output_dim}, which should be even.")
            sys.exit(1)
        self.l_blstm = nn.LSTM(input_dim, output_dim // 2, bidirectional=True)

    def forward(self, x):
        blstm_data, _ = self.l_blstm(x.permute(1, 0, 2))
        return blstm_data.permute(1, 0, 2)

# Define MaxFeatureMap2D Layer
class MaxFeatureMap2D(nn.Module):
    def __init__(self, max_dim=1):
        super().__init__()
        self.max_dim = max_dim

    def forward(self, inputs):
        shape = list(inputs.size())
        if shape[self.max_dim] % 2 != 0:
            sys.exit("MaxFeatureMap dimension must be even")
        shape[self.max_dim] = shape[self.max_dim] // 2
        shape.insert(self.max_dim, 2)
        m, _ = inputs.view(*shape).max(self.max_dim)
        return m

# Define LCNN Model
class LCNN(nn.Module):
    def __init__(self, input_channels=1, num_coefficients=NUM_COEFFICIENTS):
        super().__init__()
        self.num_coefficients = num_coefficients

        self.m_transform = nn.Sequential(
            nn.Conv2d(input_channels, 64, (5, 5), 1, padding=(2, 2)),
            MaxFeatureMap2D(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, (1, 1)),
            MaxFeatureMap2D(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 96, (3, 3), 1, padding=(1, 1)),
            MaxFeatureMap2D(),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(48),
            nn.Conv2d(48, 96, (1, 1)),
            MaxFeatureMap2D(),
            nn.BatchNorm2d(48),
            nn.Conv2d(48, 128, (3, 3), 1, padding=(1, 1)),
            MaxFeatureMap2D(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 128, (1, 1)),
            MaxFeatureMap2D(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, (3, 3), 1, padding=(1, 1)),
            MaxFeatureMap2D(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, (1, 1)),
            MaxFeatureMap2D(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, (3, 3), 1, padding=(1, 1)),
            MaxFeatureMap2D(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.7)
        )

        self.m_before_pooling = nn.Sequential(
            BLSTMLayer((self.num_coefficients // 16) * 32, (self.num_coefficients // 16) * 32),
            BLSTMLayer((self.num_coefficients // 16) * 32, (self.num_coefficients // 16) * 32)
        )

        self.m_output_act = nn.Linear((self.num_coefficients // 16) * 32, 1)

    def _compute_embedding(self, x):
        batch_size = x.shape[0]
        output_emb = torch.zeros([batch_size, 1], device=x.device, dtype=x.dtype)
        x = x.permute(0, 1, 3, 2)
        hidden_features = self.m_transform(x)
        hidden_features = hidden_features.permute(0, 2, 1, 3).contiguous()
        frame_num = hidden_features.shape[1]
        hidden_features = hidden_features.view(batch_size, frame_num, -1)
        hidden_features_lstm = self.m_before_pooling(hidden_features)
        tmp_emb = self.m_output_act((hidden_features_lstm + hidden_features).mean(1))
        output_emb[:] = tmp_emb
        return output_emb

    def forward(self, x):
        return self._compute_embedding(x)

# Define FrontendLCNN Model
class FrontendLCNN(nn.Module):
    def __init__(self, device="cpu", frontend_name="mfcc"):
        super(FrontendLCNN, self).__init__()
        self.device = device
        self.frontend = get_frontend([frontend_name])
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # Set input channels to 3
        self.fc = nn.Linear(32 * 40 * 40, 512)  # Adjust dimensions as needed
        print(f"Using {frontend_name} frontend")

    def _compute_frontend(self, x):
        frontend = self.frontend(x)
        if frontend.ndim == 5:
            frontend = frontend[:, :, :, :, 0]  # Convert to 4D tensor if needed
        elif frontend.size(1) == 3:  # If it has 3 channels, average them to get a single channel
            frontend = frontend.mean(dim=1, keepdim=True)  # Average across the channel dimension
        elif frontend.ndim < 4:
            frontend = frontend.unsqueeze(1)  # Ensure it is 4D
        return frontend

    def _compute_embedding(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def forward(self, x):
        x = self._compute_frontend(x)
        return self._compute_embedding(x)

# Test the model
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FrontendLCNN(device=device, frontend_name="mfcc")
    model.to(device)
    batch_size = 12
    mock_input = torch.rand((batch_size, 1, 128, 128), device=device)
    output = model(mock_input)
    print("Output shape:", output.shape)
