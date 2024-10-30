"""
This code is modified version of LCNN baseline
from ASVSpoof2021 challenge - https://github.com/asvspoof-challenge/2021/blob/main/LA/Baseline-LFCC-LCNN/project/baseline_LA/model.py
"""
import sys
import torch
import torch.nn as torch_nn
from src import frontends

NUM_COEFFICIENTS = 384

# BLSTM Layer definition
class BLSTMLayer(torch_nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        if output_dim % 2 != 0:
            print("Output_dim of BLSTMLayer is {:d}".format(output_dim))
            print("BLSTMLayer expects a layer size of even number")
            sys.exit(1)
        self.l_blstm = torch_nn.LSTM(
            input_dim,
            output_dim // 2,
            bidirectional=True
        )

    def forward(self, x):
        blstm_data, _ = self.l_blstm(x.permute(1, 0, 2))
        return blstm_data.permute(1, 0, 2)

# MaxFeatureMap Layer
class MaxFeatureMap2D(torch_nn.Module):
    def __init__(self, max_dim=1):
        super().__init__()
        self.max_dim = max_dim

    def forward(self, inputs):
        shape = list(inputs.size())
        if self.max_dim >= len(shape):
            print("MaxFeatureMap: maximize on %d dim" % (self.max_dim))
            print("But input has %d dimensions" % (len(shape)))
            sys.exit(1)
        if shape[self.max_dim] // 2 * 2 != shape[self.max_dim]:
            print("MaxFeatureMap: maximize on %d dim" % (self.max_dim))
            print("But this dimension has an odd number of data")
            sys.exit(1)
        shape[self.max_dim] = shape[self.max_dim] // 2
        shape.insert(self.max_dim, 2)
        m, _ = inputs.view(*shape).max(self.max_dim)
        return m

class LCNN(torch_nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        input_channels = kwargs.get("input_channels", 1)
        num_coefficients = kwargs.get("num_coefficients", NUM_COEFFICIENTS)
        self.num_coefficients = num_coefficients
        self.v_emd_dim = 1

        self.m_transform = torch_nn.Sequential(
            torch_nn.Conv2d(input_channels, 64, (5, 5), 1, padding=(2, 2)),
            MaxFeatureMap2D(),
            torch.nn.MaxPool2d((2, 2), (2, 2)),
            torch_nn.Conv2d(32, 64, (1, 1), 1, padding=(0, 0)),
            MaxFeatureMap2D(),
            torch.nn.BatchNorm2d(32, affine=False),
            torch_nn.Conv2d(32, 96, (3, 3), 1, padding=(1, 1)),
            MaxFeatureMap2D(),
            torch.nn.MaxPool2d((2, 2), (2, 2)),
            torch_nn.BatchNorm2d(48, affine=False),
            torch_nn.Conv2d(48, 96, (1, 1), 1, padding=(0, 0)),
            MaxFeatureMap2D(),
            torch.nn.BatchNorm2d(48, affine=False),
            torch_nn.Conv2d(48, 128, (3, 3), 1, padding=(1, 1)),
            MaxFeatureMap2D(),
            torch.nn.MaxPool2d((2, 2), (2, 2)),
            torch_nn.Conv2d(64, 128, (1, 1), 1, padding=(0, 0)),
            MaxFeatureMap2D(),
            torch_nn.BatchNorm2d(64, affine=False),
            torch_nn.Conv2d(64, 64, (3, 3), 1, padding=(1, 1)),
            MaxFeatureMap2D(),
            torch.nn.BatchNorm2d(32, affine=False),
            torch_nn.Conv2d(32, 64, (1, 1), 1, padding=(0, 0)),
            MaxFeatureMap2D(),
            torch_nn.BatchNorm2d(32, affine=False),
            torch_nn.Conv2d(32, 64, (3, 3), 1, padding=(1, 1)),
            MaxFeatureMap2D(),
            torch_nn.MaxPool2d((2, 2), (2, 2)),
            torch_nn.Dropout(0.7)
        )

        self.m_before_pooling = torch_nn.Sequential(
            BLSTMLayer((self.num_coefficients // 16) * 32, (self.num_coefficients // 16) * 32),
            BLSTMLayer((self.num_coefficients // 16) * 32, (self.num_coefficients // 16) * 32)
        )

        self.m_output_act = torch_nn.Linear((self.num_coefficients // 16) * 32, self.v_emd_dim)

    def _compute_embedding(self, x):
        batch_size = x.shape[0]
        output_emb = torch.zeros(
            [batch_size, self.v_emd_dim],
            device=x.device,
            dtype=x.dtype
        )

        # Ensure x is 4D for Conv2d compatibility
        if x.dim() == 5:
            x = x.mean(dim=-1)  # Averages the extra dimension

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
        feature_vec = self._compute_embedding(x)
        return feature_vec

class FrontendLCNN(LCNN):
    def __init__(self, device: str = "cuda", **kwargs):
        super().__init__(**kwargs)
        self.device = device
        frontend_name = kwargs.get("frontend_name", [])
        self.frontend = frontends.get_frontend(frontend_name)
        print(f"Using {frontend_name} frontend")

    def _compute_frontend(self, x):
        frontend = self.frontend(x)
        if frontend.ndim < 4:
            return frontend.unsqueeze(1)  # Ensures shape: (batch, 1, n_lfcc, frames)
        return frontend

    def forward(self, x):
        x = self._compute_frontend(x)
        feature_vec = self._compute_embedding(x)
        return feature_vec


if __name__ == "__main__":

    device = "cuda"
    print("Definition of model")
    model = FrontendLCNN(input_channels=2, num_coefficients=80, device=device, frontend_algorithm=["mel_spec"])
    model = model.to(device)
    batch_size = 12
    mock_input = torch.rand((batch_size, 64_600,), device=device)
    output = model(mock_input)
    print(output.shape)
