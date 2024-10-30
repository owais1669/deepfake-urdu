from typing import List, Union, Callable
import torch
import torchaudio

SAMPLING_RATE = 16_000
win_length = 400  # 25ms window
hop_length = 160  # 10ms hop length

device = "cuda" if torch.cuda.is_available() else "cpu"

# Define MFCC and LFCC transforms
MFCC_FN = torchaudio.transforms.MFCC(
    sample_rate=SAMPLING_RATE,
    n_mfcc=80,  # Reduced to avoid zero filterbank issue
    melkwargs={
        "n_fft": 512,
        "win_length": win_length,
        "hop_length": hop_length,
    },
).to(device)


LFCC_FN = torchaudio.transforms.LFCC(
    sample_rate=SAMPLING_RATE,
    n_lfcc=80,  # Adjust if needed
    speckwargs={
        "n_fft": 512,
        "win_length": win_length,
        "hop_length": hop_length,
    },
).to(device)

delta_fn = torchaudio.transforms.ComputeDeltas(win_length=400, mode="replicate")

# Define the frontend selector
def get_frontend(frontends: List[str]) -> Union[Callable, torchaudio.transforms.MFCC, torchaudio.transforms.LFCC]:
    if not frontends:
        raise ValueError("Frontend list is empty! Please specify 'mfcc' or 'lfcc' in the config.")
    
    if "mfcc" in frontends:
        return prepare_mfcc_double_delta
    elif "lfcc" in frontends:
        return prepare_lfcc_double_delta
    else:
        raise ValueError(f"{frontends} frontend is not supported!")

# Define double delta functions
def prepare_mfcc_double_delta(input):
    if input.ndim < 4:
        input = input.unsqueeze(1)  # (bs, 1, n_mfcc, frames)
    x = MFCC_FN(input)
    delta = delta_fn(x)
    double_delta = delta_fn(delta)
    x = torch.cat((x, delta, double_delta), 2)  # -> [bs, 1, 80 * 3, frames]
    return x[:, :, :, :3000]  # Trim if necessary

def prepare_lfcc_double_delta(input):
    if input.ndim < 4:
        input = input.unsqueeze(1)  # (bs, 1, n_lfcc, frames)
    x = LFCC_FN(input)
    delta = delta_fn(x)
    double_delta = delta_fn(delta)
    x = torch.cat((x, delta, double_delta), 2)  # -> [bs, 1, 80 * 3, frames]
    return x[:, :, :, :3000]  # Trim if necessary
