import argparse
import os
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import argparse
import os
import torch
from src.models import models
# Constants for feature extraction
SAMPLING_RATE = 16_000
win_length = 400  # 25ms window
hop_length = 160  # 10ms hop length

# Device selection
device = "cuda" if torch.cuda.is_available() else "cpu"

# MFCC, LFCC, and Delta Transform Functions
MFCC_FN = torchaudio.transforms.MFCC(
    sample_rate=SAMPLING_RATE,
    n_mfcc=128,
    melkwargs={
        "n_fft": 512,
        "win_length": win_length,
        "hop_length": hop_length,
    },
).to(device)

LFCC_FN = torchaudio.transforms.LFCC(
    sample_rate=SAMPLING_RATE,
    n_lfcc=128,
    speckwargs={
        "n_fft": 512,
        "win_length": win_length,
        "hop_length": hop_length,
    },
).to(device)

delta_fn = torchaudio.transforms.ComputeDeltas(
    win_length=400,
    mode="replicate",
)

# Double Delta Feature Preparation Functions
def prepare_mfcc_double_delta(waveform):
    mfcc = MFCC_FN(waveform)
    delta = delta_fn(mfcc)
    double_delta = delta_fn(delta)
    return torch.cat([mfcc, delta, double_delta], dim=0)

def prepare_lfcc_double_delta(waveform):
    lfcc = LFCC_FN(waveform)
    delta = delta_fn(lfcc)
    double_delta = delta_fn(delta)
    return torch.cat([lfcc, delta, double_delta], dim=0)

# Dataset Processing Function
def process_audio_dataset(directory_path, feature_type='mfcc'):
    data = []
    labels = []
    
    folder_label_map = {
        "Bonafide": 1,
        "Spoofed_Tacotron": 0,
        "Spoofed_TTS": 0
    }

    for folder_type, label in folder_label_map.items():
        folder_path = os.path.join(directory_path, folder_type)
        if not os.path.exists(folder_path):
            print(f"Warning: {folder_path} does not exist. Skipping.")
            continue

        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith(('.wav', '.mp3', '.flac')):
                    file_path = os.path.join(root, file)
                    try:
                        waveform, sr = torchaudio.load(file_path)
                        # Resample to 16,000 Hz if needed
                        if sr != SAMPLING_RATE:
                            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLING_RATE)(waveform)
                        
                        # Trim/pad the audio to 6 seconds
                        max_samples = 6 * SAMPLING_RATE
                        if waveform.size(1) > max_samples:
                            waveform = waveform[:, :max_samples]  # Trim to 6 seconds
                        elif waveform.size(1) < max_samples:
                            padding = max_samples - waveform.size(1)
                            waveform = F.pad(waveform, (0, padding), "constant", 0)  # Pad with zeros

                        # Extract features based on the specified type
                        if feature_type == 'mfcc':
                            features = prepare_mfcc_double_delta(waveform.to(device))
                        elif feature_type == 'lfcc':
                            features = prepare_lfcc_double_delta(waveform.to(device))
                        else:
                            raise ValueError("Invalid feature type: choose 'mfcc' or 'lfcc'.")

                        data.append(features.cpu())
                        labels.append(label)
                    except Exception as e:
                        print(f"Error processing file {file_path}: {e}")
                        continue

    return data, labels

# Custom Dataset Class
class AudioDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Training Function
from tqdm import tqdm  # Import tqdm for progress bar

# Training Function
def train_model(model, train_loader, val_loader, epochs=10, learning_rate=0.001, model_name="", feature_type=""):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        # Wrap the training DataLoader with tqdm for a progress bar
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{epochs}", unit="batch") as pbar:
            for features, labels in train_loader:
                features, labels = features.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
                # Update the progress bar
                pbar.set_postfix(loss=total_loss / (pbar.n + 1))
                pbar.update(1)  # Increment the progress bar

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}")

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f'Validation Accuracy: {accuracy:.2f}%')

    # Save the model
    os.makedirs("trained_models", exist_ok=True)
    model_save_path = f"trained_models/{model_name}_{feature_type}.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a deepfake audio detection model.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--feature_type", type=str, default="mfcc", choices=["mfcc", "lfcc"], help="Feature extraction type")
    parser.add_argument("--model_name", type=str, required=True, help="Model name to use for training")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for training")
    
    args = parser.parse_args()

    # Process dataset as before
    data, labels = process_audio_dataset(args.dataset_path, args.feature_type)
    train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.2, random_state=42)
    train_dataset = AudioDataset(train_data, train_labels)
    val_dataset = AudioDataset(val_data, val_labels)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Load model with get_model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_config = {}  # Adjust as necessary, or pass configurations for specific models here
    model = models.get_model(args.model_name, model_config, device)

    # Train model
    train_model(model, train_loader, val_loader, args.epochs, args.learning_rate, args.model_name, args.feature_type)
