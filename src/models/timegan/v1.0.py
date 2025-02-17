import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# ----- Hyperparameters -----
SEQ_LEN      = 24         # sequence length (e.g., 24 hours)
FEATURE_DIM  = 5          # number of features (Open, High, Low, Close, Volume)
HIDDEN_DIM   = 24         # latent dimension size
NUM_LAYERS   = 2
BATCH_SIZE   = 64
EPOCHS       = 100        # adjust as needed
NOISE_DIM    = HIDDEN_DIM # dimension for noise input

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- Data Loading & Preprocessing -----
data_filepath = '/content/BTCUSDT_1h_Jan_2025.csv'
df = pd.read_csv(data_filepath)
# Select key columns. Adjust if your CSV column names differ.
columns = ['Open', 'High', 'Low', 'Close', 'Volume']
df = df[columns]
print(df.head())


save_path =  "/content/BTCUSDT_processed.csv"
df.to_csv(save_path, index=False)

# Normalize the data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df)

# Create sequences (sliding window)
def create_sequences(data, seq_len):
    sequences = []
    for i in range(len(data) - seq_len + 1):
        seq = data[i:i+seq_len]
        sequences.append(seq)
    return np.array(sequences)

real_data_np = create_sequences(df, SEQ_LEN)
real_data_np = real_data_np.astype(np.float32)
print("Real data shape:", real_data_np.shape)


# Create DataLoader
dataset = TensorDataset(torch.tensor(real_data_np, dtype=torch.float32))
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ----- Model Components -----
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
    def forward(self, x):
        h, _ = self.lstm(x)
        return h

class Recovery(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers):
        super(Recovery, self).__init__()
        self.lstm = nn.LSTM(hidden_dim, output_dim, num_layers, batch_first=True)
        
    def forward(self, h):
        x_tilde, _ = self.lstm(h)
        return x_tilde

class Generator(nn.Module):
    def __init__(self, noise_dim, hidden_dim, num_layers):
        super(Generator, self).__init__()
        self.lstm = nn.LSTM(noise_dim, hidden_dim, num_layers, batch_first=True)
        
    def forward(self, z):
        e_hat, _ = self.lstm(z)
        return e_hat

class Supervisor(nn.Module):
    def __init__(self, hidden_dim, num_layers):
        super(Supervisor, self).__init__()
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        
    def forward(self, h):
        h_sup, _ = self.lstm(h)
        return h_sup

class Discriminator(nn.Module):
    def __init__(self, hidden_dim, num_layers):
        super(Discriminator, self).__init__()
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_dim, 1)
        
    def forward(self, h):
        out, _ = self.lstm(h)
        last_hidden = out[:, -1, :]
        score = self.fc(last_hidden)
        return torch.sigmoid(score)

# ----- Instantiate Models -----
encoder    = Encoder(FEATURE_DIM, HIDDEN_DIM, NUM_LAYERS).to(device)
recovery   = Recovery(HIDDEN_DIM, FEATURE_DIM, NUM_LAYERS).to(device)
generator  = Generator(NOISE_DIM, HIDDEN_DIM, NUM_LAYERS).to(device)
supervisor = Supervisor(HIDDEN_DIM, NUM_LAYERS).to(device)
discriminator = Discriminator(HIDDEN_DIM, NUM_LAYERS).to(device)

# ----- Losses & Optimizers -----
mse_loss = nn.MSELoss()
bce_loss = nn.BCELoss()

enc_rec_optimizer = optim.Adam(list(encoder.parameters()) + list(recovery.parameters()), lr=0.001)
sup_optimizer     = optim.Adam(supervisor.parameters(), lr=0.001)
gen_optimizer     = optim.Adam(list(generator.parameters()) + list(supervisor.parameters()), lr=0.001)
disc_optimizer    = optim.Adam(discriminator.parameters(), lr=0.001)

# ----- Training Loop -----
for epoch in range(EPOCHS):
    for batch in data_loader:
        X = batch[0].to(device)  # [BATCH_SIZE, SEQ_LEN, FEATURE_DIM]
        batch_size = X.size(0)
        
        # --- Phase 1: Autoencoder Training ---
        H = encoder(X)
        X_tilde = recovery(H)
        loss_autoencoder = mse_loss(X_tilde, X)
        enc_rec_optimizer.zero_grad()
        loss_autoencoder.backward()
        enc_rec_optimizer.step()
        
        # --- Phase 2: Supervisor Training ---
        H = encoder(X).detach()
        H_sup = supervisor(H[:, :-1, :])
        loss_supervised = mse_loss(H_sup, H[:, 1:, :])
        sup_optimizer.zero_grad()
        loss_supervised.backward()
        sup_optimizer.step()
        
        # --- Phase 3: Adversarial Training ---
        Z = torch.randn(batch_size, SEQ_LEN, NOISE_DIM).to(device)
        E_hat = generator(Z)
        H_hat = supervisor(E_hat)
        
        H_real = encoder(X).detach()
        y_real = torch.ones(batch_size, 1).to(device)
        y_fake = torch.zeros(batch_size, 1).to(device)
        
        D_real = discriminator(H_real)
        D_fake = discriminator(H_hat.detach())
        loss_disc_real = bce_loss(D_real, y_real)
        loss_disc_fake = bce_loss(D_fake, y_fake)
        loss_discriminator = loss_disc_real + loss_disc_fake
        
        disc_optimizer.zero_grad()
        loss_discriminator.backward()
        disc_optimizer.step()
        
        D_fake_for_gen = discriminator(H_hat)
        loss_generator_adv = bce_loss(D_fake_for_gen, y_real)
        loss_generator_sup = mse_loss(supervisor(E_hat)[:, :-1, :], E_hat[:, 1:, :])
        loss_generator = loss_generator_adv + loss_generator_sup
        
        gen_optimizer.zero_grad()
        loss_generator.backward()
        gen_optimizer.step()
        
    if (epoch+1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}] | AE Loss: {loss_autoencoder.item():.4f} | Sup Loss: {loss_supervised.item():.4f} | "
              f"D Loss: {loss_discriminator.item():.4f} | G Loss: {loss_generator.item():.4f}")

# ----- Generate Synthetic Data After Training -----
with torch.no_grad():
    # Here, we generate synthetic data sequences.
    Z_sample = torch.randn(10, SEQ_LEN, NOISE_DIM).to(device)
    E_sample = generator(Z_sample)
    H_sample = supervisor(E_sample)
    X_generated = recovery(H_sample)
    X_generated = X_generated.cpu().numpy()
    
print("Synthetic data shape:", X_generated.shape)


df_generated = pd.DataFrame(X_generated.reshape(-1, FEATURE_DIM), columns=columns)
save_dir = "/content"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "synthetic_data.csv")
df_generated.to_csv(save_path, index=False)
print(f"Synthetic data saved to {save_path}")
