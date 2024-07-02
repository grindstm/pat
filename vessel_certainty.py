import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader
import math


class PATDataset(Dataset):
    def __init__(self, pressure_data, sensor_positions, N):
        """
        :param pressure_data: Numpy array of shape [time, sensors], where time is 1275 and sensors is 64.
        :param sensor_positions: Tuple of (x, y, z) positions.
        :param N: Tuple of (Nx, Ny, Nz) representing the dimensions of the volume.
        """
        self.pressure_data = pressure_data
        self.sensor_positions = sensor_positions
        self.N = N

        # Reshape the pressure data to fit the spatial dimensions
        self.spatial_data = self.reshape_to_spatial()

    def __len__(self):
        return self.pressure_data.shape[1]  # Number of sensors

    def __getitem__(self, idx):
        # Return the reshaped pressure data corresponding to all time points for a single sensor
        return self.spatial_data[:, :, idx]

    def reshape_to_spatial(self):
        # Reshape pressure data to match sensor spatial positions
        spatial_data = np.zeros((self.N[2], self.N[1], self.N[0]))

        for i, (x, y, z) in enumerate(zip(*self.sensor_positions)):
            # Place each sensor's time series data into the corresponding spatial position
            xi = int(np.round((x - sensor_margin) / (self.N[0] - 2 * sensor_margin) * (self.N[0] - 1)))
            yi = int(np.round((y - sensor_margin) / (self.N[1] - 2 * sensor_margin) * (self.N[1] - 1)))
            zi = int(z)  # Assuming z is already an integer appropriate for indexing
            spatial_data[zi, yi, xi] = self.pressure_data[:, i]

        return spatial_data
    
class PATNet(nn.Module):
    def __init__(self):
        super(PATNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose3d(16, 1, kernel_size=2, stride=2),
            nn.ReLU()
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def train_model(model, dataloader, epochs, device, max_intensity):
    # Loss function
    criterion = nn.MSELoss()
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data in dataloader:
            inputs = data['input'].to(device)  # Ensure your dataloader passes this key
            targets = data['target'].to(device)  # Ensure your dataloader passes this key

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        # Evaluate using PSNR
        model.eval()
        total_psnr = 0
        with torch.no_grad():
            for data in dataloader:
                inputs = data['input'].to(device)
                targets = data['target'].to(device)
                outputs = model(inputs)
                mse = criterion(outputs, targets).item()
                psnr = 20 * math.log10(max_intensity / math.sqrt(mse))
                total_psnr += psnr

        avg_psnr = total_psnr / len(dataloader)
        print(f'Epoch {epoch+1}: Avg. Loss: {avg_loss:.4f}, Avg. PSNR: {avg_psnr:.2f} dB')

dataset = PATDataset(pressure_data, sensor_positions, N)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)  # Batch size and shuffle as needed

device = torch.device("cuda")
epochs = 10
max_intensity = 1.0  # Adjust according to the maximum value in your dataset
model = PATNet()
dataloader = DataLoader(PATDataset(), batch_size=1, shuffle=True)  # Adjust accordingly
train_model(model, dataloader, epochs, device, max_intensity)
