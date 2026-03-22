import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(r'e:\Caterpiller\project')
from data.preprocess import PlantDataset

class SurrogateModel(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64):
        super(SurrogateModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # Output: next error
        )
        
    def forward(self, x):
        return self.net(x)

def train_surrogate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csv_path = r'e:\Caterpiller\ML_RL.csv'
    dataset = PlantDataset(csv_path)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    model = SurrogateModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    num_epochs = 20
    print(f"Training surrogate model for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        epoch_loss = 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(dataloader):.6f}")
            
    # Save model
    model_path = r'e:\Caterpiller\project\models\surrogate_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Surrogate model saved to {model_path}")

if __name__ == "__main__":
    train_surrogate()
