import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import os

class PlantDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        # We need to model e(t+1) = f(e(t), disturbance(t), Kp_corr(t), Ki_corr(t))
        # Assuming rows are sequential
        self.data = df.values.astype(np.float32)
        self.inputs = self.data[:-1] # e(t), dist(t), Kp_corr(t), Ki_corr(t)
        self.targets = self.data[1:, 0:1] # e(t+1)
        
        # Shuffle the pairs to prevent overfitting/underfitting
        indices = np.random.permutation(len(self.inputs))
        self.inputs = self.inputs[indices]
        self.targets = self.targets[indices]
        
        self.scaler_in = StandardScaler()
        self.scaler_out = StandardScaler()
        
        self.inputs_scaled = self.scaler_in.fit_transform(self.inputs)
        self.targets_scaled = self.scaler_out.fit_transform(self.targets)
        
    def __len__(self):
        return len(self.inputs_scaled)
    
    def __getitem__(self, idx):
        return torch.tensor(self.inputs_scaled[idx]), torch.tensor(self.targets_scaled[idx])

if __name__ == "__main__":
    csv_path = r'e:\Caterpiller\ML_RL.csv'
    dataset = PlantDataset(csv_path)
    print(f"Dataset size: {len(dataset)}")
    os.makedirs(r'e:\Caterpiller\project\data', exist_ok=True)
    # Save scalers for later use in environment
    import pickle
    with open(r'e:\Caterpiller\project\data\scalers.pkl', 'wb') as f:
        pickle.dump({'in': dataset.scaler_in, 'out': dataset.scaler_out}, f)
