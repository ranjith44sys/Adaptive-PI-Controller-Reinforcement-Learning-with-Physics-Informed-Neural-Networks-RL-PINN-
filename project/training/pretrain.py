import torch
import numpy as np
import pandas as pd
import pickle
import sys
import os

# Add project root to path
sys.path.append(r'e:\Caterpiller\project')
from models.actor_critic import Actor

class PrioritizedReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(2e5), alpha=0.4, beta=0.6, max_priority=5.0):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.alpha = alpha
        self.beta = beta
        self.max_priority = max_priority
        
        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))
        self.dist = np.zeros((max_size, 1))
        self.priorities = np.zeros((max_size,))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def add(self, state, action, next_state, reward, done, dist):
        self.state[self.ptr] = np.array(state, dtype=np.float32).flatten()
        self.action[self.ptr] = np.array(action, dtype=np.float32).flatten()
        self.next_state[self.ptr] = np.array(next_state, dtype=np.float32).flatten()
        self.reward[self.ptr] = float(reward)
        self.not_done[self.ptr] = float(1.0 - done)
        self.dist[self.ptr] = float(dist)
        
        # Initial priority: max priority in buffer or high constant
        max_prio = self.priorities.max() if self.size > 0 else 1.0
        self.priorities[self.ptr] = max_prio
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
    def sample(self, batch_size):
        if self.size == self.max_size:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.ptr]
            
        # Section 6 — PER priority: abs(td_error)**0.6 + abs(disturbance)*2.0 
        # Note: td_error priority updated after training. Here we use stored priorities.
        # But we must incorporate disturbance weighting in the sampling too.
        
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        ind = np.random.choice(self.size, batch_size, p=probs)
        weights = (self.size * probs[ind]) ** (-self.beta)
        weights /= weights.max()
        
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            torch.FloatTensor(weights).to(self.device),
            ind
        )

    def update_priorities(self, indices, td_errors):
        # Section 4 — Calibrated PER (TD-error only, capped)
        td_errors = np.array(td_errors).flatten()
        for idx, td_error in zip(indices, td_errors):
            priority = min(abs(td_error) + 1e-5, self.max_priority)
            self.priorities[int(idx)] = priority

def pretrain_actor(actor, csv_path, epochs=10):
    df = pd.read_csv(csv_path)
    # Mapping: (error, disturbance) -> (Kp_corr, Ki_corr)
    # We need to construct the state used in the RL env: [error, de/dt, dist_scaled]
    # Assuming rows are sequential for de/dt
    states = []
    actions = []
    
    # Simple state construction for pretraining
    error = df['error'].values
    dist = df['disturbance'].values
    kp = df['Kp_corr'].values
    ki = df['Ki_corr'].values
    
    setpoint = 900.0
    max_dist = 500.0
    for i in range(1, len(df)):
        de = error[i] - error[i-1]
        # State: [e/setp, de/setp, dist/max_dist, prev_Kp_centred, prev_Ki_centred]
        state = [
            error[i] / (setpoint + 1e-6),
            de / (setpoint + 1e-6),
            dist[i] / max_dist,
            (kp[i-1] - 1.25) / 0.75,
            (ki[i-1] - 1.25) / 0.75
        ]
        action = [kp[i], ki[i]]
        states.append(state)
        actions.append(action)
        
    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.float32)
    
    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    states_t = torch.FloatTensor(states).to(device)
    actions_t = torch.FloatTensor(actions).to(device)
    
    batch_size = 256
    num_batches = len(states) // batch_size
    
    print(f"Pretraining actor with Behavior Cloning for {epochs} epochs...")
    for epoch in range(epochs):
        indices = np.random.permutation(len(states))
        total_loss = 0
        for i in range(0, len(states), batch_size):
            batch_indices = indices[i:i+batch_size]
            b_state = states_t[batch_indices]
            b_action = actions_t[batch_indices]
            
            optimizer.zero_grad()
            pred_action = actor(b_state)
            loss = criterion(pred_action, b_action)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 2 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], BC Loss: {total_loss/num_batches:.6f}")
            
    return actor

if __name__ == "__main__":
    # Test pretraining skeleton
    actor = Actor(3, 2, 50.0)
    # pretrain_actor(actor, r'e:\Caterpiller\ML_RL.csv') 
