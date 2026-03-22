import gym
from gym import spaces
import numpy as np
import torch
import torch.nn as nn
import pickle
import sys
import os

# Add project root to path
sys.path.append(r'e:\Caterpiller\project')
from models.surrogate_model import SurrogateModel

class PIControllerEnv(gym.Env):
    def __init__(self, surrogate_model_path, scalers_path):
        super(PIControllerEnv, self).__init__()
        
        # Load surrogate model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.surrogate = SurrogateModel().to(self.device)
        self.surrogate.load_state_dict(torch.load(surrogate_model_path, map_location=self.device))
        self.surrogate.eval()
        
        # Load scalers
        with open(scalers_path, 'rb') as f:
            scalers = pickle.load(f)
            self.scaler_in = scalers['in']
            self.scaler_out = scalers['out']
            
        # Action space: [Kp_corr, Ki_corr]
        self.action_space = spaces.Box(low=0.5, high=2.0, shape=(2,), dtype=np.float32)
        
        # State space: [e/set, de/set, dist/max_dist, e_integral/norm, prev_Kp_c, prev_Ki_c]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        
        self.setpoint = 900.0
        self.max_disturbance = 500.0
        self.max_steps = 100
        self.reset()
        
    def reset(self):
        self.current_error = np.random.uniform(-100, 100)
        self.prev_error = self.current_error
        self.error_integral = 0.0
        self.disturbance = np.random.uniform(0, 1.0) # This will be set by CurriculumSampler
        self.prev_action = np.array([1.0, 1.0], dtype=np.float32)
        self.steps = 0
        return self._get_obs()
    
    def _get_obs(self):
        de = self.current_error - self.prev_error
        # integral clipped to [-50, 50] * setpoint
        integral_norm = self.error_integral / (self.setpoint * 50.0 + 1e-6)
        
        state = np.array([
            self.current_error / (self.setpoint + 1e-6),
            de / (self.setpoint + 1e-6),
            self.disturbance, # Disturbance normalization [0, 1]
            integral_norm,
            (self.prev_action[0] - 1.25) / 0.75,
            (self.prev_action[1] - 1.25) / 0.75,
        ], dtype=np.float32)
        return state
    

    def step(self, action):
        action = np.array(action, dtype=np.float32)
        kp_corr, ki_corr = action
        
        # Predict next error using surrogate model
        obs_raw = np.array([[self.current_error, self.disturbance * self.max_disturbance, kp_corr, ki_corr]])
        model_input_scaled = self.scaler_in.transform(obs_raw)
        
        with torch.no_grad():
            input_tensor = torch.tensor(model_input_scaled, dtype=torch.float32).to(self.device)
            next_error_scaled = self.surrogate(input_tensor).cpu().numpy()
            
        next_error = self.scaler_out.inverse_transform(next_error_scaled)[0, 0]
        
        # Update integral
        self.error_integral += self.current_error
        self.error_integral = np.clip(self.error_integral, -50.0 * self.setpoint, 50.0 * self.setpoint)

        # Section 1 — Recalibrated Reward (1.8x mild scaling)
        e_n   = next_error / (self.setpoint + 1e-6)
        de_n  = (next_error - self.current_error) / (self.setpoint + 1e-6)
        over  = max(0.0, -e_n)
        da    = abs(action - self.prev_action).mean()
        conv  = 1.0 / (1.0 + abs(e_n) + 1e-6)

        d_ratio = abs(self.disturbance) / 0.30  # normalised ∈ [0, 1]
        scale   = 1.0 + 0.8 * d_ratio           # ∈ [1.0, 1.8]

        reward = scale * -(
            1.0  * abs(e_n)     # tracking error (primary)
          + 0.3  * abs(de_n)    # oscillation damping
          + 0.5  * over         # overshoot
          + 0.15 * da           # smoothness
        ) + 0.8 * conv          # convergence bonus

        # Update states
        self.prev_error = self.current_error
        self.current_error = next_error
        self.prev_action = action
        self.steps += 1
        
        done = self.steps >= self.max_steps or abs(self.current_error) > 1000
        
        return self._get_obs(), reward, done, {}

if __name__ == "__main__":
    env = PIControllerEnv(r'e:\Caterpiller\project\models\surrogate_model.pth', r'e:\Caterpiller\project\data\scalers.pkl')
    obs = env.reset()
    print("Initial obs:", obs)
    obs, reward, done, info = env.step([1.0, 1.0])
    print("Next obs:", obs)
    print("Reward:", reward)
