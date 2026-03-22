import torch
import numpy as np
import os
import sys
import pickle

# Add project root to path
sys.path.append(r'e:\Caterpiller\project')
from env.pi_env import PIControllerEnv
from models.actor_critic import TD3_PINN
from models.surrogate_model import SurrogateModel
from training.pretrain import PrioritizedReplayBuffer

class UniformSampler:
    def sample_disturbance(self):
        return np.random.uniform(0.02, 0.32)

class DisturbanceTracker:
    def __init__(self):
        self.bins = [0, 0, 0]
    def log(self, d):
        if d < 0.10: self.bins[0] += 1
        elif d < 0.20: self.bins[1] += 1
        else: self.bins[2] += 1
    def coverage_ok(self):
        total = sum(self.bins) + 1e-6
        return all(b / total > 0.20 for b in self.bins)

class AntiPlateauReward:
    def __init__(self, window=15, threshold=0.015):
        self.window    = window
        self.threshold = threshold
        self.buf       = []

    def step(self, error_norm):
        self.buf.append(abs(error_norm))
        if len(self.buf) > self.window: self.buf.pop(0)

    def penalty(self, base_error_term):
        if len(self.buf) < self.window: return 0.0
        old_avg = sum(self.buf[:5]) / 5
        new_avg = sum(self.buf[-5:]) / 5
        rel_imp = (old_avg - new_avg) / (old_avg + 1e-6)
        if rel_imp < self.threshold:
            severity = (self.threshold - rel_imp) / self.threshold
            return -0.30 * base_error_term * severity  # max -0.30
        return 0.0

def train_rl_pinn():
    surrogate_path = r'e:\Caterpiller\project\models\surrogate_model.pth'
    scalers_path = r'e:\Caterpiller\project\data\scalers.pkl'
    
    state_dim = 6
    action_dim = 2
    batch_size = 256
    max_episodes = 2000
    max_steps = 100
    warmup_steps = 10000
    total_steps = 0
    lambda_pinn = 0.02
    lambda_div = 0.10
    
    sampler = UniformSampler()
    tracker = DisturbanceTracker()
    anti_plateau = AntiPlateauReward()
    env = PIControllerEnv(surrogate_path, scalers_path)
    
    surrogate = SurrogateModel()
    surrogate.load_state_dict(torch.load(surrogate_path))
    surrogate.eval()
    
    with open(scalers_path, 'rb') as f:
        scalers = pickle.load(f)
        scaler_in = scalers['in']
    
    agent = TD3_PINN(state_dim, action_dim, 2.0, surrogate, scaler_in, None)
    replay_buffer = PrioritizedReplayBuffer(state_dim, action_dim, max_size=200000)
    
    print("Starting Recalibrated RL-PINN training (Uniform Sampling + Smooth Anti-Plateau)...")
    for episode in range(max_episodes):
        state = env.reset()
        env.disturbance = sampler.sample_disturbance()
        tracker.log(env.disturbance)
        
        episode_reward = 0
        anti_plateau.buf = []
        
        for step in range(max_steps):
            total_steps += 1
            
            if total_steps < warmup_steps:
                action = env.action_space.sample()
            else:
                sigma = max(0.05, 0.3 * np.exp(-3e-5 * total_steps))
                action = agent.select_action(state)
                action = (action + np.random.normal(0, sigma, size=action_dim)).clip(0.5, 2.0)
            
            next_state, reward, done, _ = env.step(action)
            
            # Section 5 — Anti-Plateau Penalty
            e_n = next_state[0]
            anti_plateau.step(e_n)
            base_error_term = 1.0 * abs(e_n)
            penalty = anti_plateau.penalty(base_error_term)
            total_reward = reward + penalty
            
            replay_buffer.add(state, action, next_state, total_reward, done, env.disturbance)
            state = next_state
            episode_reward += total_reward
            
            if replay_buffer.size > batch_size and total_steps >= 2000:
                metrics = agent.train(
                    replay_buffer, 
                    batch_size=batch_size, 
                    lambda_pinn=lambda_pinn, 
                    lambda_div=lambda_div,
                    total_it=total_steps
                )
                
                # Section 6 — PINN Ratio Enforcement
                if metrics and total_steps % 2000 == 0:
                    rl_term = abs(metrics.get('actor_loss', 0.0))
                    pinn_loss = metrics.get('pinn_loss', 0.0)
                    div_loss = metrics.get('div_loss', 0.0)
                    
                    pinn_ratio = (lambda_pinn * pinn_loss) / (rl_term + 1e-6)
                    div_ratio = (lambda_div * div_loss) / (rl_term + 1e-6)
                    
                    print(f"Step {total_steps} | PINN Ratio: {pinn_ratio:.4f} | Div Ratio: {div_ratio:.4f}")
                    
                    if pinn_ratio > 0.05:
                        print("WARNING: PINN ratio > 0.05. Reducing lambda_pinn to 0.01.")
                        lambda_pinn = 0.01
                        
            if done:
                break
        
        if (episode + 1) % 50 == 0:
            coverage = tracker.coverage_ok()
            print(f"Ep {episode+1} | Steps {total_steps} | Avg Reward: {episode_reward/max_steps:.2f} | Bins: {tracker.bins} | Coverage: {coverage}")
            torch.save(agent.actor.state_dict(), r'e:\Caterpiller\project\models\actor_pi_pinn.pth')
            
    torch.save(agent.actor.state_dict(), r'e:\Caterpiller\project\models\actor_pi_pinn.pth')
    print("Recalibrated RL-PINN model saved.")

if __name__ == "__main__":
    train_rl_pinn()
