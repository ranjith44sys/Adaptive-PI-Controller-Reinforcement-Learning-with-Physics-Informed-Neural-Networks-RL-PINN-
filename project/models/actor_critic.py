import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        raw = torch.tanh(self.l3(x))
        # action = 0.5 + (raw + 1.0) * 0.75  → [0.5, 2.0]
        return 0.5 + (raw + 1.0) * 0.75

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        
        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)
        
    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        
        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

class TD3_PINN:
    def __init__(self, state_dim, action_dim, max_action, surrogate_model, scaler_in, scaler_out):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)
        
        self.max_action = max_action
        self.surrogate = surrogate_model.to(self.device)
        self.scaler_in = scaler_in
        self.scaler_out = scaler_out
        
    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()
    
    def train(self, replay_buffer, batch_size=256, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2, lambda_pinn=0.01, lambda_div=0.25, total_it=0):
        # Sample from prioritized replay buffer
        state, action, next_state, reward, not_done, weights, indices = replay_buffer.sample(batch_size)
        
        with torch.no_grad():
            noise = (torch.randn_like(action) * policy_noise).clamp(-noise_clip, noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(0.5, 2.0)
            
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * discount * target_Q
            
        current_Q1, current_Q2 = self.critic(state, action)
        
        # Section 6 — PER Critic Loss
        td_errors1 = current_Q1 - target_Q
        td_errors2 = current_Q2 - target_Q
        critic_loss = (weights * td_errors1**2).mean() + (weights * td_errors2**2).mean()
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update priorities in buffer
        td_errors = (torch.abs(td_errors1) + torch.abs(td_errors2)).detach().cpu().numpy() / 2.0
        replay_buffer.update_priorities(indices, td_errors)
        
        # Delayed policy updates
        if total_it % policy_freq == 0:
            actions = self.actor(state)
            rl_loss = -self.critic.Q1(state, actions).mean()
            
            # Anti-collapse diversity loss
            diversity_loss = -(actions[:, 0].std() + actions[:, 1].std())
            
            # PINN Loss: Next error prediction
            # Reconstruct surrogate input.
            # State is: [e/setp, de/setp, dist/max_dist, e_int, prev_Kp_centred, prev_Ki_centred]
            # Surrogate expects: [e_raw, dist_raw, Kp_corr, Ki_corr]
            
            setpoint = 900.0
            max_dist = 500.0
            
            error = state[:, 0:1]
            dist = state[:, 2:3]
            kp_corr = actions[:, 0:1]
            ki_corr = actions[:, 1:2]
            
            e_raw = error * setpoint
            dist_raw = dist * max_dist # d normalized is dist/max_dist
            
            # Form raw input batch
            raw_input = torch.cat([e_raw, dist_raw, kp_corr, ki_corr], 1)
            
            # Apply scaling as per preprocess.py (StandardScaler)
            # Since scaler_in is an sklearn object, we'll assume it's moved to the device or handle it carefully.
            # In a real PyTorch training loop, we should use a Torch-based scaler or pre-compute stats.
            # For now, we'll convert to numpy, scale, and back, OR implement the scaling logic in Torch.
            
            # Standard Scaling: (x - mean) / std
            mean = torch.FloatTensor(self.scaler_in.mean_).to(self.device)
            scale = torch.FloatTensor(self.scaler_in.scale_).to(self.device)
            scaled_input = (raw_input - mean) / (scale + 1e-6)
            
            next_error_pred_scaled = self.surrogate(scaled_input)
            
            # We want next_error to be 0. We can penalize the scaled next_error directly.
            pinn_loss = torch.mean(torch.abs(next_error_pred_scaled))
            
            # Total Actor Loss
            actor_loss = rl_loss + lambda_pinn * pinn_loss + lambda_div * diversity_loss
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            
            # Diagnostic: Compute gradient norm
            grad_norm = 0
            for p in self.actor.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
            
            self.actor_optimizer.step()
            
            # Update targets
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                
            return {
                "kp_std": actions[:, 0].std().item(),
                "ki_std": actions[:, 1].std().item(),
                "actor_loss": actor_loss.item(),
                "pinn_loss": pinn_loss.item(),
                "pinn_ratio": pinn_loss.item() / (abs(rl_loss.item()) + 1e-6),
                "grad_norm": grad_norm
            }
        return None
