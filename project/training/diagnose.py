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
from training.pretrain import ReplayBuffer, pretrain_actor

def train_rl_pinn():
    # Paths
    surrogate_path = r'e:\Caterpiller\project\models\surrogate_model.pth'
    scalers_path = r'e:\Caterpiller\project\data\scalers.pkl'
    csv_path = r'e:\Caterpiller\ML_RL.csv'
    
    # Hyperparams
    state_dim = 3
    action_dim = 2
    max_action = 50.0
    batch_size = 128
    max_episodes = 100
    max_steps = 100
    lambda_pinn = 0.5 # Weight for physics loss
    
    # Env
    env = PIControllerEnv(surrogate_path, scalers_path)
    
    # Agent
    surrogate = SurrogateModel()
    surrogate.load_state_dict(torch.load(surrogate_path))
    surrogate.eval()
    
    with open(scalers_path, 'rb') as f:
        scalers = pickle.load(f)
        scaler_in = scalers['in']
        scaler_out = scalers['out']
    
    agent = TD3_PINN(state_dim, action_dim, max_action, surrogate, scaler_in, scaler_out)
    
    # 1. Pretraining (Behavior Cloning)
    agent.actor = pretrain_actor(agent.actor, csv_path, epochs=5)
    agent.actor_target.load_state_dict(agent.actor.state_dict())
    
    # 2. RL Training
    replay_buffer = ReplayBuffer(state_dim, action_dim)
    
    print("Starting RL training with PINN enhancement...")
    first_update = True
    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            action = agent.select_action(state)
            action = (action + np.random.normal(0, 0.1, size=action_dim)).clip(0.1, max_action)
            
            next_state, reward, done, _ = env.step(action)
            replay_buffer.add(state, action, next_state, reward, done)
            
            state = next_state
            episode_reward += reward
            
            if replay_buffer.size > batch_size:
                if first_update:
                    print(f"DIAGNOSTIC [4]: Replay buffer size at first update: {replay_buffer.size}")
                    first_update = False
                
                metrics = agent.train(replay_buffer, batch_size, lambda_pinn=lambda_pinn)
                
                if metrics and (step % 50 == 0):
                    pinn_ratio = metrics['pinn_loss'] / (abs(metrics['actor_loss']) + 1e-6)
                    print(f"DIAGNOSTIC [1-3]: Batch Step {step}")
                    print(f"  - kp_std: {metrics['kp_std']:.6f}, ki_std: {metrics['ki_std']:.6f} (Fail if -> 0)")
                    print(f"  - grad_norm: {metrics['grad_norm']:.6e} (Fail if -> 0)")
                    print(f"  - pinn_ratio: {pinn_ratio:.4f} (Fail if > 0.1)")
            
            if done:
                break
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}, Reward: {episode_reward:.2f}")
            
    # Save final model
    torch.save(agent.actor.state_dict(), r'e:\Caterpiller\project\models\actor_pi_pinn.pth')
    print("RL-PINN model saved.")

if __name__ == "__main__":
    train_rl_pinn()
