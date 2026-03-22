import torch
import numpy as np
import os
import sys
import pickle

# Add project root to path
sys.path.append(r'e:\Caterpiller\project')
from env.pi_env import PIControllerEnv
from models.actor_critic import Actor

def diagnose_regression():
    surrogate_path = r'e:\Caterpiller\project\models\surrogate_model.pth'
    scalers_path = r'e:\Caterpiller\project\data\scalers.pkl'
    actor_path = r'e:\Caterpiller\project\models\actor_pi_pinn.pth'
    
    env = PIControllerEnv(surrogate_path, scalers_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor = Actor(6, 2, 2.0).to(device)
    
    if os.path.exists(actor_path):
        actor.load_state_dict(torch.load(actor_path, map_location=device))
        actor.eval()
    
    disturbances = [0.05, 0.15, 0.30]
    
    print("SECTION 0 — REGRESSION DIAGNOSIS")
    print("="*40)
    
    for dist in disturbances:
        env.reset()
        env.disturbance = dist
        w_error, w_delta, w_overshoot, w_smooth, w_converge = env.compute_regime_weights(dist)
        weights = [w_error, w_delta, w_overshoot, w_smooth, w_converge]
        ratio = max(weights) / (min(weights) + 1e-6)
        print(f"Disturbance {dist*100:.0f}% | Weight Ratio: {ratio:.1f}x (Target < 8x)")
        
    print("\nGAIN DISTRIBUTION (1000 steps across regimes):")
    all_gains = []
    for _ in range(1000):
        d = np.random.uniform(0.02, 0.32)
        state = np.zeros(6, dtype=np.float32)
        state[2] = d / 0.30 # normalized
        # rest zeros for simplicity or random?
        state[0] = np.random.normal(0, 0.1) # normalized error
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state.reshape(1, -1)).to(device)
            action = actor(state_t).cpu().numpy().flatten()
            all_gains.append(action)
            
    all_gains = np.array(all_gains)
    kp_low = np.sum(all_gains[:, 0] < 0.55) / 1000
    ki_low = np.sum(all_gains[:, 1] < 0.55) / 1000
    
    print(f"Kp clustered at floor (<0.55): {kp_low*100:.1f}%")
    print(f"Ki clustered at floor (<0.55): {ki_low*100:.1f}%")
    if kp_low > 0.5 or ki_low > 0.5:
        print("CONFIRMED: Gains are binding to floors.")

if __name__ == "__main__":
    diagnose_regression()
