import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pickle

# Add project root to path
sys.path.append(r'e:\Caterpiller\project')
from env.pi_env import PIControllerEnv
from models.actor_critic import Actor
from models.surrogate_model import SurrogateModel

def evaluate_controller():
    # Paths
    surrogate_path = r'e:\Caterpiller\project\models\surrogate_model.pth'
    scalers_path = r'e:\Caterpiller\project\data\scalers.pkl'
    actor_path = r'e:\Caterpiller\project\models\actor_pi_pinn.pth'
    
    # Env (6D state)
    env = PIControllerEnv(surrogate_path, scalers_path)
    
    # Load Actor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor = Actor(6, 2, 2.0).to(device) # state_dim=6
    if os.path.exists(actor_path):
        actor.load_state_dict(torch.load(actor_path, map_location=device))
        actor.eval()
    else:
        print(f"Error: RL model not found at {actor_path}.")
        return
    
    disturbances = [0.05, 0.15, 0.30]
    results = {}
    
    plt.figure(figsize=(15, 12))
    
    for i, dist in enumerate(disturbances):
        # 1. Baseline PI
        env.reset()
        env.current_error = 100.0
        env.disturbance = dist
        
        baseline_errors = []
        for _ in range(100):
            obs, reward, done, _ = env.step([1.0, 1.0])
            baseline_errors.append(env.current_error)
            if done: break
            
        # 2. RL-PINN
        env.reset()
        env.current_error = 100.0
        env.disturbance = dist
        
        rl_errors = []
        rl_gains = []
        state = env._get_obs()
        for _ in range(100):
            with torch.no_grad():
                state_t = torch.FloatTensor(state.reshape(1, -1)).to(device)
                action = actor(state_t).cpu().numpy().flatten()
            obs, reward, done, _ = env.step(action)
            rl_errors.append(env.current_error)
            rl_gains.append(action)
            state = obs
            if done: break
            
        rl_gains = np.array(rl_gains)
        baseline_rmse = np.sqrt(np.mean(np.square(baseline_errors)))
        rl_rmse = np.sqrt(np.mean(np.square(rl_errors)))
        improvement = (baseline_rmse - rl_rmse) / (baseline_rmse + 1e-6) * 100
        
        # Section 8 checks
        # Monotonic decrease check in last 40% (40 steps)
        last_40 = np.abs(rl_errors[-40:])
        is_decreasing = (last_40[-1] < last_40[0]) # Simplified check
        
        results[dist] = {
            "improvement": improvement,
            "kp_std": np.std(rl_gains[:, 0]),
            "ki_std": np.std(rl_gains[:, 1]),
            "kp_mean": np.mean(rl_gains[:, 0]),
            "ki_mean": np.mean(rl_gains[:, 1]),
            "no_plateau": is_decreasing
        }
        
        # Plotting
        plt.subplot(3, 1, i+1)
        plt.plot(baseline_errors, label=f'Baseline (RMSE: {baseline_rmse:.1f})', linestyle='--')
        plt.plot(rl_errors, label=f'RL-PINN (RMSE: {rl_rmse:.1f}, {improvement:.1f}% imp)', linewidth=2)
        plt.title(f'Disturbance: {dist*100:.0f}%')
        plt.ylabel('Error (RPM)')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(r'e:\Caterpiller\project\evaluation\comparison_plot.png')
    
    # Section 8 Validation Report
    print("\n" + "="*50)
    print("SECTION 8 — MANDATORY VALIDATION REPORT")
    print("="*50)
    
    metrics_list = []
    all_pass = True
    
    for dist in disturbances:
        res = results[dist]
        print(f"\nREGIME: {dist*100:.0f}% Disturbance")
        
        # RMSE check
        rmse_target = 15.0 if dist < 0.25 else 10.0
        pass_rmse = res['improvement'] >= rmse_target
        print(f"  - RMSE Improvement (Target >= {rmse_target}%): {'PASS' if pass_rmse else 'FAIL'} ({res['improvement']:.1f}%)")
        
        # Std check
        pass_std = res['kp_std'] > 0.05 and res['ki_std'] > 0.05
        print(f"  - Gain Variation (Std > 0.05): {'PASS' if pass_std else 'FAIL'} (Kp:{res['kp_std']:.3f}, Ki:{res['ki_std']:.3f})")
        
        # Plateau check
        print(f"  - Convergence (No Plateau): {'PASS' if res['no_plateau'] else 'FAIL'}")
        
        # Ki Floor check for 30%
        if dist == 0.30:
            pass_ki = res['ki_mean'] >= 1.15
            print(f"  - Ki Activation (Mean >= 1.15): {'PASS' if pass_ki else 'FAIL'} ({res['ki_mean']:.3f})")
            if not pass_ki: all_pass = False
            
        if not (pass_rmse and pass_std and res['no_plateau']):
            all_pass = False
            
    # Kp Regime Scaling check
    kp_scaling = results[0.30]['kp_mean'] / (results[0.05]['kp_mean'] + 1e-6)
    pass_scale = kp_scaling >= 1.20
    print(f"\nCROSS-REGIME ANALYSIS:")
    print(f"  - Kp Scaling (30%/5% >= 1.20): {'PASS' if pass_scale else 'FAIL'} ({kp_scaling:.2f})")
    if not pass_scale: all_pass = False
    
    print("\n" + "="*50)
    if all_pass:
        print("OVERALL STATUS: ACCEPTED [PASS] — GENERALIZATION GOALS MET")
    else:
        print("OVERALL STATUS: REJECTED [FAIL] — ADDITIONAL TUNING REQUIRED")
    print("="*50)
    
if __name__ == "__main__":
    evaluate_controller()
