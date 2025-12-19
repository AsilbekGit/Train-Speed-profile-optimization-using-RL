import numpy as np
import matplotlib.pyplot as plt
import os
import config
from utils import discretize_state
import time

class CMAnalyzer:
    def __init__(self, env):
        self.env = env
        # Q-Table dimensions: [Segments, Velocity_Bins, Actions]
        self.q_shape = (env.n_segments, 100, 4)
        self.q_curr = np.zeros(self.q_shape)
        self.q_prev = np.zeros(self.q_shape)
        
        # History
        self.delta_history = []
        self.cm_history = []
        
        # Hyperparams
        self.alpha = 0.1
        self.gamma = 0.95
        self.epsilon = 0.15

    def run(self, episodes=2000):
        print(f"Starting CM Analysis for {episodes} episodes...")
        start_time = time.time()
        
        for ep in range(1, episodes + 1):
            self.env.reset()
            state = self.env._get_state()
            s_idx, v_idx = discretize_state(state)
            
            total_reward = 0.0  # Track score for this episode
            steps = 0
            
            # Initial Action
            if np.random.rand() < self.epsilon:
                action = np.random.randint(4)
            else:
                action = np.argmax(self.q_curr[s_idx, v_idx])
            
            while True:
                next_state_raw, reward, done, _ = self.env.step(action)
                ns_idx, nv_idx = discretize_state(next_state_raw)
                
                total_reward += reward
                steps += 1
                
                # Next Action (SARSA Logic)
                if np.random.rand() < self.epsilon:
                    next_action = np.random.randint(4)
                else:
                    next_action = np.argmax(self.q_curr[ns_idx, nv_idx])
                
                # Update Q
                target = reward + self.gamma * self.q_curr[ns_idx, nv_idx, next_action]
                if done:
                    target = reward
                
                self.q_curr[s_idx, v_idx, action] += self.alpha * (target - self.q_curr[s_idx, v_idx, action])
                
                s_idx, v_idx = ns_idx, nv_idx
                action = next_action
                
                if done:
                    break
            
            # --- Calculate Delta and CM ---
            # 1. Delta: Sum of absolute differences between Current Q and Previous Q
            diff = np.sum(np.abs(self.q_curr - self.q_prev))
            self.delta_history.append(diff)
            
            # 2. Update Previous Q
            self.q_prev = self.q_curr.copy()
            
            # 3. Calculate CM Ratio
            if len(self.delta_history) >= 2:
                delta_n = self.delta_history[-1]
                delta_n_minus_1 = self.delta_history[-2]
                
                if delta_n_minus_1 > 1e-9:
                    cm = delta_n / delta_n_minus_1
                else:
                    cm = 0.0
                
                # Cap for plotting safety
                cm = min(cm, 3.0)
                self.cm_history.append(cm)
            else:
                self.cm_history.append(1.0)
                cm = 1.0

            # --- PRINT STATUS EVERY EPISODE ---
            # Calculates estimated time remaining
            elapsed = time.time() - start_time
            avg_time_per_ep = elapsed / ep
            remaining = (episodes - ep) * avg_time_per_ep
            rem_str = time.strftime("%H:%M:%S", time.gmtime(remaining))

            # Dynamic printing
            print(f"Ep {ep:04d}/{episodes} | "
                  f"Reward: {total_reward:9.2f} | "
                  f"Delta: {diff:8.4f} | "
                  f"CM: {cm:.4f} | "
                  f"ETA: {rem_str}")

        self.save_plot()

    def save_plot(self):
        print("\nSaving results...")
        plt.figure(figsize=(10, 6))
        plt.plot(self.cm_history)
        plt.title("Convergence Measurement (CM) Analysis")
        plt.xlabel("Episodes")
        plt.ylabel("CM Ratio")
        plt.axhline(y=0.5, color='r', linestyle='--', label='Likely Phi Threshold')
        plt.legend()
        plt.grid(True)
        
        save_path = os.path.join(config.OUTPUT_DIR, "cm_plot.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Plot saved successfully to {save_path}")
