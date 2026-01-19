"""
Deep Q-Network - PyTorch GPU Version
=====================================
Uses CUDA for GPU acceleration (with CPU fallback)

Requirements:
    pip install torch

Key Features:
1. Automatic GPU detection with fallback to CPU
2. PyTorch neural network
3. Target network with soft updates
4. Experience replay
5. Gradient clipping
6. Huber loss
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import env_settings.config as config
from data.utils import discretize_state
import time

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random

# Check for GPU with compatibility test
def get_device():
    """Get best available device with compatibility check"""
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        return torch.device("cpu")
    
    # Try to actually use the GPU
    try:
        # Test GPU with a simple operation
        test_tensor = torch.zeros(1).cuda()
        _ = test_tensor + 1
        del test_tensor
        torch.cuda.empty_cache()
        
        device = torch.device("cuda")
        print(f"✓ GPU available and compatible: {torch.cuda.get_device_name(0)}")
        return device
        
    except RuntimeError as e:
        if "no kernel image" in str(e) or "not compatible" in str(e):
            print(f"⚠️  GPU detected but not compatible with this PyTorch version")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   To enable GPU, install PyTorch nightly:")
            print(f"   pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu124")
            print(f"   Falling back to CPU...")
            return torch.device("cpu")
        else:
            raise e

device = get_device()
print(f"\n{'='*70}")
print(f"PyTorch Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"{'='*70}\n")


class DQNNetwork(nn.Module):
    """Neural network for DQN - Architecture from paper Figure 10"""
    
    def __init__(self, state_dim=3, n_actions=4):
        super(DQNNetwork, self).__init__()
        
        # Architecture: 3 → 128 → 64 → 16 → 4
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 16)
        self.fc4 = nn.Linear(16, n_actions)
        
        # Initialize with bias toward Power action
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier and bias toward forward motion"""
        for layer in [self.fc1, self.fc2, self.fc3, self.fc4]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        
        # Bias output toward Power action
        with torch.no_grad():
            self.fc4.bias[3] = 2.0   # Power
            self.fc4.bias[2] = 1.0   # Cruise
            self.fc4.bias[1] = 0.0   # Coast
            self.fc4.bias[0] = -1.0  # Brake
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class ReplayBuffer:
    """Experience replay buffer"""
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.FloatTensor(np.array(states)).to(device),
            torch.LongTensor(actions).to(device),
            torch.FloatTensor(rewards).to(device),
            torch.FloatTensor(np.array(next_states)).to(device),
            torch.FloatTensor(dones).to(device)
        )
    
    def __len__(self):
        return len(self.buffer)


class DeepQNetwork:
    """DQN Agent with PyTorch and GPU support"""
    
    def __init__(self, env, phi_threshold=0.10, state_dim=3):
        self.env = env
        self.phi = phi_threshold
        self.state_dim = state_dim
        self.n_actions = 4
        
        # Networks on GPU
        self.policy_net = DQNNetwork(state_dim, self.n_actions).to(device)
        self.target_net = DQNNetwork(state_dim, self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer with gradient clipping via max_norm
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=50000)
        self.batch_size = 64
        self.min_replay_size = 1000
        
        # Hyperparameters
        self.gamma = 0.95
        self.epsilon = 0.15
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.998
        self.target_update_freq = 20
        self.tau = 0.005  # Soft update rate
        
        # Reward scaling
        self.reward_scale = 0.01
        
        # History for CM calculation
        self.prev_params = self._get_params()
        self.delta_history = []
        self.cm_history = []
        self.reward_history = []
        self.energy_history = []
        self.time_history = []
        self.success_history = []
        self.loss_history = []
        
        self.use_sarsa_count = 0
        self.use_qlearning_count = 0
        
        # Count parameters
        total_params = sum(p.numel() for p in self.policy_net.parameters())
        
        print(f"Deep Q-Network Initialized (PyTorch GPU):")
        print(f"  Architecture: 3 → 128 → 64 → 16 → 4")
        print(f"  Total parameters: {total_params}")
        print(f"  Device: {device}")
        print(f"  φ threshold: {self.phi}")
        print(f"  Learning rate: 0.001")
        print(f"  γ (discount): {self.gamma}")
        print(f"  ε (exploration): {self.epsilon} → {self.epsilon_min}")
        print(f"  Replay buffer: 50,000")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Target update: soft τ={self.tau}")
    
    def _get_params(self):
        """Get flattened parameters for CM calculation"""
        return torch.cat([p.data.view(-1) for p in self.policy_net.parameters()]).cpu().numpy()
    
    def normalize_state(self, raw_state):
        """Normalize state to [0, 1] range"""
        seg_idx = int(raw_state[0])
        velocity = raw_state[1]
        
        pos_norm = seg_idx / self.env.n_segments
        vel_norm = velocity / config.MAX_SPEED_MS
        dist_to_end = 1.0 - pos_norm
        
        return np.array([pos_norm, vel_norm, dist_to_end], dtype=np.float32)
    
    def select_action(self, state):
        """ε-greedy action selection"""
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax(dim=1).item()
    
    def convergence_measurement(self):
        """Calculate CM based on parameter changes"""
        if len(self.delta_history) < 2:
            return 1.0
        
        delta_n = self.delta_history[-1]
        delta_n_minus_1 = self.delta_history[-2]
        
        if delta_n_minus_1 > 1e-9:
            return delta_n / delta_n_minus_1
        return 1.0
    
    def soft_update_target(self):
        """Soft update target network"""
        for target_param, policy_param in zip(self.target_net.parameters(), 
                                               self.policy_net.parameters()):
            target_param.data.copy_(
                self.tau * policy_param.data + (1.0 - self.tau) * target_param.data
            )
    
    def train_step(self):
        """Single training step on GPU"""
        if len(self.replay_buffer) < self.min_replay_size:
            return 0.0
        
        # Sample batch (already on GPU)
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Current Q-values
        q_values = self.policy_net(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # CM-based target calculation
        cm = self.convergence_measurement()
        
        with torch.no_grad():
            if cm > self.phi:
                # SARSA-style: use action from policy network
                next_actions = self.policy_net(next_states).argmax(dim=1)
                next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                self.use_sarsa_count += 1
            else:
                # Q-learning: use max Q-value
                next_q_values = self.target_net(next_states).max(dim=1)[0]
                self.use_qlearning_count += 1
            
            targets = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Huber loss (smooth L1)
        loss = F.smooth_l1_loss(q_values, targets)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, episodes=5000):
        """Train DQN on GPU"""
        print(f"\n{'='*70}")
        print(f"STARTING DEEP Q-NETWORK TRAINING (GPU)")
        print(f"{'='*70}")
        print(f"Total Episodes: {episodes}")
        print(f"Device: {device}")
        print(f"φ threshold: {self.phi}")
        print(f"Training starts after {self.min_replay_size} transitions")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        success_count = 0
        best_energy = float('inf')
        best_time = float('inf')
        
        for ep in range(1, episodes + 1):
            # Reset environment
            self.env.reset()
            raw_state = self.env._get_state()
            state = self.normalize_state(raw_state)
            
            total_reward = 0.0
            steps = 0
            episode_loss = 0.0
            loss_count = 0
            
            while steps < config.MAX_STEPS_PER_EPISODE:
                # Select action
                action = self.select_action(state)
                
                # Take action
                next_raw_state, reward, done, info = self.env.step(action)
                next_state = self.normalize_state(next_raw_state)
                
                total_reward += reward
                steps += 1
                
                # Scale and store reward
                scaled_reward = np.clip(reward * self.reward_scale, -10, 10)
                self.replay_buffer.push(state, action, scaled_reward, next_state, float(done))
                
                # Train
                loss = self.train_step()
                if loss > 0:
                    episode_loss += loss
                    loss_count += 1
                
                # Soft update target network every step
                self.soft_update_target()
                
                if done:
                    break
                
                if self.env.v < 0.1 and steps > 100:
                    break
                
                state = next_state
            
            # Calculate parameter delta for CM
            current_params = self._get_params()
            delta_sum = np.sum(np.abs(current_params - self.prev_params))
            self.delta_history.append(delta_sum)
            self.prev_params = current_params
            
            cm = self.convergence_measurement()
            self.cm_history.append(cm)
            
            # Check success
            episode_success = self.env.seg_idx >= self.env.n_segments - 2
            if episode_success:
                success_count += 1
                if info['energy'] < best_energy:
                    best_energy = info['energy']
                if info['time'] < best_time:
                    best_time = info['time']
            
            self.success_history.append(episode_success)
            self.reward_history.append(total_reward)
            self.energy_history.append(info.get('energy', 0))
            self.time_history.append(info.get('time', 0))
            avg_loss = episode_loss / max(1, loss_count)
            self.loss_history.append(avg_loss)
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Print progress
            if ep % 100 == 0 or ep <= 10:
                elapsed = time.time() - start_time
                eps_per_sec = ep / elapsed
                eta = (episodes - ep) / eps_per_sec if eps_per_sec > 0 else 0
                eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))
                
                marker = "✓" if episode_success else "✗"
                rate = (success_count / ep) * 100
                mode = "SARSA" if cm > self.phi else "Q-Learn"
                
                # GPU memory info
                if torch.cuda.is_available():
                    gpu_mem = torch.cuda.memory_allocated() / 1e6
                    gpu_str = f"GPU: {gpu_mem:.0f}MB"
                else:
                    gpu_str = "CPU"
                
                print(f"{marker} Ep {ep:04d}/{episodes} | "
                      f"Success: {rate:5.1f}% | "
                      f"Energy: {info.get('energy', 0):7.1f} kWh | "
                      f"Time: {info.get('time', 0):5.0f}s | "
                      f"Loss: {avg_loss:7.4f} | "
                      f"Mode: {mode:7s} | "
                      f"ε: {self.epsilon:.3f} | "
                      f"{gpu_str} | "
                      f"ETA: {eta_str}")
        
        elapsed_total = time.time() - start_time
        eps_per_sec = episodes / elapsed_total
        
        print(f"\n{'='*70}")
        print(f"DEEP Q-NETWORK TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"Total time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_total))}")
        print(f"Speed: {eps_per_sec:.1f} episodes/second")
        print(f"Final success rate: {(success_count/episodes)*100:.1f}%")
        print(f"SARSA updates: {self.use_sarsa_count}")
        print(f"Q-learning updates: {self.use_qlearning_count}")
        if best_energy < float('inf'):
            print(f"\nBest Performance:")
            print(f"  Best energy: {best_energy:.1f} kWh")
            print(f"  Best time: {best_time:.0f} s")
        
        self.save_results()
    
    def save_results(self):
        """Save results and model"""
        print("\n💾 Saving Deep-Q results...")
        
        output_dir = os.path.join(config.OUTPUT_DIR, "deep_q")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save PyTorch model
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, os.path.join(output_dir, "dqn_model.pt"))
        
        # Save history
        np.savez(os.path.join(output_dir, "dqn_history.npz"),
                 cm_history=np.array(self.cm_history),
                 reward_history=np.array(self.reward_history),
                 energy_history=np.array(self.energy_history),
                 time_history=np.array(self.time_history),
                 success_history=np.array(self.success_history),
                 loss_history=np.array(self.loss_history))
        
        # Plot
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        if self.success_history:
            cumulative = np.cumsum(self.success_history) / np.arange(1, len(self.success_history)+1) * 100
            axes[0, 0].plot(cumulative, linewidth=2, color='green')
        axes[0, 0].set_title('Success Rate (%)')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(self.loss_history, linewidth=1, alpha=0.7)
        axes[0, 1].set_title('Training Loss')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].grid(True, alpha=0.3)
        
        successful_eps = [i for i, s in enumerate(self.success_history) if s]
        if successful_eps:
            successful_energy = [self.energy_history[i] for i in successful_eps]
            axes[1, 0].plot(successful_eps, successful_energy, 'b.', alpha=0.5, markersize=2)
            if len(successful_energy) > 50:
                window = 50
                ma = np.convolve(successful_energy, np.ones(window)/window, mode='valid')
                axes[1, 0].plot(successful_eps[window-1:], ma, 'r-', linewidth=2, label='Moving Avg')
                axes[1, 0].legend()
        axes[1, 0].set_title('Energy (Successful Episodes)')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Energy (kWh)')
        axes[1, 0].grid(True, alpha=0.3)
        
        if successful_eps:
            successful_time = [self.time_history[i] for i in successful_eps]
            axes[1, 1].plot(successful_eps, successful_time, 'g.', alpha=0.5, markersize=2)
        axes[1, 1].set_title('Time (Successful Episodes)')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Time (s)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "dqn_training.png"), dpi=300)
        plt.close()
        
        print(f"✓ Model saved: {output_dir}/dqn_model.pt")
        print(f"✓ History saved: {output_dir}/dqn_history.npz")
        print(f"✓ Plot saved: {output_dir}/dqn_training.png")
    
    def load_model(self, path):
        """Load saved model"""
        checkpoint = torch.load(path, map_location=device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"✓ Model loaded from: {path}")
    
    def generate_speed_profile(self):
        """Generate optimal speed profile"""
        print("\n📊 Generating optimal speed profile (DQN)...")
        
        self.policy_net.eval()
        self.env.reset()
        
        segments = []
        velocities = []
        actions = []
        energies = []
        
        steps = 0
        with torch.no_grad():
            while steps < config.MAX_STEPS_PER_EPISODE:
                raw_state = self.env._get_state()
                state = self.normalize_state(raw_state)
                
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = self.policy_net(state_tensor)
                action = q_values.argmax(dim=1).item()
                
                segments.append(self.env.seg_idx)
                velocities.append(self.env.v)
                actions.append(action)
                energies.append(self.env.energy_kwh)
                
                _, _, done, _ = self.env.step(action)
                steps += 1
                
                if done or self.env.v < 0.1:
                    break
        
        self.policy_net.train()
        
        # Save
        output_dir = os.path.join(config.OUTPUT_DIR, "deep_q")
        np.savez(os.path.join(output_dir, "speed_profile.npz"),
                 segments=np.array(segments),
                 velocities=np.array(velocities),
                 actions=np.array(actions),
                 energies=np.array(energies))
        
        # Plot
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        axes[0].plot(segments, np.array(velocities) * 3.6, 'b-', linewidth=1)
        axes[0].set_ylabel('Speed (km/h)')
        axes[0].set_title('Optimal Speed Profile (DQN - GPU)')
        axes[0].grid(True, alpha=0.3)
        
        action_names = ['Brake', 'Coast', 'Cruise', 'Power']
        colors = ['red', 'orange', 'blue', 'green']
        for i, name in enumerate(action_names):
            mask = np.array(actions) == i
            if np.any(mask):
                axes[1].scatter(np.array(segments)[mask], [i]*np.sum(mask),
                              c=colors[i], s=1, label=name, alpha=0.5)
        axes[1].set_ylabel('Action')
        axes[1].set_yticks([0, 1, 2, 3])
        axes[1].set_yticklabels(action_names)
        axes[1].legend(loc='right')
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(segments, energies, 'g-', linewidth=1)
        axes[2].set_xlabel('Segment')
        axes[2].set_ylabel('Cumulative Energy (kWh)')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "speed_profile.png"), dpi=300)
        plt.close()
        
        print(f"✓ Speed profile saved")
        print(f"  Final energy: {energies[-1] if energies else 0:.1f} kWh")
        print(f"  Final segment: {segments[-1] if segments else 0}/{self.env.n_segments}")
        
        return segments, velocities, actions, energies