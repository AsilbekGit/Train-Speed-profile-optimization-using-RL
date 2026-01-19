"""
Train Environment - FIXED FOR CM ANALYSIS
==========================================
Key Fixes:
1. Simplified reward function - focuses on COMPLETION, not energy
2. Fixed speed limit contradiction
3. Better done condition
"""

import numpy as np
import env_settings.config as config

class TrainEnv:
    def __init__(self, physics, grades, limits, curves):
        self.phy = physics
        self.grades = grades
        self.limits = limits
        self.curves = curves
        
        self.n_segments = len(grades)
        self.action_space = 4
        
        assert len(limits) == self.n_segments
        assert len(curves) == self.n_segments
        
        print(f"Environment initialized:")
        print(f"  Segments: {self.n_segments}")
        print(f"  Total distance: {self.n_segments * config.DX / 1000:.1f} km")
        print(f"  Grade range: [{min(grades):.2f}%, {max(grades):.2f}%]")
        print(f"  Speed limit range: [{min(limits):.1f}, {max(limits):.1f}] m/s")
        
        self.reset()

    def reset(self):
        self.seg_idx = 0
        self.pos_in_seg = 0.0
        self.v = 15.0  # Start at reasonable speed
        self.t = 0.0
        self.energy_kwh = 0.0
        self.done = False
        self.max_segment_reached = 0
        self.total_distance = 0.0
        return self._get_state()

    def _get_state(self):
        return np.array([self.seg_idx, self.v], dtype=np.float32)

    def step(self, action):
        if self.done:
            return self._get_state(), 0, True, {}

        old_segment = self.seg_idx
        old_v = self.v

        # Get current track conditions
        grade = self.grades[self.seg_idx]
        curve = self.curves[self.seg_idx]
        limit = self.limits[self.seg_idx]

        # Calculate forces
        resistance = self.phy.get_total_resistance(self.v, grade, curve)
        f_trac = 0.0
        f_brake = 0.0

        # Action interpretation
        if action == 0:  # Brake
            f_brake = abs(self.phy.mass * config.MAX_DEC)
        elif action == 1:  # Coast
            pass
        elif action == 2:  # Cruise
            f_trac = max(0, resistance)
        elif action == 3:  # Power
            f_trac = self.phy.get_max_traction_force(self.v)

        # Physics
        f_net = f_trac - f_brake - resistance
        acc = np.clip(f_net / self.phy.mass, config.MAX_DEC, config.MAX_ACC)
        
        # Update velocity
        v_next = np.clip(self.v + acc * config.DT, 0.0, config.MAX_SPEED_MS)
        
        # Update position
        avg_velocity = (self.v + v_next) / 2.0
        ds = max(0.0, avg_velocity * config.DT)
        
        self.pos_in_seg += ds
        self.total_distance += ds

        # Energy consumption
        if f_trac > 0 and self.v > 0:
            p_elec_watts = (f_trac * self.v) / self.phy.eta
            e_kwh_step = (p_elec_watts * config.DT) / 3.6e6
        else:
            e_kwh_step = 0.0
        
        self.energy_kwh += e_kwh_step
        self.v = v_next
        self.t += config.DT

        # Segment transition
        while self.pos_in_seg >= config.DX:
            self.pos_in_seg -= config.DX
            self.seg_idx += 1
            
            if self.seg_idx >= self.n_segments:
                self.done = True
                self.seg_idx = self.n_segments - 1
                break

        # Track best progress
        if self.seg_idx > self.max_segment_reached:
            self.max_segment_reached = self.seg_idx

        # ============================================================
        # SIMPLIFIED REWARD FOR CM ANALYSIS
        # ============================================================
        # Focus on COMPLETION, not energy efficiency
        # Energy optimization comes AFTER finding φ
        
        reward = 0.0
        
        # 1. Progress reward - simple distance moved
        reward += ds * 0.1  # Small reward per meter
        
        # 2. Segment advancement bonus
        if self.seg_idx > old_segment:
            reward += 2.0
        
        # 3. Velocity maintenance - reward keeping speed
        if self.v > 5.0:
            reward += 0.1
        
        # 4. Speed limit violation - but REASONABLE penalty
        if self.v > limit + 2.0:  # Allow 2 m/s buffer
            reward -= 5.0
        
        # 5. Stopped penalty - discourage stopping
        if self.v < 1.0:
            reward -= 10.0
        
        # 6. COMPLETION BONUS - this is what matters!
        if self.done and self.seg_idx >= self.n_segments - 1:
            reward += 1000.0  # Big bonus for completion
        
        # NO energy penalty during CM analysis!
        # (Energy optimization is for later, after φ is found)

        info = {
            'segment': self.seg_idx,
            'velocity': self.v,
            'time': self.t,
            'energy': self.energy_kwh,
            'grade': grade,
            'speed_limit': limit,
            'progress_percent': (self.seg_idx / self.n_segments) * 100
        }

        return self._get_state(), reward, self.done, info