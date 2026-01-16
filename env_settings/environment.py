import numpy as np
import env_settings.config as config

class TrainEnv:
    """
    Train Environment for RL Speed Profile Optimization
    Based on Section 2 (Train Motion Model) from the paper
    """
    
    def __init__(self, physics, grades, limits, curves):
        self.phy = physics
        self.grades = grades
        self.limits = limits
        self.curves = curves
        
        self.n_segments = len(grades)
        self.action_space = 4
        
        assert len(limits) == self.n_segments, f"Limits mismatch: {len(limits)} vs {self.n_segments}"
        assert len(curves) == self.n_segments, f"Curves mismatch: {len(curves)} vs {self.n_segments}"
        
        print(f"Environment initialized:")
        print(f"  Segments: {self.n_segments}")
        print(f"  Total distance: {self.n_segments * config.DX / 1000:.1f} km")
        print(f"  Grade range: [{min(grades):.2f}%, {max(grades):.2f}%]")
        print(f"  Speed limit range: [{min(limits):.1f}, {max(limits):.1f}] m/s")
        
        self.reset()

    def reset(self):
        """Reset environment to initial state"""
        self.seg_idx = 0
        self.pos_in_seg = 0.0
        
        # Start at reasonable operational speed
        self.v = 18.0  # ~65 km/h - not too fast, not too slow
        
        self.t = 0.0
        self.energy_kwh = 0.0
        self.done = False
        self.max_segment_reached = 0
        
        return self._get_state()

    def _get_state(self):
        """Return current state [segment_index, velocity]"""
        return np.array([self.seg_idx, self.v], dtype=np.float32)

    def step(self, action):
        """
        Execute one step in the environment
        
        Actions:
            0: Brake (full deceleration)
            1: Coast (no traction, no brake)
            2: Cruise (maintain speed)
            3: Power (full acceleration)
        """
        if self.done:
            return self._get_state(), 0, True, {}

        old_segment = self.seg_idx
        old_position_km = (self.seg_idx * config.DX + self.pos_in_seg) / 1000.0

        # Get current track conditions
        grade = self.grades[self.seg_idx]
        curve = self.curves[self.seg_idx]
        limit = self.limits[self.seg_idx]

        # Calculate forces
        resistance = self.phy.get_total_resistance(self.v, grade, curve)
        f_trac = 0.0
        f_brake = 0.0

        # Action interpretation (Figure 2 from paper)
        if action == 0:  # Brake
            f_brake = abs(self.phy.mass * config.MAX_DEC)
        elif action == 1:  # Coast
            pass  # No traction, no brake
        elif action == 2:  # Cruise
            f_trac = max(0, resistance)  # Match resistance
        elif action == 3:  # Power
            f_trac = self.phy.get_max_traction_force(self.v)

        # Physics (Equation 6 from paper)
        f_net = f_trac - f_brake - resistance
        acc = np.clip(f_net / self.phy.mass, config.MAX_DEC, config.MAX_ACC)
        
        # Update velocity
        v_next = np.clip(self.v + acc * config.DT, 0.0, config.MAX_SPEED_MS)
        
        # Update position (using average velocity)
        avg_velocity = (self.v + v_next) / 2.0
        ds = max(0.0, avg_velocity * config.DT)
        
        self.pos_in_seg += ds

        # Calculate energy consumption
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
                self.pos_in_seg = 0.0
                break

        # Track best progress
        if self.seg_idx > self.max_segment_reached:
            self.max_segment_reached = self.seg_idx

        # Current position in km
        self.position_km = (self.seg_idx * config.DX + self.pos_in_seg) / 1000.0

        # ============================================================
        # REWARD FUNCTION
        # ============================================================
        # Goal: Balance between completing route, saving time, saving energy
        # Based on Equation 45 from paper
        
        reward = 0.0
        distance_moved_km = self.position_km - old_position_km
        
        # 1. Progress reward - encourage forward movement
        progress_reward = distance_moved_km * 150  # Moderate scaling
        reward += progress_reward
        
        # 2. Forward segment bonus - reward advancing to new segments
        if self.seg_idx > old_segment:
            reward += 3.0  # Small bonus per segment
        
        # 3. Energy penalty - discourage excessive power use
        reward -= e_kwh_step * 15.0
        
        # 4. Time penalty - small constant
        reward -= 0.02
        
        # 5. Speed limit violation - strong penalty
        if self.v > limit:
            violation = self.v - limit
            reward -= 50.0 + violation * 10.0
        
        # 6. Penalty for being too slow (stuck)
        if self.v < 2.0 and not self.done:
            reward -= 5.0
        
        # 7. Near-completion bonus (helps agent learn the final steps)
        progress_pct = self.seg_idx / self.n_segments
        if progress_pct > 0.95 and not self.done:
            # Extra reward for being close to goal
            reward += 2.0
        
        # 8. SUCCESS REWARD - completing the route
        if self.done and self.seg_idx >= self.n_segments - 1:
            # Base success bonus
            reward += 3000.0
            
            # Time efficiency bonus
            expected_time = self.n_segments * config.DX / 20.0  # Expected at 20 m/s
            if self.t < expected_time:
                reward += 500.0
            
            # Energy efficiency bonus
            expected_energy = self.n_segments * config.DX * 0.022  # ~22 kWh/km
            if self.energy_kwh < expected_energy:
                reward += 300.0

        # Info dictionary for debugging
        info = {
            'segment': self.seg_idx,
            'velocity': self.v,
            'time': self.t,
            'energy': self.energy_kwh,
            'position_in_segment': self.pos_in_seg,
            'distance_moved': ds,
            'acceleration': acc,
            'speed_limit': limit,
            'grade': grade,
            'violation': self.v > limit,
            'progress_percent': (self.seg_idx / self.n_segments) * 100
        }

        return self._get_state(), reward, self.done, info