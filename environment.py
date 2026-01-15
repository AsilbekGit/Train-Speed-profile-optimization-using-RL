import numpy as np
import config

class TrainEnv:
    def __init__(self, physics, grades, limits, curves):
        self.phy = physics
        self.grades = grades
        self.limits = limits
        self.curves = curves
        
        self.n_segments = len(grades)
        self.action_space = 4 # Brake, Coast, Cruise, Power
        
        # Validation
        assert len(limits) == self.n_segments, "Mismatch: limits length"
        assert len(curves) == self.n_segments, "Mismatch: curves length"
        
        print(f"Environment initialized:")
        print(f"  Segments: {self.n_segments}")
        print(f"  Actions: {self.action_space}")
        print(f"  DX: {config.DX}m")
        print(f"  DT: {config.DT}s")
        
        self.reset()

    def reset(self):
        """Reset environment to initial state"""
        self.seg_idx = 0
        self.pos_in_seg = 0.0
        
        # SIGNIFICANTLY INCREASED: Start at operational speed
        # Paper assumes train accelerates to x_start first
        # Need higher speed to handle uphills without getting stuck
        self.v = 25.0  # 90 km/h - strong operational speed
        
        self.t = 0.0
        self.energy_kwh = 0.0
        self.done = False
        return self._get_state()

    def _get_state(self):
        """Return state as numpy array [segment_index, velocity]"""
        return np.array([self.seg_idx, self.v], dtype=np.float32)

    def step(self, action):
        """
        Execute one step in the environment
        
        Args:
            action: 0=Brake, 1=Coast, 2=Cruise, 3=Power
            
        Returns:
            next_state, reward, done, info
        """
        if self.done:
            return self._get_state(), 0, True, {}

        # Calculate OLD position (before step) in km
        old_position_km = (self.seg_idx * config.DX + self.pos_in_seg) / 1000.0

        # Get current segment data
        grade = self.grades[self.seg_idx]
        curve = self.curves[self.seg_idx]
        limit = self.limits[self.seg_idx]

        # Calculate Forces
        resistance = self.phy.get_total_resistance(self.v, grade, curve)
        f_trac = 0.0
        f_brake = 0.0

        if action == 0:  # Full Brake
            f_brake = abs(self.phy.mass * config.MAX_DEC)
        
        elif action == 1:  # Coast
            pass  # No traction or braking, only resistance
            
        elif action == 2:  # Cruise (maintain speed)
            # FIXED: Always apply traction to counter resistance
            # Don't disable traction on downhill
            f_trac = max(0, resistance)  # Apply force to maintain speed
                
        elif action == 3:  # Max Power
            f_trac = self.phy.get_max_traction_force(self.v)

        # Net Force
        f_net = f_trac - f_brake - resistance
        
        # Calculate Acceleration (F = ma)
        acc = f_net / self.phy.mass
        acc = np.clip(acc, config.MAX_DEC, config.MAX_ACC)

        # Update Velocity using kinematic equation
        v_next = self.v + acc * config.DT
        v_next = max(0.0, v_next)  # Velocity can't be negative
        
        # Cap at maximum speed
        v_next = min(v_next, config.MAX_SPEED_MS)

        # Update Position using kinematic equation
        avg_velocity = (self.v + v_next) / 2.0
        ds = avg_velocity * config.DT
        
        # Ensure forward progress
        ds = max(0.0, ds)
        
        self.pos_in_seg += ds

        # Energy Calculation
        if f_trac > 0 and self.v > 0:
            p_mech_watts = f_trac * self.v
            p_elec_watts = p_mech_watts / self.phy.eta
            e_kwh_step = (p_elec_watts * config.DT) / 3.6e6
        else:
            e_kwh_step = 0.0
        
        self.energy_kwh += e_kwh_step

        # Update velocity
        self.v = v_next
        self.t += config.DT

        # Check Segment Transition
        while self.pos_in_seg >= config.DX:
            self.pos_in_seg -= config.DX
            self.seg_idx += 1
            
            # Check if reached end
            if self.seg_idx >= self.n_segments:
                self.done = True
                self.seg_idx = self.n_segments - 1
                self.pos_in_seg = 0.0
                break

        # Calculate NEW position (after step) in km
        self.position_km = (self.seg_idx * config.DX + self.pos_in_seg) / 1000.0

        # Calculate reward
        reward = 0.0

        # Progress reward - INCREASED to encourage forward movement
        progress_reward = (self.position_km - old_position_km) * 200  # Was 100, now 200

        # Energy penalty - REDUCED to not discourage movement
        energy_penalty = -e_kwh_step * 20.0  # Was 50, now 20

        # Time penalty - REDUCED
        time_penalty = -0.01  # Was 0.05, now 0.01

        reward = progress_reward + energy_penalty + time_penalty

        # Speed limit violation - keep but reduce penalty
        if self.v > limit:
            reward -= 20.0 + (self.v - limit) * 5  # Reduced from 50 and 10

        # Success bonus - MUCH LARGER
        if self.done:
            reward += 5000.0  # Was 1000, now 5000 - make completion very attractive
            # Bonus for time efficiency
            if self.t < config.MAX_STEPS_PER_EPISODE * 0.5:
                reward += 2000.0

        # Penalty for going too slow (helps detect stuck without hard termination)
        if self.v < 2.0 and not self.done:
            reward -= 5.0  # Penalty for being too slow
        
        info = {
            'segment': self.seg_idx,
            'velocity': self.v,
            'time': self.t,
            'energy': self.energy_kwh,
            'position_in_segment': self.pos_in_seg,
            'distance_moved': ds,
            'acceleration': acc,
            'speed_limit': limit,
            'violation': self.v > limit
        }

        return self._get_state(), reward, self.done, info