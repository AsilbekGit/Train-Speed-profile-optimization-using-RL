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
        self.v = 0.0  # Start from rest
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
            # Apply just enough force to counter resistance
            if resistance > 0:
                f_trac = resistance
            else:
                # On downhill, resistance might be negative
                # Don't apply traction if gravity is accelerating us
                f_trac = 0
                
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

        # Update Position using kinematic equation: s = vt + 0.5at²
        # Use average velocity for more accurate position update
        avg_velocity = (self.v + v_next) / 2.0
        ds = avg_velocity * config.DT
        
        # Ensure forward progress (important for stuck detection)
        ds = max(0.0, ds)
        
        self.pos_in_seg += ds

        # Energy Calculation
        # Power is consumed only if Traction force > 0
        # P_mechanical = F_traction * velocity
        # P_electrical = P_mechanical / efficiency
        if f_trac > 0 and self.v > 0:
            p_mech_watts = f_trac * self.v
            p_elec_watts = p_mech_watts / self.phy.eta
            # Energy in kWh = (Power in Watts * Time in seconds) / (1000 * 3600)
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
                self.seg_idx = self.n_segments - 1  # Cap at last segment
                self.pos_in_seg = 0.0
                break

        # Calculate Reward (as per your design)
        reward = 0.0
        
        # Energy penalty (higher weight to encourage efficiency)
        reward -= e_kwh_step * 100.0
        
        # Time penalty (encourage faster travel)
        reward -= 0.1
        
        # Speed limit violation penalty
        if self.v > limit:
            violation = self.v - limit
            reward -= 50.0 + violation * 10.0
        
        # Success bonus
        if self.done:
            # Large bonus for completing the route
            reward += 1000.0
            
            # Additional bonus for efficiency
            # Average energy should be around 22 kWh per segment
            expected_energy = self.n_segments * 0.022  # 22 kWh/segment
            if self.energy_kwh < expected_energy:
                # Bonus for using less energy than expected
                reward += (expected_energy - self.energy_kwh) * 10.0

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