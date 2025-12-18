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
        
        self.reset()

    def reset(self):
        self.seg_idx = 0
        self.pos_in_seg = 0.0
        self.v = 0.0
        self.t = 0.0
        self.energy_kwh = 0.0
        self.done = False
        return self._get_state()

    def _get_state(self):
        return np.array([self.seg_idx, self.v], dtype=np.float32)

    def step(self, action):
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

        if action == 0: # Full Brake
            f_brake = abs(self.phy.mass * config.MAX_DEC)
        
        elif action == 1: # Coast
            pass # Forces 0
            
        elif action == 2: # Cruise
            # Try to match resistance
            if resistance > 0:
                f_trac = resistance
            else:
                f_trac = 0
                
        elif action == 3: # Max Power
            f_trac = self.phy.get_max_traction_force(self.v)

        # Net Force
        f_net = f_trac - f_brake - resistance
        
        # Calculate Accel (F = ma)
        acc = f_net / self.phy.mass
        acc = np.clip(acc, config.MAX_DEC, config.MAX_ACC)

        # Update Velocity (Euler)
        v_next = self.v + acc * config.DT
        v_next = max(0.0, v_next)

        # Update Position
        ds = self.v * config.DT + 0.5 * acc * (config.DT**2)
        self.pos_in_seg += ds

        # Energy Calc
        # Power is consumed only if Traction > 0
        p_inst = (f_trac * self.v) / self.phy.eta if f_trac > 0 else 0.0
        e_kwh_step = (p_inst * config.DT) / 3.6e6
        self.energy_kwh += e_kwh_step

        self.v = v_next
        self.t += config.DT

        # Check Segment Transition
        while self.pos_in_seg >= config.DX:
            self.pos_in_seg -= config.DX
            self.seg_idx += 1
            if self.seg_idx >= self.n_segments:
                self.done = True
                self.seg_idx = self.n_segments - 1
                break

        # Rewards
        reward = 0.0
        reward -= e_kwh_step * 100.0       # Energy Penalty
        reward -= 0.1                      # Time Penalty
        
        if self.v > limit:                 # Speeding Penalty
            reward -= 50.0 + (self.v - limit) * 10

        if self.done:
            reward += 1000.0               # Success Bonus

        return self._get_state(), reward, self.done, {}