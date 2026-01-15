import numpy as np
import config

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
        print(f"  Actions: {self.action_space}")
        
        self.reset()

    def reset(self):
        """Reset environment to initial state"""
        self.seg_idx = 0
        self.pos_in_seg = 0.0
        self.v = 25.0  # 90 km/h
        self.t = 0.0
        self.energy_kwh = 0.0
        self.done = False
        
        # Track milestones for rewards
        self.milestones_reached = set()
        self.max_segment_reached = 0
        
        return self._get_state()

    def _get_state(self):
        return np.array([self.seg_idx, self.v], dtype=np.float32)

    def step(self, action):
        if self.done:
            return self._get_state(), 0, True, {}

        old_position_km = (self.seg_idx * config.DX + self.pos_in_seg) / 1000.0
        old_segment = self.seg_idx

        # Get current conditions
        grade = self.grades[self.seg_idx]
        curve = self.curves[self.seg_idx]
        limit = self.limits[self.seg_idx]

        # Forces
        resistance = self.phy.get_total_resistance(self.v, grade, curve)
        f_trac = 0.0
        f_brake = 0.0

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
        
        v_next = np.clip(self.v + acc * config.DT, 0.0, config.MAX_SPEED_MS)
        avg_velocity = (self.v + v_next) / 2.0
        ds = max(0.0, avg_velocity * config.DT)
        
        self.pos_in_seg += ds

        # Energy
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

        # Track progress
        if self.seg_idx > self.max_segment_reached:
            self.max_segment_reached = self.seg_idx

        self.position_km = (self.seg_idx * config.DX + self.pos_in_seg) / 1000.0

        # ============================================================
        # REWARD FUNCTION - Shaped for Long Route Learning
        # ============================================================
        
        reward = 0.0
        
        # 1. STRONG PROGRESS REWARD (most important for learning)
        distance_moved_km = self.position_km - old_position_km
        progress_reward = distance_moved_km * 500  # VERY HIGH (was 200)
        reward += progress_reward
        
        # 2. MILESTONE BONUSES - Critical for long routes!
        # Every 10km, give a bonus
        current_milestone = int(self.position_km // 10)
        old_milestone = int(old_position_km // 10)
        
        if current_milestone > old_milestone and current_milestone not in self.milestones_reached:
            milestone_bonus = 500  # Big bonus for each 10km
            reward += milestone_bonus
            self.milestones_reached.add(current_milestone)
        
        # 3. FORWARD PROGRESS BONUS (moved to new segment)
        if self.seg_idx > old_segment:
            reward += 5.0  # Small bonus per segment
        
        # 4. DISTANCE-TO-GO SHAPING (encourage getting closer to goal)
        distance_to_goal = (self.n_segments - self.seg_idx) * config.DX / 1000.0
        # Negative reward proportional to distance remaining
        reward -= distance_to_goal * 0.1
        
        # 5. Energy penalty (VERY SMALL - don't discourage movement!)
        reward -= e_kwh_step * 5.0  # Was 20, now 5
        
        # 6. Time penalty (VERY SMALL)
        reward -= 0.001  # Was 0.01, now 0.001
        
        # 7. Speed limit violation (moderate penalty)
        if self.v > limit:
            reward -= 10.0 + (self.v - limit) * 2
        
        # 8. Penalty for being too slow (but don't kill exploration)
        if self.v < 5.0 and not self.done:
            reward -= 2.0
        
        # 9. HUGE SUCCESS BONUS (make completion DOMINANT)
        if self.done and self.seg_idx >= self.n_segments - 1:
            reward += 10000.0  # Was 5000, now 10000!
            
            # Additional bonus for efficiency
            if self.t < 3000:  # Under 50 minutes
                reward += 3000.0
            if self.energy_kwh < 3000:  # Under 3000 kWh
                reward += 2000.0
        
        # 10. NEW: "Best so far" bonus
        # If reached farther than ever before, give bonus
        if self.seg_idx == self.max_segment_reached and self.seg_idx > old_segment:
            reward += 10.0  # Exploration bonus
        
        info = {
            'segment': self.seg_idx,
            'velocity': self.v,
            'time': self.t,
            'energy': self.energy_kwh,
            'position_in_segment': self.pos_in_seg,
            'distance_moved': ds,
            'acceleration': acc,
            'speed_limit': limit,
            'violation': self.v > limit,
            'distance_to_goal_km': distance_to_goal,
            'milestones': len(self.milestones_reached),
            'progress_percent': (self.seg_idx / self.n_segments) * 100
        }

        return self._get_state(), reward, self.done, info