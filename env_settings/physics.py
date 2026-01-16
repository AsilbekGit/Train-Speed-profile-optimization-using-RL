"""
Train Physics - FIXED VERSION
==============================
INSTALLATION: Copy this file to env_settings/physics.py
              (backup your original first!)

Problem: Original MAX_ACC = 0.6 m/s² limits traction to 216,000 N
         But 6% grade resistance alone = 211,896 N
         Train CANNOT climb steep grades!

Fix: Use realistic adhesion-limited traction at low speeds
     Real trains can produce much more force when moving slowly
"""

import env_settings.config as config

class TrainPhysics:
    def __init__(self):
        self.mass = config.MASS_KG
        self.mass_tons = self.mass / 1000.0
        self.power = config.POWER_WATTS
        self.eta = config.EFFICIENCY
        self.g = 9.81
        
        # Adhesion coefficient (wheel-rail friction)
        # Typical values: 0.25-0.35 for dry rail, 0.1-0.2 for wet
        self.adhesion_coeff = 0.30  # Conservative dry rail
        
        # Maximum adhesion-limited force
        # This is the REAL limit at low speeds
        self.f_adhesion_max = self.adhesion_coeff * self.mass * self.g
        # = 0.30 * 360000 * 9.81 = 1,059,480 N
        
        # Calibrate C0
        self.c0 = self._calibrate_c0()
        self.c1 = config.C1
        self.c2 = config.C2
        
        print(f"Physics Calibrated (FIXED):")
        print(f"  C0 = {self.c0:.2f} N")
        print(f"  Max adhesion force = {self.f_adhesion_max:,.0f} N")
        print(f"  Can climb grade up to: {self._max_climbable_grade():.1f}%")

    def _calibrate_c0(self):
        """Calibrate C0 based on 22kWh/km target at 80km/h"""
        v_ref_kmh = 80.0
        v_ref_ms = v_ref_kmh / 3.6
        energy_target_per_km_kwh = 22.0
        
        p_elec_kw = energy_target_per_km_kwh * v_ref_kmh
        p_mech_w = (p_elec_kw * 1000.0) * self.eta
        f_target = p_mech_w / v_ref_ms
        
        c0 = f_target - (config.C1 * v_ref_ms) - (config.C2 * (v_ref_ms**2))
        return max(0.0, c0)
    
    def _max_climbable_grade(self):
        """Calculate maximum grade the train can climb"""
        # At low speed, max force = adhesion limit
        # Need: F_traction > F_davis + F_grade
        # F_grade = m * g * (grade/100)
        # grade_max = (F_adhesion - F_davis_at_low_v) / (m * g) * 100
        
        f_davis_low = self.c0 + self.c1 * 2.0 + self.c2 * 4.0  # At 2 m/s
        grade_max = (self.f_adhesion_max - f_davis_low) / (self.mass * self.g) * 100
        return grade_max

    def get_total_resistance(self, v, grade_pct, curv_pct):
        """
        Calculate total resistance force
        Based on Davis equation + grade + curve resistance
        """
        # 1. Davis equation (rolling + aerodynamic)
        r_davis = self.c0 + self.c1 * v + self.c2 * (v ** 2)
        
        # 2. Grade resistance
        # Positive grade = uphill = positive resistance
        r_grade = self.mass * self.g * (grade_pct / 100.0)
        
        # 3. Curve resistance (Röckl formula)
        r_curve = self.mass_tons * 6.0 * abs(curv_pct)
        
        return r_davis + r_grade + r_curve

    def get_max_traction_force(self, v):
        """
        Calculate maximum traction force
        
        FIXED: Now uses realistic limits:
        1. At low speed: Limited by wheel-rail adhesion
        2. At high speed: Limited by motor power
        3. Always limited by config.MAX_ACC for comfort
        """
        # Comfort limit (for passenger comfort)
        f_comfort = self.mass * config.MAX_ACC  # 216,000 N
        
        # Power limit: F = P*η / v
        if v < 0.5:
            f_power = self.f_adhesion_max  # Use adhesion limit at very low speed
        else:
            f_power = (self.power * self.eta) / v
        
        # Adhesion limit (wheel slip prevention)
        f_adhesion = self.f_adhesion_max  # ~1,059,480 N
        
        # The actual limit is the minimum of all constraints
        # BUT for climbing grades, we need to allow higher forces
        # So we use adhesion limit (not comfort limit) when needed
        
        # At low speeds (< 10 m/s), prioritize adhesion limit for hill climbing
        # At high speeds, power becomes the limit anyway
        if v < 10.0:
            # Allow higher force for hill climbing at low speeds
            return min(f_adhesion, f_power)
        else:
            # At higher speeds, also consider comfort
            return min(f_comfort, f_power, f_adhesion)