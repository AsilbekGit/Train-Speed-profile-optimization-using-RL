"""
Train Physics - COMPLETE REPLACEMENT
=====================================
Copy this ENTIRE file to: env_settings/physics.py

This fixes the traction force calculation so train can climb steep grades.
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
        # Typical values: 0.25-0.35 for dry rail
        self.adhesion_coeff = 0.30
        
        # Maximum adhesion-limited force (for hill climbing)
        self.f_adhesion_max = self.adhesion_coeff * self.mass * self.g
        # = 0.30 * 360000 * 9.81 = 1,059,480 N
        
        # Calibrate C0 based on 22kWh/km target at 80km/h
        self.c0 = self._calibrate_c0()
        self.c1 = config.C1
        self.c2 = config.C2
        
        print(f"Physics Calibrated. C0 calculated as: {self.c0:.2f} N")
        print(f"  Max adhesion force: {self.f_adhesion_max:,.0f} N (for hill climbing)")

    def _calibrate_c0(self):
        v_ref_kmh = 80.0
        v_ref_ms = v_ref_kmh / 3.6
        energy_target_per_km_kwh = 22.0
        
        # Power (kW) = Energy(kWh) / Time(h)
        p_elec_kw = energy_target_per_km_kwh * v_ref_kmh
        
        # Mechanical Power (Watts)
        p_mech_w = (p_elec_kw * 1000.0) * self.eta
        
        # Force required at constant speed = Power / v
        f_target = p_mech_w / v_ref_ms
        
        # F = C0 + C1v + C2v^2
        c0 = f_target - (config.C1 * v_ref_ms) - (config.C2 * (v_ref_ms**2))
        
        return max(0.0, c0)

    def get_total_resistance(self, v, grade_pct, curv_pct):
        """Calculate total resistance force"""
        # 1. Davis (Rolling + Air)
        r_davis = self.c0 + self.c1*v + self.c2*(v**2)
        
        # 2. Grade Resistance (mg * sin(theta) ~ mg * grade/100)
        r_grade = self.mass * self.g * (grade_pct / 100.0)
        
        # 3. Curve Resistance (Roeckl's Approx)
        r_curve = self.mass_tons * 6.0 * abs(curv_pct)
        
        return r_davis + r_grade + r_curve

    def get_max_traction_force(self, v):
        """
        Calculate maximum traction force
        
        THIS IS THE FIXED VERSION!
        - At low speed: Uses adhesion limit (~1,000,000 N) for hill climbing
        - At high speed: Uses power limit
        """
        # Power limit: F = P*η / v
        if v < 0.5:
            f_power = self.f_adhesion_max
        else:
            f_power = (self.power * self.eta) / v
        
        # Adhesion limit (wheel slip prevention)
        f_adhesion = self.f_adhesion_max  # ~1,059,480 N
        
        # Comfort limit (for passenger comfort at high speeds)
        f_comfort = self.mass * config.MAX_ACC  # 216,000 N
        
        # At low speeds (hill climbing), prioritize adhesion limit
        # At high speeds, also consider comfort
        if v < 10.0:
            # Low speed: need high force for hills
            return min(f_adhesion, f_power)
        else:
            # High speed: consider comfort too
            return min(f_comfort, f_power, f_adhesion)