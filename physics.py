import config

class TrainPhysics:
    def __init__(self):
        self.mass = config.MASS_KG
        self.mass_tons = self.mass / 1000.0
        self.power = config.POWER_WATTS
        self.eta = config.EFFICIENCY
        self.g = 9.81
        
        # Calibrate C0 based on 22kWh/km target at 80km/h
        self.c0 = self._calibrate_c0()
        self.c1 = config.C1
        self.c2 = config.C2

    def _calibrate_c0(self):
        v_ref_kmh = 80.0
        v_ref_ms = v_ref_kmh / 3.6
        energy_target_per_km_kwh = 22.0
        
        # Power (kW) = Energy(kWh) / Time(h)
        # Time to travel 1km at v_ref = 1/v_ref_kmh
        # P_elec_kW = E * v_ref_kmh
        p_elec_kw = energy_target_per_km_kwh * v_ref_kmh
        
        # Mechanical Power (Watts)
        p_mech_w = (p_elec_kw * 1000.0) * self.eta
        
        # Force required at constant speed = Power / v
        f_target = p_mech_w / v_ref_ms
        
        # F = C0 + C1v + C2v^2
        # C0 = F - C1v - C2v^2
        c0 = f_target - (config.C1 * v_ref_ms) - (config.C2 * (v_ref_ms**2))
        
        print(f"Physics Calibrated. C0 calculated as: {max(0, c0):.2f} N")
        return max(0.0, c0)

    def get_total_resistance(self, v, grade_pct, curv_pct):
        # 1. Davis (Rolling + Air)
        r_davis = self.c0 + self.c1*v + self.c2*(v**2)
        
        # 2. Grade Resistance (mg * sin(theta) ~ mg * grade/100)
        r_grade = self.mass * self.g * (grade_pct / 100.0)
        
        # 3. Curve Resistance (Roeckl's Approx: 600 N/ton / Radius)
        # Assuming input curv_pct is proxy for 1/Radius scaled
        # Using formula: R = Mass_tons * 6 * Curvature_Percent
        r_curve = self.mass_tons * 6.0 * abs(curv_pct)
        
        return r_davis + r_grade + r_curve

    def get_max_traction_force(self, v):
        # Force limited by engine power: F = P/v
        # Also limited by adhesion/friction (mass * max_acc)
        if v < 1.0:
            return self.mass * config.MAX_ACC
        
        f_power_limit = (self.power * self.eta) / v
        f_friction_limit = self.mass * config.MAX_ACC
        
        return min(f_friction_limit, f_power_limit)