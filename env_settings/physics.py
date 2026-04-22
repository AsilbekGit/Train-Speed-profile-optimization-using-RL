"""
Train Physics — Corrected Unit Handling
========================================
Davis equation: r_t(V) = C0 + C1*V + C2*V^2
  - V in km/h  (internally we convert from m/s)
  - r_t in kg*s/ton (specific resistance per ton)
  - Total force: F_davis = r_t * mass_tons * g   [Newtons]

Tractive effort (paper Eq. 1):
  F_t = P * eta / v                               [Newtons]
  Capped by adhesion limit at low speeds.

Grade resistance (paper Eq. 4):
  R_g = m * g * sin(θ) ≈ m * g * (grade%/100)     [Newtons]

Curve resistance (Roeckl approximation):
  R_c = mass_tons * k_c * |curvature|              [Newtons]
"""

import env_settings.config as config


class TrainPhysics:
    def __init__(self):
        self.mass = config.MASS_KG          # 360 000 kg
        self.mass_tons = config.MASS_TONS   # 360 tons
        self.power = config.POWER_WATTS     # 3 640 000 W
        self.eta = config.EFFICIENCY        # 0.85
        self.g = config.GRAVITY             # 9.81 m/s^2

        # Davis coefficients (Uzbekistan ER9E) — V in km/h, output in kg*s/ton
        self.c0 = config.C0    # 1.1
        self.c1 = config.C1    # 0.01
        self.c2 = config.C2    # 0.000227

        # Adhesion coefficient (wheel-rail friction, dry rail)
        self.adhesion_coeff = 0.30
        self.f_adhesion_max = self.adhesion_coeff * self.mass * self.g
        # = 0.30 * 360000 * 9.81 ≈ 1 059 480 N

        # Comfort limit for passengers at higher speeds
        self.f_comfort = self.mass * config.MAX_ACC  # 360000 * 0.6 = 216 000 N

        # Sanity check at 80 km/h
        r80 = self._davis_specific(80.0)
        f80 = r80 * self.mass_tons * self.g
        print(f"Physics initialized (Uzbekistan ER9E):")
        print(f"  Davis coeffs: C0={self.c0}, C1={self.c1}, C2={self.c2}")
        print(f"  Davis units: V in km/h, r_t in kg*s/ton")
        print(f"  Davis at 80 km/h (flat): {r80:.4f} kg*s/ton -> {f80:.0f} N total")
        print(f"  Max adhesion force: {self.f_adhesion_max:,.0f} N")

    # ----------------------------------------------------------------
    def _davis_specific(self, v_kmh):
        """Specific resistance in kg*s/ton for velocity V (km/h)."""
        return self.c0 + self.c1 * v_kmh + self.c2 * (v_kmh ** 2)

    # ----------------------------------------------------------------
    def get_total_resistance(self, v, grade_pct, curv_pct):
        """
        Total resistance acting on the train [N].

        Parameters
        ----------
        v         : float — speed in m/s  (converted to km/h for Davis)
        grade_pct : float — gradient in percent (positive = uphill)
        curv_pct  : float — curvature in percent
        """
        # Convert m/s -> km/h for Davis equation
        v_kmh = v * 3.6

        # 1. Davis (rolling + aerodynamic)
        r_specific = self._davis_specific(v_kmh)             # kg*s/ton
        r_davis = r_specific * self.mass_tons * self.g       # N

        # 2. Grade resistance: F = m*g*sin(θ) ≈ m*g*(grade/100)
        r_grade = self.mass * self.g * (grade_pct / 100.0)   # N

        # 3. Curve resistance (Roeckl approximation)
        r_curve = self.mass_tons * 6.0 * abs(curv_pct)       # N

        return r_davis + r_grade + r_curve

    # ----------------------------------------------------------------
    def get_max_traction_force(self, v):
        """
        Maximum traction force at speed v (m/s) [N].

        Three limits:
        1. Power limit:    F = P·η / v
        2. Adhesion limit: F = μ·m·g
        3. Comfort limit:  F = m·a_max
        """
        # Power limit
        if v < 0.5:
            f_power = self.f_adhesion_max
        else:
            f_power = (self.power * self.eta) / v

        if v < 10.0:
            # Low speed — adhesion-limited for hill climbing
            return min(self.f_adhesion_max, f_power)
        else:
            # Higher speed — also respect passenger comfort
            return min(self.f_comfort, f_power, self.f_adhesion_max)