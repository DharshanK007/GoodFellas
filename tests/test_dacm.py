"""
Unit Tests for Dynamic Advective Coupling (DACM) Geometry & Propagation.
"""

import unittest
import numpy as np

from skyguard.dacm.geometry import haversine_distance_km, station_direction_vector
from skyguard.dacm.wind_vector import (
    meteorological_wind_to_vector,
    wind_vector_to_speed_direction,
    circular_wind_stability,
)
from skyguard.dacm.alignment import compute_wind_alignment
from skyguard.dacm.propagation import AdvectivePropagationEstimator


class TestDACM(unittest.TestCase):

    def test_haversine_distance(self):
        # Distance between Bangalore (12.9716, 77.5946) and Chennai (13.0827, 80.2707) is ~290 km
        dist = haversine_distance_km(12.9716, 77.5946, 13.0827, 80.2707)
        self.assertTrue(280.0 < dist < 305.0)

    def test_station_direction_vector(self):
        # Station A -> Station B (Due East along equator / latitude)
        r_vec, bearing = station_direction_vector(12.0, 77.0, 12.0, 78.0)
        self.assertAlmostEqual(bearing, 90.0, delta=1.0)
        self.assertAlmostEqual(r_vec[0], 1.0, delta=0.05)  # East component
        self.assertAlmostEqual(r_vec[1], 0.0, delta=0.05)  # North component

    def test_wind_vector_meteorological_conversions(self):
        # Wind blowing from West (270°) transports air Eastward (+u)
        u, v = meteorological_wind_to_vector(10.0, 270.0)
        self.assertAlmostEqual(u, 10.0, places=3)
        self.assertAlmostEqual(v, 0.0, places=3)

        speed, deg = wind_vector_to_speed_direction(u, v)
        self.assertAlmostEqual(speed, 10.0, places=2)
        self.assertAlmostEqual(deg, 270.0, places=1)

    def test_wind_alignment_and_calm_conditions(self):
        # r_AB points East [1, 0]
        r_ab = np.array([1.0, 0.0], dtype=np.float32)

        # 1. Aligned wind: from West (270°) -> transports East [10, 0]
        v_aligned = np.array([10.0, 0.0], dtype=np.float32)
        align, is_calm = compute_wind_alignment(v_aligned, r_ab, wind_speed_a=10.0)
        self.assertAlmostEqual(align, 1.0, places=2)
        self.assertFalse(is_calm)

        # 2. Opposing wind: from East (90°) -> transports West [-10, 0]
        v_opposed = np.array([-10.0, 0.0], dtype=np.float32)
        align_opp, _ = compute_wind_alignment(v_opposed, r_ab, wind_speed_a=10.0)
        self.assertEqual(align_opp, 0.0)

        # 3. Calm wind attenuation (< 0.5 m/s)
        v_calm = np.array([0.2, 0.0], dtype=np.float32)
        align_calm, is_calm_flag = compute_wind_alignment(v_calm, r_ab, wind_speed_a=0.2)
        self.assertTrue(is_calm_flag)
        self.assertTrue(align_calm < 0.5)

    def test_circular_wind_stability(self):
        # Constant directions -> stability ≈ 1.0
        steady_dirs = [270.0, 271.0, 269.0, 270.5]
        s_w = circular_wind_stability(steady_dirs)
        self.assertTrue(s_w > 0.98)

        # Shifting/turbulent directions -> low stability
        turbulent_dirs = [0.0, 90.0, 180.0, 270.0]
        s_w_turb = circular_wind_stability(turbulent_dirs)
        self.assertTrue(s_w_turb < 0.2)

    def test_propagation_time_estimator(self):
        estimator = AdvectivePropagationEstimator()
        # Distance = 36 km, wind speed = 10 m/s (36 km/h), alignment = 1.0 -> travel time = 1.0 hour = 60 min
        prop_win = estimator.estimate_travel_time(
            distance_km=36.0, wind_speed_ms=10.0, directional_alignment=1.0, wind_stability=0.95
        )
        self.assertTrue(prop_win.is_valid)
        self.assertAlmostEqual(prop_win.tau_minutes, 60.0, delta=1.0)
        self.assertTrue(prop_win.min_arrival_minutes < 60.0 < prop_win.max_arrival_minutes)


if __name__ == "__main__":
    unittest.main()
