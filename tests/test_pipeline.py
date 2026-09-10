"""
Integration Tests for the Complete SkyGuard Real-Time Pipeline.
"""

import unittest
from skyguard.data.anomaly_injection import AnomalyInjector
from skyguard.realtime.pipeline import SkyGuardPipeline, StreamProcessResult
from skyguard.anomaly.fusion import AnomalyClassification


class TestPipelineIntegration(unittest.TestCase):

    def test_full_pipeline_stream(self):
        # Generate benchmark stream
        dataset, meta = AnomalyInjector.create_synthetic_network_benchmark(
            num_stations=5, num_timesteps=40
        )
        pipeline = SkyGuardPipeline()

        # Fit on baseline
        train_obs = dataset.observations[:60]
        loss_history = pipeline.fit_and_train_models(train_obs, epochs=15)
        self.assertTrue(len(loss_history["loss_ch1"]) > 0)
        self.assertTrue(len(loss_history["loss_ch2"]) > 0)

        # Process sequential stream
        unique_ts = sorted(list(set(o.timestamp for o in dataset.observations)))
        processed_count = 0

        for ts in unique_ts[:20]:
            slice_obs = dataset.get_time_slice(ts)
            for obs in slice_obs:
                result = pipeline.process_observation(obs, current_network_snapshot=slice_obs)
                self.assertIsInstance(result, StreamProcessResult)
                self.assertIsNotNone(result.decision)
                self.assertIsNotNone(result.health_report)
                self.assertIsNotNone(result.explanation_report)
                self.assertIsNotNone(result.corrected_observation)
                processed_count += 1

        self.assertEqual(processed_count, 20 * 5)


if __name__ == "__main__":
    unittest.main()
