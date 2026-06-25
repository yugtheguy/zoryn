import unittest
import os
import shutil
import cv2
import numpy as np
from generate_sample_data import create_synthetic_road_mask
from pipeline import run_master_pipeline

class TestEndToEndPipeline(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_artifacts_tmp/"
        os.makedirs(self.test_dir, exist_ok=True)
        self.mask_path = os.path.join(self.test_dir, "test_mask.png")
        mask = create_synthetic_road_mask(width=256, height=256, road_thickness=8)
        cv2.imwrite(self.mask_path, mask)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_pipeline_execution(self):
        out_dir = os.path.join(self.test_dir, "exports/")
        exports = run_master_pipeline(
            input_mask_path=self.mask_path,
            output_dir=out_dir,
            max_gap_dist=50.0,
            simulate_k_values=[1, 2]
        )
        self.assertTrue(os.path.exists(exports["graph_data"]))
        self.assertTrue(os.path.exists(exports["critical_nodes"]))
        self.assertTrue(os.path.exists(exports["geojson_export"]))

if __name__ == '__main__':
    unittest.main()
