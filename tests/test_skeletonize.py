import unittest
import numpy as np
from m2_graph import skeletonize_road_mask

class TestSkeletonize(unittest.TestCase):
    def test_skeletonize_simple_rect(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[20:30, 10:40] = 255 # Thick horizontal blob
        
        skel = skeletonize_road_mask(mask, min_component_size=10)
        self.assertEqual(skel.shape, (50, 50))
        # After thinning, road thickness should be 1 pixel (max column sum along road ~ 1)
        nonzero_counts = np.sum(skel > 0, axis=0)
        self.assertTrue(np.all(nonzero_counts[15:35] <= 2))

    def test_noise_removal(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10, 10] = 255 # Isolated 1px noise
        skel = skeletonize_road_mask(mask, min_component_size=5)
        self.assertEqual(np.sum(skel > 0), 0)

if __name__ == '__main__':
    unittest.main()
