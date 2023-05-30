#import class from gui folder
from utils.color_difference_metrics import ColorDifferenceMetrics
import pytest
import cv2
import numpy as np


class TestColorDifferenceMetrics():
    def test_init(self):
        # Test that the class is initialized
        color_difference_metrics = ColorDifferenceMetrics()
        assert color_difference_metrics is not None

    # Tests for the cie76_lab method
    def test_cie76_lab_same_image(self):
        # Test that the cie76_lab method returns zero if the images are 
        # identical
        color_difference_metrics = ColorDifferenceMetrics()
        reference_image = cv2.imread("tests/test_images/1.jpeg")
        test_image = cv2.imread("tests/test_images/1.jpeg")
        deltaE = color_difference_metrics.cie76_lab(reference_image, test_image)
        # Assert that the deltaE is zero for all pixels
        assert np.all(deltaE == np.zeros_like(reference_image[:,:,0]))

    def test_cie76_luv_same_image(self):
        # Test that the cie76_luv method returns zero if the images are identical
        color_difference_metrics = ColorDifferenceMetrics()
        reference_image = cv2.imread("tests/test_images/1.jpeg")
        test_image = cv2.imread("tests/test_images/1.jpeg")
        deltaE = color_difference_metrics.cie76_luv(reference_image, test_image)
        # Assert that the deltaE is zero for all pixels
        assert np.all(deltaE == np.zeros_like(reference_image[:,:,0]))
    
    def test_cie94_same_image(self):
        # Test the cie94 method if the images are identical
        color_difference_metrics = ColorDifferenceMetrics()
        reference_image = cv2.imread("tests/test_images/1.jpeg")
        test_image = cv2.imread("tests/test_images/1.jpeg")
        k_L = 1
        k_1 = 0.045
        k_2 = 0.015
        deltaE = color_difference_metrics.cie94(reference_image, test_image, k_L, k_1, k_2)
        # Assert that the deltaE is zero for all pixels
        assert np.all(deltaE == np.zeros_like(reference_image[:,:,0]))

    def test_ciede2000_same_image(self):
        # Test the ciede2000 method if the images are identical
        color_difference_metrics = ColorDifferenceMetrics()
        reference_image = cv2.imread("tests/test_images/1.jpeg")
        test_image = cv2.imread("tests/test_images/1.jpeg")
        deltaE = color_difference_metrics.ciede2000(reference_image, test_image)
        # Assert that the deltaE is zero for all pixels
        assert np.all(deltaE == np.zeros_like(reference_image[:,:,0]))
    
    def test_rgb_same_image(self):
        # Test the rgb method if the images are identical
        color_difference_metrics = ColorDifferenceMetrics()
        reference_image = cv2.imread("tests/test_images/1.jpeg")
        test_image = cv2.imread("tests/test_images/1.jpeg")
        deltaE = color_difference_metrics.dergb(reference_image, test_image)
        # Assert that the deltaE is zero for all pixels
        assert np.all(deltaE == np.zeros_like(reference_image[:,:,0]))

    def test_cmc_same_image(self):
        # Test the cmc method if the images are identical
        color_difference_metrics = ColorDifferenceMetrics()
        reference_image = cv2.imread("tests/test_images/1.jpeg")
        test_image = cv2.imread("tests/test_images/1.jpeg")
        ratio = 2  
        deltaE = color_difference_metrics.cmc(reference_image, test_image, ratio)
        # Assert that the deltaE is zero for all pixels
        assert np.all(deltaE == np.zeros_like(reference_image[:,:,0]))

    def test_icsm_same_image(self):
        # Test the icsm method if the images are identical
        color_difference_metrics = ColorDifferenceMetrics()
        reference_image = cv2.imread("tests/test_images/1.jpeg")
        test_image = cv2.imread("tests/test_images/1.jpeg")
        ref_angle = 0
        deltaE = color_difference_metrics.icsm(reference_image, test_image, ref_angle)
        # Assert that the deltaE is zero for all pixels
        assert np.all(deltaE == np.zeros_like(reference_image[:,:,0]))



    


        

