#import class from gui folder
from utils.color_difference_metrics import cie76_lab, cie94, cie76_luv, dergb, icsm, cmc, ciede2000

import pytest
import cv2
import numpy as np

np.random.seed(3)

class TestColorDifferenceMetrics():
    def test_cie76_lab_same_image(self):
        """
        Test that the cie76_lab method returns zero if the images are 
        identical
        """
        reference_image = cv2.imread("tests/test_images/1.jpeg")
        test_image = cv2.imread("tests/test_images/1.jpeg")
        deltaE = cie76_lab(reference_image, test_image)
        # Assert that the deltaE is zero for all pixels
        assert np.all(deltaE == np.zeros_like(reference_image[:,:,0]))

    def test_cie76_luv_same_image(self):
        """
        Test that the cie76_luv method returns zero if the images are identical
        """
        reference_image = cv2.imread("tests/test_images/1.jpeg")
        test_image = cv2.imread("tests/test_images/1.jpeg")
        deltaE = cie76_luv(reference_image, test_image)
        # Assert that the deltaE is zero for all pixels
        assert np.all(deltaE == np.zeros_like(reference_image[:,:,0]))

    def test_cie94_same_image(self):
        """
        Test the cie94 method if the images are identical
        """
        reference_image = cv2.imread("tests/test_images/1.jpeg")
        test_image = cv2.imread("tests/test_images/1.jpeg")
        k_L = 1
        k_1 = 0.045
        k_2 = 0.015
        deltaE = cie94(reference_image, test_image, k_L, k_1, k_2)
        # Assert that the deltaE is zero for all pixels
        assert np.all(deltaE == np.zeros_like(reference_image[:,:,0]))

    def test_ciede2000_same_image(self):
        """
        Test the ciede2000 method if the images are identical
        """
        reference_image = cv2.imread("tests/test_images/1.jpeg")
        test_image = cv2.imread("tests/test_images/1.jpeg")
        deltaE = ciede2000(reference_image, test_image)
        # Assert that the deltaE is zero for all pixels
        assert np.allclose(deltaE, np.zeros_like(reference_image[:,:,0]),atol=1)

    def test_rgb_same_image(self):
        """
        Test the rgb method if the images are identical
        """
        reference_image = cv2.imread("tests/test_images/1.jpeg")
        test_image = cv2.imread("tests/test_images/1.jpeg")
        deltaE = dergb(reference_image, test_image)
        # Assert that the deltaE is zero for all pixels
        assert np.all(deltaE == np.zeros_like(reference_image[:,:,0]))

    def test_cmc_same_image(self):
        """
        Test the cmc method if the images are identical
        """
        reference_image = cv2.imread("tests/test_images/1.jpeg")
        test_image = cv2.imread("tests/test_images/1.jpeg")
        ratio = 2  
        deltaE = cmc(reference_image, test_image, ratio)
        # Assert that the deltaE is zero for all pixels
        assert np.all(deltaE == np.zeros_like(reference_image[:,:,0]))

    def test_icsm_same_image(self):
        """
        Test the icsm method if the images are identical
        """
        reference_image = cv2.imread("tests/test_images/1.jpeg")
        test_image = cv2.imread("tests/test_images/1.jpeg")
        # Set the ref angle
        ref_angle = 0
        deltaE = icsm(reference_image, test_image, ref_angle)
        # Assert that the deltaE is zero for all pixels
        assert np.all(deltaE == np.zeros_like(reference_image[:,:,0]))

    def test_cie76_luv_known_color_difference(self):
        """
        Test the metric against a fixed color difference value
        """
        # Generate two black images of set size 4
        size = 4
        reference_image = np.zeros((size, size, 3), dtype=np.uint8)
        test_image = np.zeros((size, size, 3), dtype=np.uint8)

        # Generate a known delta from rgb arrays
        known_luv_difference = cie76_luv(np.full((1, 1, 3), 5, dtype=np.uint8),np.full((1, 1, 3), 0, dtype=np.uint8))
        # The increment for the arrays needs convertion to luv space
        luv_increment = cv2.cvtColor(np.full((1, 1, 3), 5, dtype=np.uint8), cv2.COLOR_RGB2LUV)

        # Randomly select a position
        random_row, random_col = np.random.randint(0, size), np.random.randint(0, size)

        # Convert arrays to luv and add delta in test array at rand position
        ref_luv= cv2.cvtColor(reference_image, cv2.COLOR_RGB2LUV)
        test_luv =cv2.cvtColor(test_image, cv2.COLOR_RGB2LUV)
        test_luv[random_row, random_col] = luv_increment

        # Convert back to rgb
        ref_rgb= cv2.cvtColor(ref_luv, cv2.COLOR_LUV2RGB)
        test_rgb =cv2.cvtColor(test_luv, cv2.COLOR_LUV2RGB)


        assert cie76_luv(ref_rgb,test_rgb)[random_row][random_col] == known_luv_difference

    def test_cie76_lab_known_color_difference(self):
        # Generate two black images of set size 4
        size = 4
        reference_image = np.zeros((size, size, 3), dtype=np.uint8)
        test_image = np.zeros((size, size, 3), dtype=np.uint8)

        # Generate a known delta from rgb
        known_lab_difference = cie76_lab(np.full((1, 1, 3), 5, dtype=np.uint8),np.full((1, 1, 3), 0, dtype=np.uint8))
        lab_increment = cv2.cvtColor(np.full((1, 1, 3), 5, dtype=np.uint8), cv2.COLOR_RGB2LAB)

        # Randomly select a position
        random_row, random_col = np.random.randint(0, size), np.random.randint(0, size)

        # Convert arrays to luv and add delta in test array at rand position
        ref_lab= cv2.cvtColor(reference_image, cv2.COLOR_RGB2LAB)
        test_lab =cv2.cvtColor(test_image, cv2.COLOR_RGB2LAB)
        test_lab[random_row, random_col] = lab_increment

        # Convert back to rgb
        ref_rgb= cv2.cvtColor(ref_lab, cv2.COLOR_LAB2RGB)
        test_rgb =cv2.cvtColor(test_lab, cv2.COLOR_LAB2RGB)


        assert cie76_lab(ref_rgb,test_rgb)[random_row][random_col] == known_lab_difference

    def test_cie94_known_color_difference(self):
        """
        Test the metric against a fixed color difference value
        """
        # Generate two black images of set size 4
        size = 4
        reference_image = np.zeros((size, size, 3), dtype=np.uint8)
        test_image = np.zeros((size, size, 3), dtype=np.uint8)

        # Generate a known delta from rgb
        known_lab_difference = cie94(np.full((1, 1, 3), 5, dtype=np.uint8),np.full((1, 1, 3), 0, dtype=np.uint8))
        lab_increment = cv2.cvtColor(np.full((1, 1, 3), 5, dtype=np.uint8), cv2.COLOR_RGB2LAB)

        # Randomly select a position
        random_row, random_col = np.random.randint(0, size), np.random.randint(0, size)

        # Convert arrays to luv and add delta in test array at rand position
        ref_lab= cv2.cvtColor(reference_image, cv2.COLOR_RGB2LAB)
        test_lab =cv2.cvtColor(test_image, cv2.COLOR_RGB2LAB)
        test_lab[random_row, random_col] = lab_increment

        # Convert back to rgb
        ref_rgb= cv2.cvtColor(ref_lab, cv2.COLOR_LAB2RGB)
        test_rgb =cv2.cvtColor(test_lab, cv2.COLOR_LAB2RGB)


        assert cie94(ref_rgb,test_rgb)[random_row][random_col] == known_lab_difference    

    def test_cie2000_known_color_difference(self):
        """
        Test the metric against a fixed color difference value
        """
        # Generate two black images of set size 4
        size = 4
        reference_image = np.zeros((size, size, 3), dtype=np.uint8)
        test_image = np.zeros((size, size, 3), dtype=np.uint8)

        # Generate a known delta from rgb
        known_lab_difference = ciede2000(np.full((1, 1, 3), 5, dtype=np.uint8),np.full((1, 1, 3), 0, dtype=np.uint8))
        lab_increment = cv2.cvtColor(np.full((1, 1, 3), 5, dtype=np.uint8), cv2.COLOR_RGB2LAB)

        # Randomly select a position
        random_row, random_col = np.random.randint(0, size), np.random.randint(0, size)

        # Convert arrays to luv and add delta in test array at rand position
        ref_lab= cv2.cvtColor(reference_image, cv2.COLOR_RGB2LAB)
        test_lab =cv2.cvtColor(test_image, cv2.COLOR_RGB2LAB)
        test_lab[random_row, random_col] = lab_increment

        # Convert back to rgb
        ref_rgb= cv2.cvtColor(ref_lab, cv2.COLOR_LAB2RGB)
        test_rgb =cv2.cvtColor(test_lab, cv2.COLOR_LAB2RGB)


        assert ciede2000(ref_rgb,test_rgb)[random_row][random_col] == known_lab_difference   

    def test_cmc_known_color_difference(self):
        """
        Test the metric against a fixed color difference value
        """
        # Generate two black images of set size 4
        size = 4
        reference_image = np.zeros((size, size, 3), dtype=np.uint8)
        test_image = np.zeros((size, size, 3), dtype=np.uint8)

        # Generate a known delta from rgb
        known_lab_difference = cmc(np.full((1, 1, 3), 5, dtype=np.uint8),np.full((1, 1, 3), 0, dtype=np.uint8))
        lab_increment = cv2.cvtColor(np.full((1, 1, 3), 5, dtype=np.uint8), cv2.COLOR_RGB2LAB)

        # Randomly select a position
        random_row, random_col = np.random.randint(0, size), np.random.randint(0, size)

        # Convert arrays to luv and add delta in test array at rand position
        ref_lab= cv2.cvtColor(reference_image, cv2.COLOR_RGB2LAB)
        test_lab =cv2.cvtColor(test_image, cv2.COLOR_RGB2LAB)
        test_lab[random_row, random_col] = lab_increment

        # Convert back to rgb
        ref_rgb= cv2.cvtColor(ref_lab, cv2.COLOR_LAB2RGB)
        test_rgb =cv2.cvtColor(test_lab, cv2.COLOR_LAB2RGB)


        assert cmc(ref_rgb,test_rgb)[random_row][random_col] == known_lab_difference   

    def test_rgb_known_color_difference(self):
        """
        Test the metric against a fixed color difference value
        """
        # Generate two black images of set size 4
        size = 4
        ref_rgb= np.zeros((size, size, 3), dtype=np.uint8)
        test_rgb = np.zeros((size, size, 3), dtype=np.uint8)

        # Generate a known delta from rgb
        known_lab_difference = dergb(np.full((1, 1, 3), 5, dtype=np.uint8),np.full((1, 1, 3), 0, dtype=np.uint8))

        # Randomly select a position
        random_row, random_col = np.random.randint(0, size), np.random.randint(0, size)

        # Add delta in test array at rand position
        test_rgb[random_row, random_col] = np.full((1, 1, 3), 5, dtype=np.uint8)

        assert dergb(ref_rgb,test_rgb)[random_row][random_col] == known_lab_difference   

    



            


                

