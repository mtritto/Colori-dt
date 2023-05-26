import numpy as np
import cv2
import math
from PIL import Image

class ColorDifferenceMetrics:
    def __init__(self) -> None:
        pass

    def cie76_lab(self, reference_image, test_image):
        # Convert images to LAB color space
        reference_image = cv2.cvtColor(reference_image, cv2.COLOR_RGB2LAB)
        test_image = cv2.cvtColor(test_image, cv2.COLOR_RGB2LAB)

        # Calculate color difference in the LAB color space
        deltaE = np.sqrt(np.sum(np.square(reference_image.astype("uint8") - test_image.astype("uint8")),axis=2)).astype("uint8")

        return deltaE 

    def cie76_luv(self, reference_image, test_image):
        # Convert images to LUV color space
        reference_image = cv2.cvtColor(reference_image, cv2.COLOR_RGB2LUV)
        test_image = cv2.cvtColor(test_image, cv2.COLOR_RGB2LUV)

        # Calculate color difference in the LUV color space
        deltaE = np.sqrt(np.sum(np.square(reference_image.astype("uint8") - test_image.astype("uint8")),axis=2)).astype("uint8")

        return deltaE

    def cie94(self, reference_image, test_image, k_L, k_1, k_2):
        # Convert images to LAB color space    # Convert images to LAB color space
        reference_image = cv2.cvtColor(reference_image, cv2.COLOR_RGB2LAB)
        test_image = cv2.cvtColor(test_image, cv2.COLOR_RGB2LAB)

        # Split LAB channels
        l1, a1, b1 = cv2.split(reference_image)
        l2, a2, b2 = cv2.split(test_image)

        # Calculate chroma and hue differences
        c1 = np.sqrt(np.power(a1,2) + np.power(b1,2))
        c2 = np.sqrt(np.power(a2,2) + np.power(b2,2))
        delta_L = l1 - l2
        delta_C = c1 - c2
        delta_a = a1 - a2
        delta_b = b1 - b2
        delta_C_prime = np.sqrt(np.power(delta_a,2) + np.power(delta_b,2) - np.power(delta_C,2))

        # Calculate hue difference
        delta_h_prime = np.zeros_like(delta_L).astype("float32")
        mask = np.logical_and(delta_a != 0, delta_b != 0)
        delta_h_prime[mask] = np.arctan2(b1[mask] - b2[mask], a1[mask] - a2[mask])
        mask = delta_h_prime < 0
        delta_h_prime[mask] += 2*math.pi

        # Set k_C and k_H to fixed values
        k_C = 1
        k_H = 1

        # Calculate weighting factors
        s_l = 1
        S_C = 1 + k_1*c1
        s_h = 1 + k_2*c1

        # Calculate total color difference
        delta_E = np.sqrt(np.power((delta_L/(k_L*s_l)),2) + np.power((delta_C_prime/(k_C*S_C)),2) + np.power((delta_h_prime/(k_H*s_h)),2)).astype("uint8")

        return delta_E

    def ciede2000(self, reference_image, test_image, k_L, k_1, k_2):
        # Convert images to LAB color space
        reference_image = cv2.cvtColor(reference_image, cv2.COLOR_RGB2LAB)
        test_image = cv2.cvtColor(test_image, cv2.COLOR_RGB2LAB)
        
        #Split LAB channels
        l1, a1, b1 = cv2.split(reference_image)
        l2, a2, b2 = cv2.split(test_image)

        # Calculate chroma and hue differences
        c1 = np.sqrt(np.power(a1,2) + np.power(b1,2))
        c2 = np.sqrt(np.power(a2,2) + np.power(b2,2))
        c_avg = (c1 + c2) / 2
        g = 0.5 * (1 - np.sqrt(np.power(c_avg,7) / (np.power(c_avg,7) + 6103515625)))
        a1_p = (1 + g) * a1
        a2_p = (1 + g) * a2
        c1_p = np.sqrt(np.power(a1_p,2) + np.power(b1,2))
        c2_p = np.sqrt(np.power(a2_p,2) + np.power(b2,2))
        delta_C_p = c2_p - c1_p
        delta_h_p = np.zeros_like(delta_C_p)
        mask = np.logical_and(a1_p != 0, b1 != 0)
        delta_h_p[mask] = np.arctan2(b1[mask], a1_p[mask])
        mask = delta_h_p < 0
        delta_h_p[mask] += 2*math.pi
        delta_h_p[mask] -= math.pi
        delta_h_p[mask] *= -1
        delta_H_p = 2 * np.sqrt(c1_p * c2_p) * np.sin(delta_h_p / 2)

        # Calculate lightness difference
        delta_L = l2 - l1

        # Calculate chroma difference
        delta_C = c2 - c1

        # Calculate hue difference
        delta_h = np.zeros_like(delta_C)
        mask = np.logical_and(a1 != 0, b1 != 0)
        delta_h[mask] = np.arctan2(b2[mask] - b1[mask], a2[mask] - a1[mask])
        mask = delta_h < 0
        delta_h[mask] += 2*math.pi

        # Set k_C and k_H to fixed values
        k_C = 1
        k_H = 1

        # Calculate weighting factors
        L_avg = (l1 + l2) / 2
        s_l = 1 + (k_L * k_1 * np.power((L_avg - 50),2)) / np.sqrt(20 + np.power((L_avg - 50),2))
        s_C = 1 + k_C * k_2 * c_avg
        t = 1 - 0.17 * np.cos(delta_h - math.pi/6) + 0.24 * np.cos(2 * delta_h) + 0.32 * np.cos(3 * delta_h + math.pi/30) - 0.20 * np.cos(4 * delta_h - 63 * math.pi/180)
        s_H = 1 + k_H * k_2 * c_avg * t
        r_t = -2 * np.sqrt(np.power(c_avg,7) / (np.power(c_avg,7) + 6103515625)) * np.sin((60*(math.pi/180) * np.exp(-(np.power((delta_H_p -(275*math.pi/180)) / (25*math.pi/180),2)))))
      
        # Calculate total color difference
        delta_E = np.sqrt(np.power((delta_L / s_l),2) + np.power((delta_C / s_C),2) + np.power((delta_H_p / s_H),2) + r_t * (delta_C / s_C) * (delta_H_p / s_H)).astype("uint8")

        return delta_E

        
    def cmc(self, reference_image, test_image, ratio):
        # Convert images to LAB color space
        reference_image = cv2.cvtColor(reference_image, cv2.COLOR_RGB2LAB)
        test_image = cv2.cvtColor(test_image, cv2.COLOR_RGB2LAB)
        # Split LAB channels
        l1, a1, b1 = cv2.split(reference_image)
        l2, a2, b2 = cv2.split(test_image)

        # Calculate chroma and hue differences
        c1 = np.sqrt(np.power(a1,2) + np.power(b1,2))
        c2 = np.sqrt(np.power(a2,2) + np.power(b2,2))
        delta_L = l1 - l2
        delta_C = c1 - c2
        delta_a = a1 - a2
        delta_b = b1 - b2

        # Calculate hue difference
        delta_h = np.sqrt(np.power(delta_a,2) + np.power(delta_b,2) - np.power(delta_C,2))
        delta_h[delta_h < 0] = 0

        # Calculate weighting factors
        f = np.sqrt(np.power(c1,4) / (np.power(c1,4) + 1900))
        t = np.zeros_like(delta_L)
        mask = l1 < 16
        t[mask] = 0.511
        t[~mask] = (0.040975 * l1[~mask]) / (1 + 0.01765 * l1[~mask])
        s_L = 1 + (0.015 * np.power((l1 - 50),2)) / np.sqrt(20 + np.power((l1 - 50),2))
        s_C = 1 +  c1
        s_H = 1 + (ratio * f * delta_h) / (t * c1 + ratio * f * delta_h)

        # Calculate total color difference
        delta_E = np.sqrt(np.power((delta_L / s_L),2) + np.power((delta_C / s_C),2) + np.power((delta_h / s_H),2)).astype("uint8")

        return delta_E


    def icsm(self, reference_image, test_image, ref_angle):
        # Calculate DELUV using LUV76 formula
        deluv = self.cie76_luv(reference_image, test_image)
        ref_angle = ref_angle * math.pi / 180
        
        # Convert images to LUV color space
        reference_image = cv2.cvtColor(reference_image, cv2.COLOR_RGB2LUV)
        test_image = cv2.cvtColor(test_image, cv2.COLOR_RGB2LUV)

        # Calculate color distance vector V for each pixel
        V = test_image - reference_image

        # Calculate weight matrix
        rNorm = np.sqrt(np.sum(np.power(reference_image,2), axis=2))
        tNorm = np.sqrt(np.sum(np.power(test_image,2), axis=2))
        I = np.sum(reference_image * test_image, axis=2) / (rNorm * tNorm)
        I = np.maximum(I, -1)
        I = np.minimum(I, 1)
        theta = np.arccos(I) * 180 / math.pi
        omega = 1 + theta / ref_angle

        # Calculate dref matrix
        VNorm = V / np.sum(np.abs(test_image - reference_image), axis=2, keepdims=True)
        Ldiff = test_image[:,:,0] - reference_image[:,:,0]
        sgnMatrix = np.sign(Ldiff)
        VNorm *= sgnMatrix[:,:,np.newaxis]
        a = -77.8695
        b = 30.1200
        c = 38.8599
        aMat = np.ones_like(Ldiff) * a
        bMat = np.ones_like(Ldiff) * b
        cMat = np.ones_like(Ldiff) * c
        VMat = np.dstack((aMat, bMat, cMat))
        dref = np.sum(VNorm * VMat, axis=2) + 179.2889

        # Calculate color difference
        diff = (omega * deluv) / dref
        return diff.astype('uint8')

    def dergb(self, reference_image, test_image):
        # Calculate color difference in the RGB color space
        difference = np.sum(np.square(reference_image.astype('uint8') - test_image.astype('uint8')),axis=2).astype('uint8')
        print(np.min(difference), np.max(difference))
        return difference   




        



    