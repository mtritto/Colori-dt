from utils.neural_style_transfer import NeuralStyleTransfer
import cv2
import numpy as np
import pytest

np.random.rand(3)

@pytest.fixture
def style_transfer():
    return NeuralStyleTransfer()


def test_preprocess_image_neural(style_transfer):
    '''Test if the preprocess_image_neural method returns the expected output'''
    image = np.random.rand(256, 256, 3)
    processed_image = style_transfer.preprocess_image_neural(image)
    assert processed_image.shape == (1, 224, 224, 3)


def test_deprocess_image_neural(style_transfer):
    '''Test if the deprocess_image_neural method returns the expected output'''
    processed_image = np.random.rand(1, 224, 224, 3)
    deprocessed_image = style_transfer.deprocess_image_neural(processed_image,1)
    assert deprocessed_image.shape == (224, 224, 3)


def test_gram_matrix(style_transfer):
    '''Test if the gram_matrix method returns the expected output'''
    input_tensor = np.random.rand(1, 224, 224, 64)
    gram_matrix = style_transfer.gram_matrix(input_tensor)
    assert gram_matrix.shape == (1, 64, 64)


