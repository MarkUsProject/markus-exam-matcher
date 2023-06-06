"""
MarkUs Exam Matcher: Reading Characters

Information
===============================
This module defines the image processing functions to extract
characters from the boxes they are contained in, and to interpret
these characters as strings.
"""


import cv2
import numpy as np
from typing import List
import tempfile

from src.markus_exam_matcher.core.char_types import CharType

# TODO: Ask about convention for importing this
from src.markus_exam_matcher._cnn.cnn import get_num


# ==================================================================
# Functions
# ==================================================================
def read_img(img_path: str) -> np.ndarray:
    """
    Read an input image.

    :param img_path: Path to image.
    :return: np.ndarray representing the image specified by
             img_path.
    """
    return cv2.imread(img_path)


def interpret_char_images(imgs: List[np.ndarray], char_type: CharType) -> str:
    """
    Return a string representing the characters written in imgs.

    :param imgs: List of images represented as np.ndarray's, where each
                 image represents a character.
    :param char_type: The type of characters that imgs represent.
    :return: String representing the characters written in imgs. Order
             is preserved.
    """
    # Write characters to a temporary directory and run CNN on images
    # in this directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        with tempfile.TemporaryDirectory(dir=tmp_dir) as img_dir:
            _write_images_to_dir(imgs, img_dir)

            if char_type == CharType.DIGIT:
                return get_num(tmp_dir, img_dir)
            else:
                # TODO: Implement reading letters
                assert False


def _write_images_to_dir(imgs: List[np.ndarray], dir: str) -> None:
    """
    Write images to a directory.

    :param imgs: List of images to write to directory.
    :param dir: Directory to write the images to.
    :return: None
    """
    for i in range(len(imgs)):
        img = imgs[i]
        # Write image to directory
        cv2.imwrite(dir + '/' + str(i).zfill(2) + '.png', img)
