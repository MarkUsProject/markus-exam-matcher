"""
MarkUs Exam Matcher: Image Transformations

Information
===============================
This module defines functions that can be used to transform an image. It
also defines a class that can be used as a pipeline to perform multiple
transformations sequentially.
"""

from __future__ import annotations

import math
from typing import Callable

import cv2
import numpy as np


class ImageTransform:
    """
    Defines a transformation pipeline that can be called on an image
    represented by a numpy ndarray.

    Instance Variables:
        - callbacks: List of transformation functions that should be called
                     in order on an image. Each function must take in only
                     one parameter (the np.ndarray image) and return only
                     an np.ndarray image.
    """

    callbacks: list[Callable]

    def __init__(self, callbacks: list[Callable]) -> None:
        self.callbacks = callbacks

    def perform_on(self, img: np.ndarray) -> np.ndarray:
        """
        Perform the transformation specified by this instance. The transformations
        are called in the order specified by the callbacks list.

        :param img: np.ndarray representing the image to perform the transformation
                    on.

        :return: The image as an np.ndarray after the transformations have been
                 applied.
        """
        for callback in self.callbacks:
            img = callback(img)

        return img


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """
    Convert an image to grayscale.

    :param img: Image to convert.
    :return: img in grayscale form.
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def to_inverted(img: np.ndarray) -> np.ndarray:
    """
    Invert an image.

    :param img: Image to invert.
    :return: img after being inverted.
    """
    BLOCK_SIZE = 15
    C = 2
    threshed = cv2.adaptiveThreshold(
        img,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=BLOCK_SIZE,
        C=C,
    )
    return threshed


def to_closed(img: np.ndarray) -> np.ndarray:
    """
    Perform closing on an image. Closing is useful for "closing" small
    holes in any numbers or letters. It also slightly thickens the
    contours, which makes digits more similar to MNIST dataset digits.

    :param img: Image to close.
    :return: img after being closed.
    """
    KERNEL_SIZE = (3, 3)
    ITERATIONS = 2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, KERNEL_SIZE)
    img = cv2.morphologyEx(img, op=cv2.MORPH_CLOSE, kernel=kernel, iterations=ITERATIONS)
    return img


def thicken_lines(img: np.ndarray) -> np.ndarray:
    """
    Thicken the lines of an inverted, grayscale image.

    :param img: Grayscale, inverted image to thicken.
    :return: img after its lines are thickened.
    """
    # Thicken lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.dilate(img, kernel, iterations=2)


def get_lines(img: np.ndarray, kernel_length: int = 50) -> np.ndarray:
    """
    Get an image containing only horizontal and vertical lines in the input image.

    :param img: Grayscale, inverted image to get lines from.
    :param kernel_length: Length of horizontal and vertical kernels. It is recommended
                          to keep this value at its default value.
    :return: Image containing only horizontal and vertical lines from the original image.
    """
    # Create structuring elements (kernels)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))

    # Apply kernels to get vertical and horizontal masks
    # These masks contain only vertical and horizontal lines, respectively
    vertical_mask = cv2.morphologyEx(img, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    horizontal_mask = cv2.morphologyEx(img, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)

    # Add masks together to get new mask containing both horizontal and vertical lines
    img_mask = cv2.bitwise_or(horizontal_mask, vertical_mask)
    return img_mask


def shift(img: np.ndarray, sx: int, sy: int) -> np.ndarray:
    """
    Shifts the image by the given x and y units.
    :param img: input image.
    :param sx: x units to shift by.
    :param sy: y units to shift by.
    :return: shifted image.

    Disclaimer: Function written by the authors at https://opensourc.es/blog/tensorflow-mnist/.
    """
    rows, cols = img.shape
    M = np.float32([[1, 0, sx], [0, 1, sy]])
    shifted = cv2.warpAffine(img, M, (cols, rows))
    return shifted


def process_num(gray: np.ndarray) -> np.ndarray:
    """
    Process an input image of a handwritten number in the same way the MNIST dataset was processed.
    :param gray: the input grayscaled image.
    :return: the processed image.

    Disclaimer: Function written by the authors at https://opensourc.es/blog/tensorflow-mnist/.
    """
    # Crop to the largest contour
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    gray = gray[y : y + h, x : x + w]

    # Deskew
    m = cv2.moments(gray)
    rows, cols = gray.shape

    if abs(m["mu02"]) >= 0.01:
        skew = m["mu11"] / m["mu02"]
        transform = np.float32([[1, skew, -0.5 * rows * skew], [0, 1, 0]])
        gray = cv2.warpAffine(
            gray, transform, (cols, rows), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR
        )

    # reshape image to be 20x20
    rows, cols = gray.shape
    if rows > cols:
        factor = 20.0 / rows
        rows = 20
        cols = int(round(cols * factor))
    else:
        factor = 20.0 / cols
        cols = 20
        rows = int(round(rows * factor))
    gray = cv2.resize(gray, (cols, rows))

    # pad the image to be 28x28
    colsPadding = (int(math.ceil((28 - cols) / 2.0)), int(math.floor((28 - cols) / 2.0)))
    rowsPadding = (int(math.ceil((28 - rows) / 2.0)), int(math.floor((28 - rows) / 2.0)))
    gray = np.pad(gray, (rowsPadding, colsPadding), "constant")

    # shift the image is the written number is centered
    shiftx, shifty = _get_best_shift(gray)
    gray = shift(gray, shiftx, shifty)
    return gray


def _get_best_shift(img: np.ndarray) -> tuple[int, int]:
    """
    Finds x and y units to shift the image by so it is centered.
    :param img: input image.
    :return: best x and y units to shift by.

    Disclaimer: Function written by the authors at https://opensourc.es/blog/tensorflow-mnist/.
    """
    m = cv2.moments(img)
    if m["m00"] == 0:
        return 0, 0

    cx = m["m10"] / m["m00"]
    cy = m["m01"] / m["m00"]

    rows, cols = img.shape
    shiftx = np.round(cols / 2.0 - cx).astype(int)
    shifty = np.round(rows / 2.0 - cy).astype(int)

    return shiftx, shifty
