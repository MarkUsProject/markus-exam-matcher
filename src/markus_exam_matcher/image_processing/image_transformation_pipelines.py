"""
MarkUs Exam Matcher: Image Transformation Pipelines

Information
===============================
This module defines pipelines that can be used to transform images
into a certain form.
"""
from .image_transformations import ImageTransform, to_grayscale, to_inverted, \
    to_closed, get_lines, thicken_lines, process_num

# Image transformation pipelines
PREPROCESSING_PIPELINE = ImageTransform([
    to_grayscale,
    to_inverted,
    to_closed
])

BOX_DETECTION_PIPELINE = ImageTransform([
    get_lines,
    thicken_lines
])

MNIST_NUM_PIPELINE = ImageTransform([
    process_num
])

# TODO: Verify process_num still works with letters (i.e, data from
# EMNIST dataset. If so, refactor name).

MNIST_LETTER_PIPELINE = ImageTransform([
    process_num
])
