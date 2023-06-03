import cv2
import sys
import numpy as np
import tempfile
from cnn import get_num
from typing import List, Tuple
from image_transformations import ImageTransform, to_grayscale, to_inverted, to_closed, thicken_lines,\
    process_num, get_lines

# TODO: Refactor anything that says digit to character because we will be using chars too.

# ==================================================================
# Global variables
# ==================================================================

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


# ==================================================================
# Functions
# ==================================================================
def is_empty_box(box: np.ndarray, width: int, height: int,
                 threshold: float = 0.001) -> bool:
    """
    Given a box, return whether it is empty.

    :param box: Grayscale, inverted image representing a valid
                box for students to write in.
    :param width: Width of the box.
    :param height: Height of the box.
    :param threshold: Threshold for when normalized number of markings
                      in box causes the box to be considered empty.
    :return: Whether the given box is empty.
    """
    if width == 0 or height == 0:
        return True

    # Get number of markings in box
    markings = cv2.countNonZero(box)

    # Normalize
    normalized = markings / float(width*height)

    # Return whether this normalized amount is considered empty
    return normalized < threshold


def get_digit_box_contours(contours: List[np.ndarray]) -> List[np.ndarray]:
    """
    Get contours representing the boxes that surround each digit.

    :param contours: Contours of the version of the original image
                     containing only horizontal and vertical lines.
    :return: List containing box contours.
    """
    epsilon = 0.1
    box_contours = []

    for contour in contours:
        # Get the dimensions of the bounding rectangle for this contour
        x, y, w, h = cv2.boundingRect(contour)

        # Error of ratio of width and height of shape being a square.
        # For squares, want width / height to be 1
        square_error = abs((float(w) / h) - 1)

        # If the bounding rectangle is approximately a square, add it
        # to the list of contours.
        if square_error < epsilon:
            box_contours.append(contour)

    return box_contours


def sort_contours(contours: List[np.ndarray], debug: bool = False) -> List[np.ndarray]:
    """
    Sort contours in the left-to-right order in which they appear.

    :param contours: List of contours to sort.
    :param debug: Specifies whether assertions should be checked.
    :return: contours in sorted (left-to-right) order.
    """
    # Get the indices of the contours in the correct order
    sorted_indices = np.argsort([cv2.boundingRect(i)[0] for i in contours])

    # Create list of contours in the sorted order
    sorted_contours = [None] * len(sorted_indices)
    i = 0
    for index in sorted_indices:
        cnt = contours[index]
        sorted_contours[i] = cnt
        i += 1

    if debug:
        for cnt in sorted_contours:
            assert cnt is not None

    return sorted_contours


def remove_erroneous_box_contours(box_contours: List[np.ndarray]) -> List[np.ndarray]:
    """
    Remove any box contours that should not be in the list of box contours.

    :param box_contours: List of box contours that might contain some contours
                         that are not boxes.
    :return: List of contours that are only box contours.

    Preconditions:
        - box_contours is sorted in left-to-right order.
    """
    filtered_box_contours = []

    # First remove contours that are embedded in another contour
    furthest_x = 0
    for contour in box_contours:
        x, y, w, h = cv2.boundingRect(contour)

        if x + w < furthest_x:
            continue
        else:
            filtered_box_contours.append(contour)
            furthest_x = x + w

    return filtered_box_contours


def get_digits(img: np.ndarray, filtered_contours: List[np.ndarray],
               verbose: bool = False, buf: int = 5) -> List[np.ndarray]:
    """
    Get images of the individual digits in the given image using the contours
    of the boxes around the images.

    :param img: Image containing the boxes of digits.
    :param filtered_contours: Contours of boxes of digits.
    :param verbose: If true, displays contours and digits as they are detected.
    :param buf: Pixels to be cropped off bounding box contours. Used to prevent
                parts of the bounding box borders from being included in the
                image.
    :return: List of images of digits inside boxes (not including the boxes).
    """
    digits = []

    for contour in filtered_contours:
        # Get digit inside the box containing it
        x, y, w, h = cv2.boundingRect(contour)
        # TODO: This is slightly crude. Could replace with contour detection to crop edges of boxes.
        number_image = img[y + buf:y + h - buf, x + buf:x + w - buf].copy()

        # If the box is empty, skip it.
        # TODO: For letter detection, may have spaces in name. Need to record space here.
        if is_empty_box(number_image, width=w, height=h):
            continue

        if verbose:
            _display_contour(img, contour)
            _display_img(number_image)

        # Transform digit to look similar to digits that the CNN was trained on
        # (MNIST digits)
        number_image = MNIST_NUM_PIPELINE.perform_on(number_image)

        digits.append(number_image)

        if verbose:
            _display_img(number_image)

    return digits


def write_images_to_dir(imgs: List[np.ndarray], dir: str) -> None:
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


def _display_img(img: np.ndarray) -> None:
    """
    Display the image img. Useful for debugging.

    :param img: Image to display.
    :return: None
    """
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def _display_contour(img: np.ndarray, cnt: np.ndarray,
                     colour: Tuple[int, int, int] = (0, 255, 0)) -> None:
    """
    Display the contour cnt overlaid onto a grayscale image img.

    :param img: Grayscale image to overlay contour onto.
    :param cnt: Contour to display.
    :param colour: RGB tuple defining the colour of the contour.
                   Defaults to green.
    :return: None
    """
    # Display contour
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(img_color, [cnt], 0, colour, 3)

    # Display the image
    _display_img(img_color)


if __name__ == '__main__':
    input_filename = sys.argv[1]
    debug = True

    # Read input image
    img = cv2.imread(input_filename)

    # Perform image pre-processing pipeline on img to get img in the
    # form required by most image processing functions.
    img = PREPROCESSING_PIPELINE.perform_on(img)

    if debug:
        _display_img(img)

    # Get an image containing only horizontal and vertical lines of img
    lines_of_img = BOX_DETECTION_PIPELINE.perform_on(img)

    if debug:
        _display_img(lines_of_img)

    # Get contours
    contours, _ = cv2.findContours(lines_of_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Only get contours that represent valid boxes where students write
    filtered_contours = get_digit_box_contours(contours)

    # Sort contours in left-to-right order
    sorted_contours = sort_contours(filtered_contours, debug=debug)

    # Remove potential erroneous box contours
    sorted_contours = remove_erroneous_box_contours(sorted_contours)

    # Get digits in order using these sorted boxes contours
    # TODO: Move MNIST processing outside of this function for SRP
    digits = get_digits(img, sorted_contours, verbose=debug)

    # Write digits to a temporary directory and run CNN on images
    # in this directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        with tempfile.TemporaryDirectory(dir=tmp_dir) as img_dir:
            write_images_to_dir(digits, img_dir)
            get_num(tmp_dir, img_dir, [])
