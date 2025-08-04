"""
MarkUs Exam Matcher: Reading QR code

Information
===============================
This module defines the function that reads the QR code.
"""

import os.path
import sys

import cv2
import numpy as np
import zxingcpp
from pdf2image import convert_from_path


def read_qr(img_path: str) -> str:
    img = cv2.imread(img_path)
    results = zxingcpp.read_barcodes(img)

    if len(results) == 0:
        print("Could not find any barcode.", file=sys.stderr)
        sys.exit(1)
    else:
        result = results[0]
        return result.text


def scan_qr_codes_from_pdfs(paths: list[str], dpi: int = 400, top_fraction: float = 0.2) -> None:
    """Scan QR codes from the provided single-page PDFs, checking only the top portion of each page.

    Print the QR codes scanned from each page (one per page).
    """
    detector = cv2.QRCodeDetector()
    for pdf_path in paths:
        pdf_filename = os.path.basename(pdf_path)
        try:
            pages = convert_from_path(
                pdf_path, dpi=dpi, fmt="jpeg", single_file=True, grayscale=True
            )
            page = pages[0]
            # Convert PIL image to OpenCV format
            cv_image = np.array(page)

            # Crop top fraction of the image
            height = cv_image.shape[0]
            crop_height = int(height * top_fraction)
            cropped = cv_image[:crop_height, :]

            # Detect and decode
            data, _, _ = detector.detectAndDecode(cropped)

            if data:
                print(f'{pdf_filename},"{data}"')
            else:
                print(f'{pdf_filename},""')
        except Exception:
            print(f'{pdf_filename},""')
