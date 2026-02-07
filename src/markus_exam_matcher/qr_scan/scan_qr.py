"""
MarkUs Exam Matcher: Reading QR code

Information
===============================
This module defines the function that reads the QR code.
"""

import os.path
import sys
from typing import Optional

import cv2
import zxingcpp
from pdf2image import convert_from_path


def read_qr(img_path: str) -> str:
    img = cv2.imread(img_path)
    result = zxingcpp.read_barcode(
        img, try_rotate=False, try_downscale=False, formats=zxingcpp.QRCode
    )

    if result is None:
        print("Could not find any barcode.", file=sys.stderr)
        sys.exit(1)
    else:
        return result.text


def scan_qr_codes_from_pdfs(paths: list[str], dpi: Optional[int] = None) -> None:
    """Scan QR codes from the provided single-page PDFs.

    Print the QR codes scanned from each page (one per page) and the orientation of the QR code,
    which is used to detect rotated pages.

    The dpi argument can be used to specify the resolution at which the page is converted to an image.
    """
    for pdf_path in paths:
        pdf_filename = os.path.basename(pdf_path)
        try:
            pages = convert_from_path(pdf_path, dpi=(dpi or 150), fmt="jpeg", single_file=True)
            page = pages[0]

            data = zxingcpp.read_barcode(
                page,
                try_rotate=False,
                try_downscale=False,
                formats=zxingcpp.QRCode,
            )

            if not data:
                # Try a higher resolution (if dpi is unspecified) and with rotation and downscaling
                pages = convert_from_path(pdf_path, dpi=(dpi or 300), fmt="jpeg", single_file=True)
                page = pages[0]
                data = zxingcpp.read_barcode(
                    page,
                    try_rotate=True,
                    try_downscale=True,
                    formats=zxingcpp.QRCode,
                )

            if data:
                print(f'{pdf_filename},"{data.text}",{data.orientation}')
            else:
                print(f'{pdf_filename},""')
        except Exception:
            print(f'{pdf_filename},""')
