import sys

import cv2
import zxingcpp


def read_qr(img_path: str) -> str:
    img = cv2.imread(img_path)
    results = zxingcpp.read_barcodes(img)

    if len(results) == 0:
        print("Could not find any barcode.")
        sys.exit(1)
    else:
        result = results[0]
        return result.text
