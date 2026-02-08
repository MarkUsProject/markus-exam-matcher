# Changelog

## [unreleased]

## [0.4.0]

- Switched QR code scanning to scan full page and report orientation
- Added second attempt for QR code scanning with higher default resolution, `try_rotate=True`, and `try_downscale=True`
- Removed dependency on `scipy`
- Improved digit OCR pre-processing and performed some code cleanup

## [0.3.0]

- Adopt uv for project structure
- Update dependencies and modify supported Python versions to 3.11-3.14
