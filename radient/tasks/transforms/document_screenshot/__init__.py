__all__ = [
    "DocumentScreenshotTransform",
    "PyMuPDFDocumentScreenshotTransform"
]

from typing import Optional

from radient.tasks.transforms.document_screenshot._base import DocumentScreenshotTransform
from radient.tasks.transforms.document_screenshot.pymupdf import PyMuPDFDocumentScreenshotTransform


def pdf_to_screenshot_transform(method: str = "PyMuPDF", **kwargs) -> DocumentScreenshotTransform:
    """Creates a transform which performs document screenshotting.
    """

    if method.lower() in ("pymupdf", None):
        return PyMuPDFDocumentScreenshotTransform(**kwargs)
    else:
        raise NotImplementedError
