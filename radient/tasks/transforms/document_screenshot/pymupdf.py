from io import BytesIO
from typing import TYPE_CHECKING
import urllib.request

from radient.tasks.transforms.document_screenshot._base import DocumentScreenshotTransform
from radient.utils.lazy_import import LazyImport

if TYPE_CHECKING:
    import pymupdf
    from PIL import Image
else:
    pymupdf = LazyImport("PyMuPDF", min_version="1.24.3")
    Image = LazyImport("PIL", attribute="Image", package_name="Pillow")


class PyMuPDFDocumentScreenshotTransform(DocumentScreenshotTransform):

    def __init__(self, zoom: float = 1.0):
        super().__init__()
        self._zoom = zoom
    
    def transform(self, data: str) -> dict[str, str]:

        # Ensure that the path is valid
        if not data.endswith(".pdf"):
            raise ValueError("Invalid path")
        
        # Ensure that the URL is valid
        if data.startswith("http"):
            with urllib.request.urlopen(data) as response:
                pdf_data = response.read()
            pdf_stream = BytesIO(pdf_data)
            pdf = pymupdf.open(stream=pdf_stream, filetype="pdf")
        else:
            pdf = pymupdf.open(data, filetype="pdf")

        # Create a transformation object
        mat = pymupdf.Matrix(self._zoom, self._zoom)

        # Output the results
        images = []
        for n in range(pdf.page_count):
            pix = pdf[n].get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
        
        return images
