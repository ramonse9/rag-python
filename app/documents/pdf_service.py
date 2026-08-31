from io import BytesIO

from pypdf import PdfReader
from pypdf.errors import PdfReadError


class PDFExtractionError(ValueError):
    """
    Error producido cuando un PDF no puede procesarse.
    """

    pass


class PDFService:
    """
    Extrae texto de documentos PDF.
    """

    def extract_text(self, content: bytes) -> str:
        """
        Extrae y combina el texto de todas las páginas.

        Args:
            content:
                Contenido binario completo del PDF.

        Returns:
            Texto extraído de todas las páginas.

        Raises:
            PDFExtractionError:
                Cuando el archivo está vacío, dañado
                o protegido mediante contraseña.
        """

        if not content:
            raise PDFExtractionError(
                "The PDF file is empty."
            )

        try:
            reader = PdfReader(
                BytesIO(content),
                strict=False,
            )
        except (PdfReadError, OSError, ValueError) as error:
            raise PDFExtractionError(
                "Could not read the PDF file."
            ) from error

        if reader.is_encrypted:
            try:
                password_result = reader.decrypt("")
            except Exception as error:
                raise PDFExtractionError(
                    "The PDF file is password protected."
                ) from error

            if password_result == 0:
                raise PDFExtractionError(
                    "The PDF file is password protected."
                )

        extracted_pages: list[str] = []

        for page_number, page in enumerate(
            reader.pages,
            start=1,
        ):
            try:
                page_text = page.extract_text() or ""
            except Exception as error:
                raise PDFExtractionError(
                    f"Could not extract text from page {page_number}."
                ) from error

            normalized_page_text = page_text.strip()

            if normalized_page_text:
                extracted_pages.append(
                    normalized_page_text
                )

        return "\n\n".join(extracted_pages).strip()