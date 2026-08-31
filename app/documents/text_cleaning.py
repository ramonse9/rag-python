import re


class TextCleaningService:
    """
    Limpia el texto extraído de un documento PDF.
    """

    def clean(self, text: str) -> str:
        """
        Normaliza el texto y elimina elementos innecesarios.
        """

        cleaned_text = re.sub(
            r"--\s*\d+\s+of\s+\d+\s*--",
            "",
            text,
            flags=re.IGNORECASE,
        )

        cleaned_text = (
            cleaned_text
            .replace("\r\n", "\n")
            .replace("\r", "\n")
        )

        cleaned_text = re.sub(
            r"[ \t]+\n",
            "\n",
            cleaned_text,
        )

        cleaned_text = re.sub(
            r"\n{3,}",
            "\n\n",
            cleaned_text,
        )

        return cleaned_text.strip()