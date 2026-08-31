from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Chunk:
    """
    Representa una sección del texto original.
    """

    index: int
    content: str
    start: int
    end: int


class ChunkingService:
    """
    Divide texto en fragmentos con superposición.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError(
                "chunk_size must be greater than zero."
            )

        if chunk_overlap < 0:
            raise ValueError(
                "chunk_overlap cannot be negative."
            )

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size."
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> list[Chunk]:
        """
        Divide el texto procurando cortar en saltos de línea
        o espacios completos.
        """

        chunks: list[Chunk] = []

        start = 0
        index = 0

        while start < len(text):
            max_end = min(
                start + self.chunk_size,
                len(text),
            )

            end = self._find_best_cut_position(
                text=text,
                start=start,
                max_end=max_end,
            )

            content = text[start:end].strip()

            if content:
                chunks.append(
                    Chunk(
                        index=index,
                        content=content,
                        start=start,
                        end=end,
                    )
                )

                index += 1

            if end >= len(text):
                break

            desired_start = end - self.chunk_overlap

            next_start = self._find_overlap_start(
                text=text,
                desired_start=desired_start,
            )

            # Garantiza que el ciclo siempre avance.
            start = (
                next_start
                if next_start > start
                else end
            )

        return chunks

    def _find_best_cut_position(
        self,
        text: str,
        start: int,
        max_end: int,
    ) -> int:
        if max_end >= len(text):
            return len(text)

        search_window = min(
            200,
            max(1, self.chunk_size // 5),
        )

        search_start = max(
            start,
            max_end - search_window,
        )

        last_newline = text.rfind(
            "\n",
            search_start,
            max_end + 1,
        )

        if last_newline > start:
            return last_newline

        last_space = text.rfind(
            " ",
            search_start,
            max_end + 1,
        )

        if last_space > start:
            return last_space

        return max_end

    def _find_overlap_start(
        self,
        text: str,
        desired_start: int,
    ) -> int:
        if desired_start <= 0:
            return 0

        if desired_start >= len(text):
            return len(text)

        if text[desired_start].isspace():
            return desired_start + 1

        for position in range(
            desired_start,
            -1,
            -1,
        ):
            if text[position].isspace():
                return position + 1

        return desired_start