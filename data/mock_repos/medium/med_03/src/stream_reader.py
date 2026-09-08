class ChunkedUTF8Reader:
    def __init__(self, chunk_size: int = 4):
        self.chunk_size = chunk_size

    def decode_stream(self, byte_data: bytes) -> str:
        """Decodes raw bytes arriving in chunks into a complete unicode string."""
        chunks = [byte_data[i:i + self.chunk_size] for i in range(0, len(byte_data), self.chunk_size)]
        text_parts = []
        # BUG: decodes each chunk separately with strict utf-8 decoding
        for chunk in chunks:
            text_parts.append(chunk.decode("utf-8"))
        return "".join(text_parts)
