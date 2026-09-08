import pytest
from src.stream_reader import ChunkedUTF8Reader

def test_ascii_chunks():
    reader = ChunkedUTF8Reader(chunk_size=3)
    data = b"Hello, World!"
    assert reader.decode_stream(data) == "Hello, World!"

def test_split_multibyte_emoji():
    # FAILS ON UNPATCHED CODE
    reader = ChunkedUTF8Reader(chunk_size=3)
    emoji_bytes = "🚀Launch!".encode("utf-8")
    result = reader.decode_stream(emoji_bytes)
    assert result == "🚀Launch!"
