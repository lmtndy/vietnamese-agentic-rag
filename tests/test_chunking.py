import pytest

from src.chunking import chunk_text


def test_chunks_overlap_and_cover_text():
    text = "".join(str(i % 10) for i in range(1000))
    chunks = chunk_text(text, size=300, overlap=50)
    assert all(len(c) <= 300 for c in chunks)
    assert chunks[0][-50:] == chunks[1][:50]      # 50 ký tự cuối đoạn trước = 50 ký tự đầu đoạn sau
    assert chunks[-1].endswith(text[-10:])        # không mất phần cuối văn bản


def test_whitespace_is_normalized():
    assert chunk_text("a \n\n  b\tc", size=100, overlap=10) == ["a b c"]


def test_empty_text_gives_no_chunks():
    assert chunk_text("   \n", size=100, overlap=10) == []


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_text("abc", size=10, overlap=10)
