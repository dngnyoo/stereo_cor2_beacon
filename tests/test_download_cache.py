"""Offline acceptance tests for immutable per-date raw FITS reuse."""

from cme_pipeline.data_preparation import download_cor2


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload
        self.text = payload.decode() if payload.startswith(b"<") else ""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size=1024 * 1024):
        yield self.payload


class FakeSession:
    def __init__(self):
        self.requested = []

    def get(self, url, **kwargs):
        self.requested.append(url)
        if url.endswith("/"):
            date = url.rstrip("/").split("/")[-1]
            name = f"{date}_002424_d7c2A.fts"
            return FakeResponse(f'<a href="{name}">{name}</a>'.encode())
        return FakeResponse(b"valid-payload-for-testing")


def test_completed_dates_are_reused_without_http(tmp_path):
    client = FakeSession()
    root = tmp_path / "data" / "raw" / "cor2" / "science"
    first = download_cor2("20240101", "science_raw", root / "20240101", session=client)
    assert len(first) == 1
    assert len(client.requested) == 2
    assert (root / "20240101" / ".download_complete.json").is_file()

    replay = download_cor2("20240101", "science_raw", root / "20240101", session=client)
    assert replay == first
    assert len(client.requested) == 2  # no request, even directory listing

    second = download_cor2("20240102", "science_raw", root / "20240102", session=client)
    assert len(second) == 1
    assert len(client.requested) == 4  # only the new date downloads


def test_corrupted_cached_file_is_refetched(tmp_path):
    client = FakeSession()
    path = tmp_path / "science" / "20240101"
    files = download_cor2("20240101", "science_raw", path, session=client)
    files[0].write_bytes(b"corrupted-but-nonempty")
    replay = download_cor2("20240101", "science_raw", path, session=client)
    assert replay[0].read_bytes() == b"valid-payload-for-testing"
    assert len(client.requested) == 4


def test_missing_cached_file_is_refetched(tmp_path):
    client = FakeSession()
    path = tmp_path / "science" / "20240101"
    files = download_cor2("20240101", "science_raw", path, session=client)
    files[0].unlink()
    replay = download_cor2("20240101", "science_raw", path, session=client)
    assert replay[0].exists()
    assert len(client.requested) == 4
