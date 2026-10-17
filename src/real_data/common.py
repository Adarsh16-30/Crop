"""
Shared helpers for the real-data modules: output locations, an HTTP session
with retries, and a seekable file object over HTTP range requests (used to
pull single members out of large remote ZIP archives).
"""

import io
import logging
import os
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

REAL_DATA_DIR = os.environ.get('REAL_DATA_DIR', 'data/real')
REAL_REPORTS_DIR = os.environ.get('REAL_REPORTS_DIR', 'reports')

USER_AGENT = 'crop-disease-detection/2.0 (research prototype)'


def data_dir(*parts: str) -> str:
    path = os.path.join(REAL_DATA_DIR, *parts)
    os.makedirs(path, exist_ok=True)
    return path


def reports_dir(*parts: str) -> str:
    path = os.path.join(REAL_REPORTS_DIR, *parts)
    os.makedirs(path, exist_ok=True)
    return path


def http_session(retries: int = 5, backoff: float = 1.0) -> requests.Session:
    """requests.Session that retries transient failures with exponential backoff."""
    session = requests.Session()
    retry = Retry(total=retries, backoff_factor=backoff,
                  status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=frozenset({'GET', 'HEAD', 'POST'}))
    adapter = HTTPAdapter(max_retries=retry, pool_maxsize=32)
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    session.headers['User-Agent'] = USER_AGENT
    return session


def download_file(url: str, dest: str, session: requests.Session | None = None,
                  chunk: int = 1 << 20) -> str:
    """Stream `url` to `dest` (atomic rename). Skips the download if `dest` exists."""
    if os.path.exists(dest):
        return dest
    session = session or http_session()
    os.makedirs(os.path.dirname(dest) or '.', exist_ok=True)
    tmp = dest + '.part'
    with session.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, 'wb') as f:
            for block in r.iter_content(chunk):
                f.write(block)
    os.replace(tmp, dest)
    return dest


class HTTPRangeFile(io.RawIOBase):
    """
    Read-only, seekable file over HTTP range requests.

    Wrapped in io.BufferedReader it can be handed to zipfile.ZipFile, which
    then reads only the central directory and the members it is asked for,
    instead of the whole archive.
    """

    def __init__(self, url: str, session: requests.Session | None = None,
                 attempts: int = 10):
        self.session = session or http_session()
        self.source_url = url
        self.pos = 0
        self.attempts = attempts
        self._resolve()

    def _resolve(self) -> None:
        """
        Follow redirects to the content URL and learn the size. Uses a 1-byte
        range GET, not HEAD: pre-signed S3 links (e.g. Dataverse downloads)
        are signed for GET only and expire, so they are re-resolved on 403.
        """
        r = self.session.get(self.source_url, headers={'Range': 'bytes=0-0'},
                             allow_redirects=True, timeout=60, stream=True)
        r.close()
        r.raise_for_status()
        if r.status_code != 206 or 'Content-Range' not in r.headers:
            raise OSError(f"{self.source_url} does not support HTTP range requests")
        self.url = r.url
        self.size = int(r.headers['Content-Range'].rsplit('/', 1)[1])

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        else:
            self.pos = self.size + offset
        return self.pos

    def read_range(self, start: int, n: int) -> bytes:
        """Bytes [start, start + n) with retries; thread-safe (no shared position)."""
        headers = {'Range': f'bytes={start}-{start + n - 1}'}
        for attempt in range(1, self.attempts + 1):
            try:
                r = self.session.get(self.url, headers=headers, timeout=300)
                if r.status_code == 206 and len(r.content) == n:
                    return r.content
                if r.status_code == 403 and self.url != self.source_url:
                    self._resolve()                  # pre-signed link expired
                    continue
                logger.warning("Range read retry %d (status %s, %d/%d bytes)",
                               attempt, r.status_code, len(r.content), n)
            except requests.RequestException as exc:   # dropped connection mid-body
                logger.warning("Range read retry %d (%s)", attempt, exc)
            time.sleep(min(2 ** attempt, 30))
        raise OSError(f"Range request failed for {self.url} at byte {start}")

    def readinto(self, buffer) -> int:
        n = min(len(buffer), self.size - self.pos)
        if n <= 0:
            return 0
        buffer[:n] = self.read_range(self.pos, n)
        self.pos += n
        return n


def open_remote_zip(url: str, session: requests.Session | None = None):
    """zipfile.ZipFile over HTTP range requests (central directory read lazily)."""
    import zipfile

    raw = HTTPRangeFile(url, session=session)
    return zipfile.ZipFile(io.BufferedReader(raw, buffer_size=1 << 20))


def read_member_parallel(zf, name: str, workers: int = 4, chunk: int = 4 << 20) -> bytes:
    """
    One member of a remote ZIP opened with open_remote_zip, fetched as
    `workers` concurrent range requests of `chunk` bytes and then inflated.

    Some repositories throttle each connection (Data INRAE: ~0.25 MB/s), so
    a few parallel ranges are several times faster than zf.read(name). Keep
    `workers` small: it is a shared public server.
    """
    import zipfile
    import zlib
    from concurrent.futures import ThreadPoolExecutor

    info = zf.getinfo(name)
    raw: HTTPRangeFile = zf.fp.raw
    local = raw.read_range(info.header_offset, 30)
    if local[:4] != b'PK\x03\x04':
        raise zipfile.BadZipFile(f"Bad local header for {name}")
    start = info.header_offset + 30 + int.from_bytes(local[26:28], 'little') + \
        int.from_bytes(local[28:30], 'little')
    offsets = range(0, info.compress_size, chunk)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(
            lambda o: raw.read_range(start + o, min(chunk, info.compress_size - o)), offsets))
    data = b''.join(parts)
    if info.compress_type == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(data, -15)
    elif info.compress_type != zipfile.ZIP_STORED:
        raise NotImplementedError(f"Compression type {info.compress_type} for {name}")
    if zlib.crc32(data) != info.CRC:
        raise zipfile.BadZipFile(f"CRC mismatch for {name}")
    return data
