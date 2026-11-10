"""Small helpers shared by benchmarking commands."""
import hashlib
from pathlib import Path


def sha256_file(path: str | Path, chunk_bytes: int = 1024 * 1024) -> str:
    """Hash large input files without reading them into memory at once."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(chunk_bytes), b''):
            digest.update(chunk)
    return digest.hexdigest()
