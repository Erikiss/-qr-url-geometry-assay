from __future__ import annotations

import bz2
import gzip
import hashlib
import io
import lzma
import re
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Iterable, Iterator
from pathlib import Path

ONION_RE = re.compile(
    r"(?:(?:https?://))?"
    r"(?:[a-z0-9-]+\.)*"
    r"(?:[a-z2-7]{56}|[a-z2-7]{16})\.onion"
    r"(?::\d{1,5})?"
    r"(?:/[^\s\"'<>]*)?",
    re.IGNORECASE,
)


def _text_stream(path: Path) -> Iterator[str]:
    suffix = path.suffix.lower()
    if suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            yield from handle
    elif suffix in {".xz", ".lzma"}:
        with lzma.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            yield from handle
    elif suffix == ".bz2":
        with bz2.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            yield from handle
    elif suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            for name in sorted(archive.namelist()):
                if name.endswith("/"):
                    continue
                with (
                    archive.open(name) as raw,
                    io.TextIOWrapper(raw, encoding="utf-8", errors="replace") as handle,
                ):
                    yield from handle
    else:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            yield from handle


def expand_paths(items: Iterable[str]) -> list[Path]:
    resolved: list[Path] = []
    for item in items:
        path = Path(item)
        if path.is_dir():
            resolved.extend(p for p in sorted(path.rglob("*")) if p.is_file())
        elif any(ch in item for ch in "*?["):
            resolved.extend(sorted(path.parent.glob(path.name)))
        elif path.is_file():
            resolved.append(path)
        else:
            raise FileNotFoundError(f"Source path does not exist: {path}")
    return resolved


def iter_source_lines(paths: Iterable[str]) -> Iterator[tuple[str, str]]:
    for path in expand_paths(paths):
        for line in _text_stream(path):
            yield str(path), line.rstrip("\r\n")


def _host_ascii(host: str) -> str | None:
    try:
        return host.rstrip(".").encode("idna").decode("ascii").lower()
    except (UnicodeError, ValueError):
        return None


def normalize_surface(value: str) -> str | None:
    value = value.strip().strip("\"'")
    if not value or value.startswith("#"):
        return None
    if "," in value and "://" not in value:
        value = value.split(",", 1)[0].strip()
    candidate = value if "://" in value else f"https://{value}"
    try:
        parsed = urllib.parse.urlsplit(candidate)
        host = _host_ascii(parsed.hostname or "")
        if not host or "." not in host or host.endswith(".onion"):
            return None
        port = f":{parsed.port}" if parsed.port else ""
        path = parsed.path or "/"
        return urllib.parse.urlunsplit(
            (parsed.scheme.lower() or "https", host + port, path, parsed.query, "")
        )
    except (ValueError, UnicodeError):
        return None


def normalize_onion(value: str) -> list[str]:
    found: list[str] = []
    for match in ONION_RE.finditer(value):
        candidate = match.group(0).rstrip(".,);]")
        candidate = candidate if "://" in candidate else f"http://{candidate}"
        try:
            parsed = urllib.parse.urlsplit(candidate)
            host = (parsed.hostname or "").lower()
            label = host.rsplit(".onion", 1)[0].split(".")[-1]
            if len(label) not in {16, 56} or any(
                ch not in "abcdefghijklmnopqrstuvwxyz234567" for ch in label
            ):
                continue
            port = f":{parsed.port}" if parsed.port else ""
            path = parsed.path or "/"
            found.append(
                urllib.parse.urlunsplit(
                    (parsed.scheme.lower(), host + port, path, parsed.query, "")
                )
            )
        except ValueError:
            continue
    return found


def stable_u64(value: str) -> int:
    return int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "big")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_inventory(
    paths: Iterable[str], hash_inputs: bool = True
) -> list[dict[str, str | int | None]]:
    records: list[dict[str, str | int | None]] = []
    for path in expand_paths(paths):
        stat = path.stat()
        records.append(
            {
                "path": str(path.resolve()),
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": sha256_file(path) if hash_inputs else None,
            }
        )
    return records


def iter_urls(
    kind: str,
    paths: Iterable[str],
    scan_limit: int | None = None,
    deduplicate: bool = True,
) -> Iterator[tuple[str, str]]:
    seen: set[int] | None = set() if deduplicate else None
    accepted = 0
    for source, line in iter_source_lines(paths):
        values = [normalize_surface(line)] if kind == "surface" else normalize_onion(line)
        for value in values:
            if not value:
                continue
            if seen is not None:
                digest = stable_u64(value)
                if digest in seen:
                    continue
                seen.add(digest)
            yield source, value
            accepted += 1
            if scan_limit is not None and accepted >= int(scan_limit):
                return


def download_file(url: str, destination: Path, expected_sha256: str | None = None) -> str:
    if urllib.parse.urlsplit(url).scheme != "https":
        raise ValueError("Only HTTPS downloads are allowed")
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    request = urllib.request.Request(url, headers={"User-Agent": "qr-url-geometry-assay/0.1"})
    with (
        urllib.request.urlopen(request, timeout=60) as response,
        destination.open("wb") as output,
    ):
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            output.write(chunk)
    actual = digest.hexdigest()
    if expected_sha256 and actual.lower() != expected_sha256.lower():
        destination.unlink(missing_ok=True)
        raise ValueError(f"SHA-256 mismatch for {url}: expected {expected_sha256}, got {actual}")
    return actual


def acquire_manifest(path: str | Path, output_dir: str | Path) -> list[dict[str, str]]:
    import yaml

    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        spec = yaml.safe_load(handle) or {}
    output = Path(output_dir)
    records: list[dict[str, str]] = []
    for item in spec.get("downloads", []):
        name = item.get("name") or Path(urllib.parse.urlsplit(item["url"]).path).name
        destination = output / name
        sha = download_file(item["url"], destination, item.get("sha256"))
        records.append({"url": item["url"], "path": str(destination), "sha256": sha})
    return records
