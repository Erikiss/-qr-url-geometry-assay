from __future__ import annotations

import base64
import hashlib
import random
import re
import string

LOWER = string.ascii_lowercase
UPPER = string.ascii_uppercase
DIGITS = string.digits
BASE32 = "abcdefghijklmnopqrstuvwxyz234567"
HEX_LOWER = "abcdef"
HEX_UPPER = "ABCDEF"
ONION_LABEL_RE = re.compile(r"([a-z2-7]{56}|[a-z2-7]{16})(?=\.onion)", re.IGNORECASE)
PERCENT_RE = re.compile(r"%[0-9A-Fa-f]{2}")
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
NULL_MODES = {"token_shuffle", "class_permute", "grammar_random"}


def _rng(seed: int, unit_key: str, corpus: str, mode: str) -> random.Random:
    material = f"{seed}:{unit_key}:{corpus}:{mode}".encode("utf-8")
    return random.Random(int.from_bytes(hashlib.sha256(material).digest()[:8], "big"))


def synthetic_onion_label(length: int, rng: random.Random) -> str:
    """Generate a syntactically valid legacy/v3 onion service label."""
    if length == 56:
        public_key = bytes(rng.getrandbits(8) for _ in range(32))
        version = b"\x03"
        checksum = hashlib.sha3_256(b".onion checksum" + public_key + version).digest()[:2]
        return base64.b32encode(public_key + checksum + version).decode("ascii").lower()
    if length == 16:
        return (
            base64.b32encode(bytes(rng.getrandbits(8) for _ in range(10))).decode("ascii").lower()
        )
    raise ValueError(f"Unsupported onion label length: {length}")


def _protected_ranges(
    payload: str, corpus: str
) -> tuple[list[tuple[int, int]], re.Match[str] | None, list[tuple[int, int]]]:
    scheme_end = payload.find("://") + 3 if "://" in payload else 0
    protected: list[tuple[int, int]] = []
    if scheme_end:
        protected.append((0, scheme_end))
    onion_label_match = ONION_LABEL_RE.search(payload) if corpus == "onion" else None
    if corpus == "onion":
        suffix_start = payload.lower().find(".onion")
        if suffix_start >= 0:
            protected.append((suffix_start, suffix_start + 6))
    percent_ranges = [(match.start(), match.end()) for match in PERCENT_RE.finditer(payload)]
    return protected, onion_label_match, percent_ranges


def _is_protected(index: int, ranges: list[tuple[int, int]]) -> bool:
    return any(start <= index < end for start, end in ranges)


def _replace_onion_label(chars: list[str], match: re.Match[str] | None, rng: random.Random) -> None:
    if match is None:
        return
    label = synthetic_onion_label(len(match.group(1)), rng)
    chars[match.start(1) : match.end(1)] = list(label)


def _token_shuffle(payload: str, rng: random.Random, corpus: str) -> str:
    chars = list(payload)
    protected, onion_match, percent_ranges = _protected_ranges(payload, corpus)
    # A v3 onion host is cryptographic rather than linguistic. Replace it with
    # another checksum-valid address, then shuffle remaining ordinary tokens.
    # Valid %HH escape triplets are kept atomic/exact under this strongest null.
    _replace_onion_label(chars, onion_match, rng)
    host_range = (onion_match.start(1), onion_match.end(1)) if onion_match else None
    all_protected = protected + percent_ranges
    for match in TOKEN_RE.finditer(payload):
        indices = [
            i
            for i in range(match.start(), match.end())
            if not _is_protected(i, all_protected)
            and not (host_range and host_range[0] <= i < host_range[1])
        ]
        values = [chars[i] for i in indices]
        rng.shuffle(values)
        for i, value in zip(indices, values, strict=True):
            chars[i] = value
    return "".join(chars)


def _class_permute(payload: str, rng: random.Random, corpus: str) -> str:
    chars = list(payload)
    protected, onion_match, percent_ranges = _protected_ranges(payload, corpus)
    _replace_onion_label(chars, onion_match, rng)
    host_range = (onion_match.start(1), onion_match.end(1)) if onion_match else None
    all_protected = protected + percent_ranges
    groups: dict[str, list[int]] = {"lower": [], "upper": [], "digit": []}
    for i, char in enumerate(payload):
        if _is_protected(i, all_protected) or (host_range and host_range[0] <= i < host_range[1]):
            continue
        if char.isascii() and char.islower():
            groups["lower"].append(i)
        elif char.isascii() and char.isupper():
            groups["upper"].append(i)
        elif char.isdigit():
            groups["digit"].append(i)
    for indices in groups.values():
        values = [chars[i] for i in indices]
        rng.shuffle(values)
        for i, value in zip(indices, values, strict=True):
            chars[i] = value
    return "".join(chars)


def _randomize_percent_triplets(chars: list[str], payload: str, rng: random.Random) -> None:
    for match in PERCENT_RE.finditer(payload):
        for index in (match.start() + 1, match.start() + 2):
            original = payload[index]
            if original.isdigit():
                chars[index] = rng.choice(DIGITS)
            elif original.islower():
                chars[index] = rng.choice(HEX_LOWER)
            else:
                chars[index] = rng.choice(HEX_UPPER)


def _grammar_random(payload: str, rng: random.Random, corpus: str) -> str:
    chars = list(payload)
    protected, onion_match, percent_ranges = _protected_ranges(payload, corpus)
    _replace_onion_label(chars, onion_match, rng)
    _randomize_percent_triplets(chars, payload, rng)
    host_range = (onion_match.start(1), onion_match.end(1)) if onion_match else None
    all_protected = protected + percent_ranges
    for i, char in enumerate(payload):
        if _is_protected(i, all_protected) or (host_range and host_range[0] <= i < host_range[1]):
            continue
        if char.isascii() and char.islower():
            chars[i] = rng.choice(LOWER)
        elif char.isascii() and char.isupper():
            chars[i] = rng.choice(UPPER)
        elif char.isdigit():
            chars[i] = rng.choice(DIGITS)
    return "".join(chars)


def grammar_matched(
    payload: str,
    *,
    seed: int,
    match_id: int,
    corpus: str,
    mode: str = "grammar_random",
    unit_key: str | None = None,
) -> str:
    """Generate one deterministic control from exactly the declared null family.

    Modes form an ordered null ladder for ordinary URLs:
    ``token_shuffle`` preserves each token's character multiset,
    ``class_permute`` preserves global ASCII unigram counts inside character classes,
    and ``grammar_random`` preserves delimiters and character-class positions only.

    Valid percent-encoding triplets are never made syntactically invalid. The two
    stronger nulls keep each %HH triplet exact; ``grammar_random`` resamples its
    hex digits while preserving digit/lowercase-hex/uppercase-hex position class.

    For v3 onion host labels the ladder intentionally collapses to a fresh,
    checksum-valid random onion address because the host itself is cryptographic,
    not a natural-language token. Path/query material still follows the chosen mode.
    Synthetic addresses are never resolved.

    ``unit_key`` can explicitly define the natural observational unit. When it is
    omitted, the SHA-256 of the effective payload is used, so the same payload gets
    the same null draw even if matching order or match_id changes. ``match_id`` is
    retained in the public signature for backwards compatibility only.

    A degenerate payload is allowed to remain unchanged when the declared null has
    no distinct permutation. We never silently fall back to a weaker null family.
    """
    if mode not in NULL_MODES:
        raise ValueError(f"Unknown synthetic null mode {mode!r}; choose from {sorted(NULL_MODES)}")
    del match_id
    effective_unit_key = (
        unit_key if unit_key is not None else hashlib.sha256(payload.encode("utf-8")).hexdigest()
    )
    rng = _rng(seed, effective_unit_key, corpus, mode)
    if mode == "token_shuffle":
        result = _token_shuffle(payload, rng, corpus)
    elif mode == "class_permute":
        result = _class_permute(payload, rng, corpus)
    else:
        result = _grammar_random(payload, rng, corpus)

    if len(result.encode("utf-8")) != len(payload.encode("utf-8")):
        raise AssertionError("Synthetic payload did not preserve UTF-8 byte length")
    return result
