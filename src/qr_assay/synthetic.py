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
ONION_LABEL_RE = re.compile(r"([a-z2-7]{56}|[a-z2-7]{16})(?=\.onion)", re.IGNORECASE)
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
NULL_MODES = {"token_shuffle", "class_permute", "grammar_random"}


def _rng(seed: int, match_id: int, corpus: str, mode: str) -> random.Random:
    material = f"{seed}:{match_id}:{corpus}:{mode}".encode()
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
) -> tuple[int, list[tuple[int, int]], re.Match[str] | None]:
    scheme_end = payload.find("://") + 3 if "://" in payload else 0
    protected: list[tuple[int, int]] = []
    if scheme_end:
        protected.append((0, scheme_end))
    onion_label_match = ONION_LABEL_RE.search(payload) if corpus == "onion" else None
    if corpus == "onion":
        suffix_start = payload.lower().find(".onion")
        if suffix_start >= 0:
            protected.append((suffix_start, suffix_start + 6))
    return scheme_end, protected, onion_label_match


def _is_protected(index: int, protected: list[tuple[int, int]]) -> bool:
    return any(start <= index < end for start, end in protected)


def _replace_onion_label(chars: list[str], match: re.Match[str] | None, rng: random.Random) -> None:
    if match is None:
        return
    label = synthetic_onion_label(len(match.group(1)), rng)
    chars[match.start(1) : match.end(1)] = list(label)


def _token_shuffle(payload: str, rng: random.Random, corpus: str) -> str:
    chars = list(payload)
    _, protected, onion_match = _protected_ranges(payload, corpus)
    # A v3 onion host is cryptographic rather than linguistic. Replace it with
    # another checksum-valid address, then apply the shuffle only to remaining tokens.
    _replace_onion_label(chars, onion_match, rng)
    host_range = (onion_match.start(1), onion_match.end(1)) if onion_match else None
    for match in TOKEN_RE.finditer(payload):
        indices = [
            i
            for i in range(match.start(), match.end())
            if not _is_protected(i, protected)
            and not (host_range and host_range[0] <= i < host_range[1])
        ]
        values = [chars[i] for i in indices]
        rng.shuffle(values)
        for i, value in zip(indices, values, strict=True):
            chars[i] = value
    return "".join(chars)


def _class_permute(payload: str, rng: random.Random, corpus: str) -> str:
    chars = list(payload)
    _, protected, onion_match = _protected_ranges(payload, corpus)
    _replace_onion_label(chars, onion_match, rng)
    host_range = (onion_match.start(1), onion_match.end(1)) if onion_match else None
    groups: dict[str, list[int]] = {"lower": [], "upper": [], "digit": []}
    for i, char in enumerate(payload):
        if _is_protected(i, protected) or (host_range and host_range[0] <= i < host_range[1]):
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


def _grammar_random(payload: str, rng: random.Random, corpus: str) -> str:
    chars = list(payload)
    _, protected, onion_match = _protected_ranges(payload, corpus)
    _replace_onion_label(chars, onion_match, rng)
    host_range = (onion_match.start(1), onion_match.end(1)) if onion_match else None
    for i, char in enumerate(payload):
        if _is_protected(i, protected) or (host_range and host_range[0] <= i < host_range[1]):
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
) -> str:
    """Generate a deterministic structural null without changing UTF-8 length.

    Modes form an ordered null ladder for ordinary URLs:
    ``token_shuffle`` preserves each token's character multiset,
    ``class_permute`` preserves global ASCII unigram counts inside character classes,
    and ``grammar_random`` preserves delimiters and character-class positions only.

    For v3 onion host labels the ladder intentionally collapses to a fresh,
    checksum-valid random onion address because the host itself is cryptographic,
    not a natural-language token. Path/query material still follows the chosen mode.
    Synthetic addresses are never resolved.
    """
    if mode not in NULL_MODES:
        raise ValueError(f"Unknown synthetic null mode {mode!r}; choose from {sorted(NULL_MODES)}")
    rng = _rng(seed, match_id, corpus, mode)
    if mode == "token_shuffle":
        result = _token_shuffle(payload, rng, corpus)
    elif mode == "class_permute":
        result = _class_permute(payload, rng, corpus)
    else:
        result = _grammar_random(payload, rng, corpus)

    if result == payload:
        # Degenerate short strings can be unchanged after a permutation. Fall back
        # to the weakest grammar-preserving randomization, still deterministically.
        result = _grammar_random(payload, rng, corpus)
    if len(result.encode("utf-8")) != len(payload.encode("utf-8")):
        raise AssertionError("Synthetic payload did not preserve UTF-8 byte length")
    return result
