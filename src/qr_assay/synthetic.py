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


def _rng(seed: int, match_id: int, corpus: str) -> random.Random:
    material = f"{seed}:{match_id}:{corpus}".encode()
    return random.Random(int.from_bytes(hashlib.sha256(material).digest()[:8], "big"))


def _synthetic_onion_label(length: int, rng: random.Random) -> str:
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


def grammar_matched(payload: str, *, seed: int, match_id: int, corpus: str) -> str:
    """Randomize tokens while preserving delimiters, case classes, and byte length.

    For onion payloads, the literal `.onion` suffix is retained and the address
    alphabet is restricted to base32. The result is never resolved or fetched.
    """

    rng = _rng(seed, match_id, corpus)
    onion_suffix_start = payload.lower().find(".onion")
    scheme_end = payload.find("://") + 3 if "://" in payload else 0
    onion_host_start = scheme_end
    onion_label_match = ONION_LABEL_RE.search(payload) if corpus == "onion" else None
    onion_label = (
        _synthetic_onion_label(len(onion_label_match.group(1)), rng) if onion_label_match else None
    )
    output: list[str] = []
    for index, char in enumerate(payload):
        if (
            index < scheme_end
            or corpus == "onion"
            and onion_suffix_start <= index < onion_suffix_start + 6
        ):
            replacement = char
        elif onion_label_match and onion_label_match.start(1) <= index < onion_label_match.end(1):
            replacement = onion_label[index - onion_label_match.start(1)]
        elif (
            corpus == "onion"
            and onion_host_start <= index < onion_suffix_start
            and char.lower() in BASE32
        ):
            replacement = rng.choice(BASE32)
        elif char.islower() and char.isascii():
            replacement = rng.choice(LOWER)
        elif char.isupper() and char.isascii():
            replacement = rng.choice(UPPER)
        elif char.isdigit():
            replacement = rng.choice(DIGITS)
        else:
            replacement = char
        output.append(replacement)
    result = "".join(output)
    if result == payload:
        # Deterministically change one replaceable token without altering grammar.
        chars = list(result)
        for i, char in enumerate(chars):
            if (
                char.islower()
                and char.isascii()
                and i >= scheme_end
                and not (onion_suffix_start <= i < onion_suffix_start + 6)
            ):
                chars[i] = LOWER[(LOWER.index(char) + 1) % len(LOWER)]
                break
        result = "".join(chars)
    if len(result.encode("utf-8")) != len(payload.encode("utf-8")):
        raise AssertionError("Synthetic payload did not preserve UTF-8 byte length")
    return result
