from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from typing import Mapping

from juno_core_v3.model_registry.contracts import PackageSignature

# Ed25519 package signing is intentionally out of scope; only HMAC-SHA256 is verified here.

SUPPORTED_ALGOS: frozenset[str] = frozenset({"hmac-sha256"})


@dataclass(frozen=True, slots=True)
class SignatureVerdict:
    ok: bool
    reason: str
    algo: str


def canonical_payload(package_dict: dict) -> bytes:
    """JSON-canonicalise the package dict with `signature` cleared.

    Stable across Python runs: sort_keys=True, compact separators, no
    whitespace, ensure_ascii=False. The `signature` key is set to None
    so we never hash the signature into its own payload.
    """
    scrubbed = dict(package_dict)
    scrubbed["signature"] = None
    return json.dumps(
        scrubbed,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def compute_hmac_signature(payload: bytes, key: bytes) -> str:
    """Return hex-encoded HMAC-SHA256 of ``payload`` under ``key``."""
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


def verify_signature(
    *,
    payload: bytes,
    signature: PackageSignature | None,
    trust_keys: Mapping[str, bytes],
) -> SignatureVerdict:
    """Constant-time verify that ``signature`` was produced by one of
    the trusted keys."""
    if signature is None:
        return SignatureVerdict(ok=False, reason="missing_signature", algo="")

    if signature.algo not in SUPPORTED_ALGOS:
        return SignatureVerdict(ok=False, reason="unsupported_algorithm", algo=signature.algo)

    try:
        actual_bytes = bytes.fromhex(signature.value)
    except ValueError:
        return SignatureVerdict(ok=False, reason="bad_signature", algo=signature.algo)

    if len(actual_bytes) != hashlib.sha256().digest_size:
        return SignatureVerdict(ok=False, reason="bad_signature", algo=signature.algo)

    for _key_id, key_bytes in trust_keys.items():
        expected_hex = compute_hmac_signature(payload, key_bytes)
        expected_bytes = bytes.fromhex(expected_hex)
        if hmac.compare_digest(actual_bytes, expected_bytes):
            return SignatureVerdict(ok=True, reason="ok", algo=signature.algo)

    return SignatureVerdict(ok=False, reason="bad_signature", algo=signature.algo)
