from __future__ import annotations

import hashlib
import hmac
import json

from juno_core_v3.model_registry.contracts import PackageSignature
from juno_core_v3.model_registry.signature import (
    SUPPORTED_ALGOS,
    canonical_payload,
    compute_hmac_signature,
    verify_signature,
)

KEY_A = b"\x01" * 32
KEY_B = b"\x02" * 32


def _payload_dict() -> dict:
    return {
        "package_id": "pkg.test",
        "version": "0.1",
        "manifest": {"slot": "final_asr", "min_ram_mb": 100},
        "signature": None,
        "metadata": {"note": "x"},
    }


# ---- canonical_payload ----------------------------------------------


def test_canonical_payload_is_sorted_and_compact() -> None:
    payload = canonical_payload({"b": 1, "a": 2, "signature": None})
    assert payload == b'{"a":2,"b":1,"signature":null}'


def test_canonical_payload_clears_signature_field() -> None:
    base = _payload_dict()
    signed = dict(base)
    signed["signature"] = {"algo": "hmac-sha256", "value": "ff" * 32}
    assert canonical_payload(base) == canonical_payload(signed)


def test_canonical_payload_does_not_mutate_input() -> None:
    signed = _payload_dict()
    signed["signature"] = {"algo": "hmac-sha256", "value": "aa" * 32}
    canonical_payload(signed)
    assert signed["signature"] == {"algo": "hmac-sha256", "value": "aa" * 32}


def test_canonical_payload_round_trips_to_same_dict_minus_signature() -> None:
    base = _payload_dict()
    decoded = json.loads(canonical_payload(base).decode("utf-8"))
    expected = dict(base)
    expected["signature"] = None
    assert decoded == expected


# ---- compute_hmac_signature -----------------------------------------


def test_compute_hmac_signature_is_deterministic() -> None:
    payload = canonical_payload(_payload_dict())
    sig1 = compute_hmac_signature(payload, KEY_A)
    sig2 = compute_hmac_signature(payload, KEY_A)
    assert sig1 == sig2
    # Hex-encoded SHA-256 digest is 64 chars.
    assert len(sig1) == 64
    assert bytes.fromhex(sig1)  # valid hex


def test_compute_hmac_signature_matches_stdlib() -> None:
    payload = b"hello world"
    expected = hmac.new(KEY_A, payload, hashlib.sha256).hexdigest()
    assert compute_hmac_signature(payload, KEY_A) == expected


def test_compute_hmac_signature_differs_per_key_and_payload() -> None:
    payload = canonical_payload(_payload_dict())
    assert compute_hmac_signature(payload, KEY_A) != compute_hmac_signature(payload, KEY_B)
    assert compute_hmac_signature(payload, KEY_A) != compute_hmac_signature(payload + b"x", KEY_A)


# ---- verify_signature ------------------------------------------------


def _signed(payload: bytes, key: bytes) -> PackageSignature:
    return PackageSignature(algo="hmac-sha256", value=compute_hmac_signature(payload, key))


def test_verify_signature_ok() -> None:
    payload = canonical_payload(_payload_dict())
    verdict = verify_signature(
        payload=payload,
        signature=_signed(payload, KEY_A),
        trust_keys={"key-a": KEY_A},
    )
    assert verdict.ok is True
    assert verdict.reason == "ok"
    assert verdict.algo == "hmac-sha256"


def test_verify_signature_ok_with_any_trusted_key() -> None:
    payload = canonical_payload(_payload_dict())
    verdict = verify_signature(
        payload=payload,
        signature=_signed(payload, KEY_B),
        trust_keys={"key-a": KEY_A, "key-b": KEY_B},
    )
    assert verdict.ok is True


def test_verify_signature_missing_signature() -> None:
    payload = canonical_payload(_payload_dict())
    verdict = verify_signature(payload=payload, signature=None, trust_keys={"key-a": KEY_A})
    assert verdict.ok is False
    assert verdict.reason == "missing_signature"
    assert verdict.algo == ""


def test_verify_signature_unsupported_algorithm() -> None:
    payload = canonical_payload(_payload_dict())
    assert "ed25519" not in SUPPORTED_ALGOS
    verdict = verify_signature(
        payload=payload,
        signature=PackageSignature(algo="ed25519", value="ff" * 32),
        trust_keys={"key-a": KEY_A},
    )
    assert verdict.ok is False
    assert verdict.reason == "unsupported_algorithm"
    assert verdict.algo == "ed25519"


def test_verify_signature_wrong_key() -> None:
    payload = canonical_payload(_payload_dict())
    verdict = verify_signature(
        payload=payload,
        signature=_signed(payload, KEY_B),
        trust_keys={"key-a": KEY_A},
    )
    assert verdict.ok is False
    assert verdict.reason == "bad_signature"


def test_verify_signature_tampered_payload() -> None:
    payload = canonical_payload(_payload_dict())
    tampered = _payload_dict()
    tampered["version"] = "0.2"
    verdict = verify_signature(
        payload=canonical_payload(tampered),
        signature=_signed(payload, KEY_A),
        trust_keys={"key-a": KEY_A},
    )
    assert verdict.ok is False
    assert verdict.reason == "bad_signature"


def test_verify_signature_non_hex_value_rejected() -> None:
    payload = canonical_payload(_payload_dict())
    verdict = verify_signature(
        payload=payload,
        signature=PackageSignature(algo="hmac-sha256", value="not-hex"),
        trust_keys={"key-a": KEY_A},
    )
    assert verdict.ok is False
    assert verdict.reason == "bad_signature"


def test_verify_signature_wrong_length_digest_rejected() -> None:
    payload = canonical_payload(_payload_dict())
    verdict = verify_signature(
        payload=payload,
        signature=PackageSignature(algo="hmac-sha256", value="ab" * 16),  # 16 bytes, not 32
        trust_keys={"key-a": KEY_A},
    )
    assert verdict.ok is False
    assert verdict.reason == "bad_signature"
