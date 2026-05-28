"""Operator CLI for the model registry.

Wraps :mod:`juno_core_v3.model_registry` so operators can inspect the
signed default registry, verify signatures, promote / rollback
packages, and emit machine-readable output for scripts.

Subcommands:

* ``list``    — list packages (optionally filtered by slot).
* ``verify``  — re-verify every package's HMAC signature against the
                resolved keystore. Exits non-zero on any failure.
* ``promote`` — move a package to PROMOTED.
* ``stage``   — move a package to STAGED.
* ``retire``  — move a package to RETIRED.
* ``rollback``— print the rollback target for a package.

Shared options:

* ``--json``  — emit structured JSON instead of human-readable text.
                Keeps ``scripts/registry_cli_v3.sh`` scriptable.
* ``--no-verify`` — build the registry without the signature check.
                Useful for inspecting dev checkouts where the keystore
                is not configured. Never use in production scripts.

This module is intentionally importable so tests exercise ``main``
directly rather than spawning a subprocess.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from juno_core_v3.model_registry.contracts import ModelSlot
from juno_core_v3.model_registry.defaults import build_default_registry
from juno_core_v3.model_registry.keystore import load_keystore
from juno_core_v3.model_registry.registry import ModelPackage, ModelRegistry
from juno_core_v3.model_registry.signature import (
    SignatureVerdict,
    canonical_payload,
    verify_signature,
)


def _load_registry(*, sign: bool) -> ModelRegistry:
    """Build the default registry, either signed or unsigned."""
    return build_default_registry(sign=sign)


def _parse_slot(value: str | None) -> ModelSlot | None:
    if value is None:
        return None
    try:
        return ModelSlot(value)
    except ValueError as exc:
        raise SystemExit(f"unknown slot: {value}") from exc


def _pkg_summary(pkg: ModelPackage) -> dict:
    return {
        "package_id": pkg.package_id,
        "version": pkg.version,
        "slot": pkg.manifest.slot.value,
        "backend": pkg.manifest.backend.value,
        "promotion": pkg.promotion.value,
        "streaming": pkg.manifest.streaming,
        "languages": list(pkg.manifest.languages),
        "min_ram_mb": pkg.manifest.min_ram_mb,
        "rollback_target": pkg.rollback_target,
        "signed": pkg.signature is not None,
    }


def _cmd_list(args: argparse.Namespace) -> int:
    reg = _load_registry(sign=not args.no_verify)
    slot = _parse_slot(args.slot)
    pkgs = reg.list(slot=slot)
    summaries = [_pkg_summary(p) for p in pkgs]
    if args.json:
        json.dump({"packages": summaries}, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0
    if not summaries:
        print("(no packages)")
        return 0
    print(
        f"{'PACKAGE_ID':<42}{'SLOT':<14}{'BACKEND':<22}{'PROMOTION':<12}STREAM  RAM_MB"
    )
    for s in summaries:
        stream = "yes" if s["streaming"] else "no"
        ram = s["min_ram_mb"] if s["min_ram_mb"] is not None else "-"
        print(
            f"{s['package_id']:<42}{s['slot']:<14}{s['backend']:<22}"
            f"{s['promotion']:<12}{stream:<7}{ram}"
        )
    return 0


def _verify_one(pkg: ModelPackage, trust_keys) -> SignatureVerdict:
    payload = canonical_payload(pkg.to_dict())
    return verify_signature(
        payload=payload,
        signature=pkg.signature,
        trust_keys=trust_keys,
    )


def _cmd_verify(args: argparse.Namespace) -> int:
    """Re-verify every package against the configured keystore.

    The default-registry builder already verifies at load time. This
    command re-runs the check explicitly so operators can confirm
    signatures after shipping without trusting the bootstrap path.
    """
    trust_keys = load_keystore()
    if trust_keys is None:
        msg = "no_keystore_resolved"
        if args.json:
            json.dump({"ok": False, "error": msg}, sys.stdout)
            sys.stdout.write("\n")
        else:
            print(f"ERROR: {msg}", file=sys.stderr)
        return 2

    reg = _load_registry(sign=True)
    results = []
    all_ok = True
    for pkg in reg.list(slot=_parse_slot(args.slot)):
        verdict = _verify_one(pkg, trust_keys)
        if not verdict.ok:
            all_ok = False
        results.append(
            {
                "package_id": pkg.package_id,
                "ok": verdict.ok,
                "reason": verdict.reason,
                "algo": verdict.algo,
            }
        )
    if args.json:
        json.dump({"ok": all_ok, "results": results}, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        for r in results:
            mark = "OK  " if r["ok"] else "FAIL"
            print(f"{mark} {r['package_id']:<42}{r['algo']:<16}{r['reason']}")
        print(f"{'ALL_OK' if all_ok else 'HAS_FAILURES'} ({len(results)} packages)")
    return 0 if all_ok else 1


def _mutate(args: argparse.Namespace, op: str) -> int:
    """Shared body for promote / stage / retire.

    We explicitly build the registry *unsigned* for mutation because
    the state transitions are local to the in-memory object and are
    reported back to the operator — there is no persistence yet, so
    treating the registry as immutable in verified mode would lie
    about what the CLI actually does. The JSON output makes this
    explicit so scripts can decide whether to proceed.
    """
    reg = _load_registry(sign=not args.no_verify)
    try:
        if op == "promote":
            reg.promote(args.package_id)
        elif op == "stage":
            reg.stage(args.package_id)
        elif op == "retire":
            reg.retire(args.package_id)
        else:  # pragma: no cover — router table error
            raise SystemExit(f"unknown mutate op: {op}")
    except KeyError:
        payload = {"ok": False, "error": "package_not_found", "package_id": args.package_id}
        if args.json:
            json.dump(payload, sys.stdout)
            sys.stdout.write("\n")
        else:
            print(f"ERROR: package not found: {args.package_id}", file=sys.stderr)
        return 2

    pkg = reg.get(args.package_id)
    assert pkg is not None  # mutate would have raised otherwise
    payload = {
        "ok": True,
        "op": op,
        "package_id": pkg.package_id,
        "promotion": pkg.promotion.value,
    }
    if args.json:
        json.dump(payload, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        print(f"{op}: {pkg.package_id} → {pkg.promotion.value}")
    return 0


def _cmd_promote(args: argparse.Namespace) -> int:
    return _mutate(args, "promote")


def _cmd_stage(args: argparse.Namespace) -> int:
    return _mutate(args, "stage")


def _cmd_retire(args: argparse.Namespace) -> int:
    return _mutate(args, "retire")


def _cmd_rollback(args: argparse.Namespace) -> int:
    """Resolve the rollback target for a package without mutating state."""
    reg = _load_registry(sign=not args.no_verify)
    try:
        target = reg.rollback(args.package_id)
    except KeyError:
        payload = {"ok": False, "error": "package_not_found", "package_id": args.package_id}
    except ValueError as exc:
        payload = {"ok": False, "error": str(exc), "package_id": args.package_id}
    else:
        payload = {
            "ok": True,
            "package_id": args.package_id,
            "rollback_target": target.package_id,
            "target_promotion": target.promotion.value,
        }
    if args.json:
        json.dump(payload, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        if payload["ok"]:
            print(
                f"rollback: {args.package_id} → {payload['rollback_target']} "
                f"({payload['target_promotion']})"
            )
        else:
            print(f"ERROR: {payload['error']}", file=sys.stderr)
    return 0 if payload["ok"] else 2


def _build_parser() -> argparse.ArgumentParser:
    # Shared global flags live on a parent parser so operators can put
    # `--json` either before or after the subcommand — argparse otherwise
    # rejects `cli list --json` because the subparser doesn't know the
    # flag. Using ``parents=`` wires the exact same flag into both.
    global_flags = argparse.ArgumentParser(add_help=False)
    global_flags.add_argument("--json", action="store_true", help="emit JSON output")
    global_flags.add_argument(
        "--no-verify",
        action="store_true",
        help="skip signature verification when loading the default registry",
    )

    # Only subparsers inherit the shared flags. Attaching them to the
    # top-level parser as well causes argparse to overwrite the value
    # when both occurrences trigger (the child's default=False wins),
    # so `cli --json list` would silently drop the flag. Subparser-only
    # placement means operators write `cli list --json` — consistent
    # with how git, kubectl, aws and friends parse flags.
    parser = argparse.ArgumentParser(
        prog="registry_cli_v3",
        description="Operator interface for the Juno Core v3 model registry.",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="list registered packages", parents=[global_flags])
    p_list.add_argument("--slot", help="filter to one slot (e.g. preview_asr)")
    p_list.set_defaults(func=_cmd_list)

    p_verify = sub.add_parser("verify", help="re-verify package signatures", parents=[global_flags])
    p_verify.add_argument("--slot", help="limit to one slot")
    p_verify.set_defaults(func=_cmd_verify)

    p_promote = sub.add_parser("promote", help="mark PROMOTED", parents=[global_flags])
    p_promote.add_argument("package_id")
    p_promote.set_defaults(func=_cmd_promote)

    p_stage = sub.add_parser("stage", help="mark STAGED", parents=[global_flags])
    p_stage.add_argument("package_id")
    p_stage.set_defaults(func=_cmd_stage)

    p_retire = sub.add_parser("retire", help="mark RETIRED", parents=[global_flags])
    p_retire.add_argument("package_id")
    p_retire.set_defaults(func=_cmd_retire)

    p_rollback = sub.add_parser(
        "rollback",
        help="show the rollback_target for a package (no mutation)",
        parents=[global_flags],
    )
    p_rollback.add_argument("package_id")
    p_rollback.set_defaults(func=_cmd_rollback)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
