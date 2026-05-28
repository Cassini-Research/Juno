#!/usr/bin/env python3
"""Apply release metadata and Sparkle OTA settings to Juno.app Info.plist."""

from __future__ import annotations

import argparse
import base64
import os
import plistlib
import re
import sys
from pathlib import Path
from urllib.parse import urlparse


SPARKLE_KEYS = (
    "SUFeedURL",
    "SUPublicEDKey",
    "SUEnableAutomaticChecks",
    "SUAllowsAutomaticUpdates",
    "SUAutomaticallyUpdate",
    "SUScheduledCheckInterval",
    "SUShowReleaseNotes",
)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _optional(value: str | None) -> str | None:
    if value is None:
        return None
    trimmed = value.strip()
    return trimmed or None


def _validate_version(value: str, *, field: str) -> str:
    if not re.fullmatch(r"[0-9A-Za-z][0-9A-Za-z._+-]*", value):
        raise ValueError(f"{field} must be a non-empty version token, got {value!r}")
    return value


def _validate_feed_url(raw: str, *, allow_insecure: bool) -> str:
    parsed = urlparse(raw)
    allowed = {"https"}
    if allow_insecure:
        allowed.update({"http", "file"})
    if parsed.scheme not in allowed or not parsed.netloc and parsed.scheme != "file":
        allowed_text = ", ".join(sorted(allowed))
        raise ValueError(f"OTA feed URL must use one of: {allowed_text}")
    return raw


def _validate_public_ed_key(raw: str) -> str:
    try:
        decoded = base64.b64decode(raw, validate=True)
    except Exception as exc:  # pragma: no cover - exact binascii type varies by Python
        raise ValueError("Sparkle public EdDSA key must be valid base64") from exc
    if len(decoded) != 32:
        raise ValueError("Sparkle public EdDSA key must decode to 32 bytes")
    return raw


def _apply_disabled(plist: dict) -> None:
    for key in SPARKLE_KEYS:
        plist.pop(key, None)
    plist.pop("JunoUpdateChannel", None)
    plist["JunoOTAEnabled"] = False


def configure(
    plist: dict,
    *,
    app_version: str | None,
    build_number: str | None,
    feed_url: str | None,
    public_ed_key: str | None,
    channel: str | None,
    disable_ota: bool,
    allow_insecure_feed: bool,
    automatic_checks: bool,
    automatic_downloads: bool,
    scheduled_interval: float,
) -> dict:
    if app_version is not None:
        plist["CFBundleShortVersionString"] = _validate_version(app_version, field="app version")
    if build_number is not None:
        plist["CFBundleVersion"] = _validate_version(build_number, field="build number")

    if disable_ota or (feed_url is None and public_ed_key is None):
        _apply_disabled(plist)
        return plist

    if feed_url is None or public_ed_key is None:
        raise ValueError("OTA requires both --ota-feed-url and --ota-public-ed-key")

    plist["JunoOTAEnabled"] = True
    plist["SUFeedURL"] = _validate_feed_url(feed_url, allow_insecure=allow_insecure_feed)
    plist["SUPublicEDKey"] = _validate_public_ed_key(public_ed_key)
    plist["SUEnableAutomaticChecks"] = bool(automatic_checks)
    plist["SUAllowsAutomaticUpdates"] = True
    plist["SUAutomaticallyUpdate"] = bool(automatic_downloads)
    plist["SUScheduledCheckInterval"] = float(scheduled_interval)
    plist["SUShowReleaseNotes"] = True
    if channel is not None and channel != "stable":
        plist["JunoUpdateChannel"] = _validate_version(channel, field="OTA channel")
    else:
        plist.pop("JunoUpdateChannel", None)
    return plist


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plist", type=Path)
    parser.add_argument("--from-env", action="store_true", help="Read JUNO_* release variables from the environment.")
    parser.add_argument("--version", default=None)
    parser.add_argument("--build", default=None)
    parser.add_argument("--ota-feed-url", default=None)
    parser.add_argument("--ota-public-ed-key", default=None)
    parser.add_argument("--ota-channel", default=None)
    parser.add_argument("--disable-ota", action="store_true")
    parser.add_argument("--allow-insecure-feed", action="store_true")
    parser.add_argument("--automatic-checks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--automatic-downloads", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--scheduled-interval", type=float, default=86400.0)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    app_version = _optional(args.version)
    build_number = _optional(args.build)
    feed_url = _optional(args.ota_feed_url)
    public_ed_key = _optional(args.ota_public_ed_key)
    channel = _optional(args.ota_channel)
    disable_ota = args.disable_ota
    allow_insecure_feed = args.allow_insecure_feed
    automatic_checks = bool(args.automatic_checks)
    automatic_downloads = bool(args.automatic_downloads)
    scheduled_interval = float(args.scheduled_interval)

    if args.from_env:
        app_version = _optional(os.getenv("JUNO_APP_VERSION")) or app_version
        build_number = _optional(os.getenv("JUNO_BUILD_NUMBER")) or build_number
        feed_url = _optional(os.getenv("JUNO_OTA_FEED_URL")) or feed_url
        public_ed_key = _optional(os.getenv("JUNO_OTA_PUBLIC_ED_KEY")) or public_ed_key
        channel = _optional(os.getenv("JUNO_OTA_CHANNEL")) or channel
        disable_ota = _env_bool("JUNO_OTA_DISABLED", disable_ota)
        allow_insecure_feed = _env_bool("JUNO_OTA_ALLOW_INSECURE_FEED", allow_insecure_feed)
        automatic_checks = _env_bool("JUNO_OTA_AUTOMATIC_CHECKS", automatic_checks)
        automatic_downloads = _env_bool("JUNO_OTA_AUTOMATIC_DOWNLOADS", automatic_downloads)
        raw_interval = _optional(os.getenv("JUNO_OTA_SCHEDULED_INTERVAL"))
        if raw_interval is not None:
            scheduled_interval = float(raw_interval)

    with args.plist.open("rb") as f:
        plist = plistlib.load(f)

    try:
        configured = configure(
            plist,
            app_version=app_version,
            build_number=build_number,
            feed_url=feed_url,
            public_ed_key=public_ed_key,
            channel=channel,
            disable_ota=disable_ota,
            allow_insecure_feed=allow_insecure_feed,
            automatic_checks=automatic_checks,
            automatic_downloads=automatic_downloads,
            scheduled_interval=scheduled_interval,
        )
    except ValueError as exc:
        print(f"configure_juno_macos_plist.py: {exc}", file=sys.stderr)
        return 2

    with args.plist.open("wb") as f:
        plistlib.dump(configured, f, sort_keys=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
