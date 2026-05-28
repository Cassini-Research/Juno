from __future__ import annotations

import argparse
import json
from pathlib import Path

from juno_v2.demo.config import DEFAULT_DEMO_PROFILE, DemoConfig, DemoPaths, load_demo_config
from juno_v2.demo.models import provision_demo_models


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bootstrap Juno local assets")
    parser.add_argument("--profile", default=DEFAULT_DEMO_PROFILE)
    parser.add_argument("--root-dir", default=".juno_v2_demo")
    parser.add_argument("--skip-model-download", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    paths = DemoPaths(root_dir=Path(args.root_dir))
    config = load_demo_config(paths=paths, profile_name=args.profile)
    if config.profile_name != args.profile:
        config = DemoConfig.from_profile(args.profile, paths=paths)
    if not args.skip_model_download:
        config = provision_demo_models(config, paths=paths, force=args.force)
    else:
        paths.ensure_dirs()
        config.save(paths=paths)
    print(json.dumps({
        "ok": True,
        "profile_name": config.profile_name,
        "profile_class": config.profile_class,
        "target_machine": config.target_machine,
        "notes": list(config.notes),
        "speech_profile": config.speech_profile,
        "preview_model_path": str(config.preview_model_path),
        "final_model_path": str(config.final_model_path),
        "config_json": str(paths.resolved_config_json()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
