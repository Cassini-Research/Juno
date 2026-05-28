from __future__ import annotations

import argparse
import json

from juno_v2.demo.config import list_demo_profiles


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='List Juno source profiles')
    parser.add_argument('--profile-class', choices=['smoke', 'standard'], default=None)
    parser.add_argument('--json', action='store_true')
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    profiles = list_demo_profiles(profile_class=args.profile_class)
    payload = [
        {
            'name': profile.name,
            'profile_class': profile.profile_class,
            'description': profile.description,
            'preview_repo_id': profile.preview_repo_id,
            'final_repo_id': profile.final_repo_id,
            'supported_languages': list(profile.supported_languages),
            'language': profile.language,
            'language_policy': profile.language_policy,
            'preview_backend': profile.preview_backend,
            'preview_service_backend': profile.preview_service_backend,
            'final_backend': profile.final_backend,
            'writer_backend': profile.writer_backend,
            'target_machine': profile.target_machine,
            'speech_profile': profile.speech_profile,
            'notes': list(profile.notes),
        }
        for profile in profiles
    ]
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    for item in payload:
        print(f"[{item['profile_class']}] {item['name']}: {item['description']}")
        print(f"  preview={item['preview_repo_id']} ({item['preview_backend']} / {item['preview_service_backend']})")
        print(f"  final={item['final_repo_id']} ({item['final_backend']})")
        print(f"  writer={item['writer_backend'] or 'builtin_only'}")
        print(f"  languages={','.join(item['supported_languages'])} policy={item['language_policy']} speech={item['speech_profile']} target={item['target_machine']}")
        for note in item['notes']:
            print(f"  note: {note}")


if __name__ == '__main__':
    main()
