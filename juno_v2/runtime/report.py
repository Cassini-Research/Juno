from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Print Juno v2 runtime truth report from a summary JSON file')
    parser.add_argument('--summary-json', type=Path, required=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    payload = json.loads(args.summary_json.read_text(encoding='utf-8'))
    truth = payload.get('metadata', {}).get('runtime_truth', {})
    print(json.dumps(truth, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
