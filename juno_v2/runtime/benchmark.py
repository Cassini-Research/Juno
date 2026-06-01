from __future__ import annotations

import argparse
import json
from pathlib import Path

from juno_v2.runtime.benchmark_data import load_builtin_cases
from juno_v2.runtime.benchmark_suite import BenchmarkCase, build_benchmark_suite_report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Build Juno v2 benchmark suite report from summary JSON artifacts')
    parser.add_argument('--summary-json', type=Path, required=True)
    parser.add_argument('--cases-json', type=Path, default=None)
    parser.add_argument('--builtin-suites', default='', help='comma-separated builtin suites: multilingual,memory')
    parser.add_argument('--compare-summary-json', type=Path, default=None, help='optional second summary for baseline-vs-streaming comparison')
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary_payload = json.loads(args.summary_json.read_text(encoding='utf-8'))
    cases: list[BenchmarkCase] = []
    if args.cases_json is not None:
        raw_cases = json.loads(args.cases_json.read_text(encoding='utf-8'))
        cases.extend(BenchmarkCase(**item) for item in raw_cases)
    if args.builtin_suites.strip():
        cases.extend(load_builtin_cases(name.strip() for name in args.builtin_suites.split(',')))
    comparison_payload = None
    if args.compare_summary_json is not None:
        comparison_payload = json.loads(args.compare_summary_json.read_text(encoding='utf-8'))
    report = build_benchmark_suite_report(summary_payload, cases=cases, comparison_summary_payload=comparison_payload)
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
