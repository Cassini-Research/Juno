from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class GateResult:
    name: str
    severity: str
    passed: bool
    observed: Any
    op: str
    expected: Any
    why: str = ''

    def to_dict(self) -> dict[str, Any]:
        return {
            'name': self.name,
            'severity': self.severity,
            'passed': self.passed,
            'observed': self.observed,
            'op': self.op,
            'expected': self.expected,
            'why': self.why,
        }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run Juno v2 summary gates against a gate JSON')
    parser.add_argument('--summary-json', type=Path, required=True)
    parser.add_argument('--gate-json', type=Path, required=True)
    return parser


def _resolve_path(payload: dict[str, Any], path: str) -> Any:
    current: Any = payload
    for segment in path.split('.'):
        if isinstance(current, dict) and segment in current:
            current = current[segment]
        else:
            return None
    return current


def _evaluate(op: str, observed: Any, expected: Any) -> bool:
    if observed is None:
        return False
    if op == '<=':
        return float(observed) <= float(expected)
    if op == '>=':
        return float(observed) >= float(expected)
    if op == '<':
        return float(observed) < float(expected)
    if op == '>':
        return float(observed) > float(expected)
    if op == '==':
        return observed == expected
    raise ValueError(f'Unsupported gate op: {op}')


def run_gates(summary_payload: dict[str, Any], gate_payload: dict[str, Any]) -> dict[str, Any]:
    results: list[GateResult] = []
    for item in gate_payload.get('checks', []):
        observed = _resolve_path(summary_payload, item['path'])
        passed = _evaluate(item['op'], observed, item['value'])
        results.append(GateResult(
            name=item['name'],
            severity=item.get('severity', 'fail'),
            passed=passed,
            observed=observed,
            op=item['op'],
            expected=item['value'],
            why=item.get('why', ''),
        ))
    hard_failures = [r for r in results if (not r.passed and r.severity == 'fail')]
    warnings = [r for r in results if (not r.passed and r.severity != 'fail')]
    return {
        'profile': gate_payload.get('profile', 'unknown'),
        'ok': not hard_failures,
        'hard_failure_count': len(hard_failures),
        'warning_count': len(warnings),
        'results': [r.to_dict() for r in results],
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    summary_payload = json.loads(args.summary_json.read_text(encoding='utf-8'))
    gate_payload = json.loads(args.gate_json.read_text(encoding='utf-8'))
    print(json.dumps(run_gates(summary_payload, gate_payload), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
