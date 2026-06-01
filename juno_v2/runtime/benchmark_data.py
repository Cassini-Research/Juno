from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from juno_v2.runtime.benchmark_suite import BenchmarkCase


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def benchmark_dir() -> Path:
    return repo_root() / 'eval' / 'v2_benchmarks'


def load_cases_json(path: Path) -> list[BenchmarkCase]:
    raw = json.loads(path.read_text(encoding='utf-8'))
    return [BenchmarkCase(**item) for item in raw]


def load_builtin_cases(names: Iterable[str]) -> list[BenchmarkCase]:
    merged: list[BenchmarkCase] = []
    for name in names:
        key = name.strip().lower()
        if not key:
            continue
        if key == 'multilingual':
            merged.extend(load_cases_json(benchmark_dir() / 'multilingual_replay_cases.json'))
        elif key == 'memory':
            merged.extend(load_cases_json(benchmark_dir() / 'memory_benefit_cases.json'))
        else:
            raise ValueError(f'Unknown benchmark suite: {name}')
    # de-dup by utterance id while preserving order
    seen: set[str] = set()
    out: list[BenchmarkCase] = []
    for case in merged:
        if case.utterance_id in seen:
            continue
        seen.add(case.utterance_id)
        out.append(case)
    return out
