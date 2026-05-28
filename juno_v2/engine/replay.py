from __future__ import annotations

import argparse
import json
from pathlib import Path

from juno_v2.audio.sources import ReplayWavAudioSource
from juno_v2.context.provider import WorkbenchContextProvider
from juno_v2.engine.factory import CanonicalEngineBuildSpec, build_canonical_engine
from juno_v2.language.catalog import DEFAULT_SUPPORTED_LANGUAGES
from juno_v2.speech.config import DEFAULT_SPEECH_PROFILE
from juno_v2.runtime.ids import new_session_id


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run Juno v2 integrated dictation replay')
    parser.add_argument('--wav-path', type=Path, required=True)
    parser.add_argument('--preview-model-path', type=Path, required=True)
    parser.add_argument('--final-model-path', type=Path, required=True)
    parser.add_argument('--preview-backend', default='faster_whisper')
    parser.add_argument('--final-backend', default='faster_whisper')
    parser.add_argument('--preview-endpoint', default=None)
    parser.add_argument('--final-endpoint', default=None)
    parser.add_argument('--language', default=None)
    parser.add_argument('--speech-profile', default=DEFAULT_SPEECH_PROFILE)
    parser.add_argument('--language-policy', default='auto_supported')
    parser.add_argument('--supported-languages', default=','.join(DEFAULT_SUPPORTED_LANGUAGES))
    parser.add_argument('--memory-dir', default='.juno_v2_memory')
    parser.add_argument('--writer-backend', default=None)
    parser.add_argument('--writer-endpoint', default=None)
    parser.add_argument('--writer-model-path', default=None)
    parser.add_argument('--summary-json-out', default=None)
    parser.add_argument('--context-source', default='workbench', choices=['static', 'workbench', 'linux_desktop', 'macos_desktop'])
    parser.add_argument('--context-helper-command', default=None)
    parser.add_argument('--insertion-target', default='none', choices=['none', 'macos_active_app'])
    parser.add_argument('--insertion-helper-command', default=None)
    parser.add_argument('--gpu-memory-budget-mb', type=int, default=None)
    parser.add_argument('--preview-gpu-memory-mb', type=int, default=0)
    parser.add_argument('--final-gpu-memory-mb', type=int, default=0)
    parser.add_argument('--writer-gpu-memory-mb', type=int, default=0)
    parser.add_argument('--preview-residency-group', default=None)
    parser.add_argument('--final-residency-group', default=None)
    parser.add_argument('--preview-residency-policy', default='resident')
    parser.add_argument('--final-residency-policy', default='resident')
    parser.add_argument('--writer-residency-policy', default='resident')
    parser.add_argument('--platform-name', default=None)
    parser.add_argument('--preview-device', default='auto')
    parser.add_argument('--final-device', default='auto')
    parser.add_argument('--preview-compute-type', default='default')
    parser.add_argument('--final-compute-type', default='default')
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    session_id = new_session_id(prefix='dictation_replay')
    artifacts = build_canonical_engine(CanonicalEngineBuildSpec(
        engine_mode='canonical_replay',
        session_id=session_id,
        preview_model_path=args.preview_model_path,
        final_model_path=args.final_model_path,
        preview_backend_name=args.preview_backend,
        final_backend_name=args.final_backend,
        preview_endpoint=args.preview_endpoint,
        final_endpoint=args.final_endpoint,
        writer_backend_name=args.writer_backend,
        writer_endpoint=args.writer_endpoint,
        writer_model_path=args.writer_model_path,
        language=args.language,
        language_policy=args.language_policy,
        supported_languages=tuple(item.strip() for item in args.supported_languages.split(',') if item.strip()),
        speech_profile_name=args.speech_profile,
        memory_dir=args.memory_dir,
        trace_log_dir=Path('logs_v2'),
        context_source=args.context_source,
        context_helper_command=args.context_helper_command,
        insertion_target=args.insertion_target,
        insertion_helper_command=args.insertion_helper_command,
        gpu_memory_budget_mb=args.gpu_memory_budget_mb,
        preview_gpu_memory_mb=args.preview_gpu_memory_mb,
        final_gpu_memory_mb=args.final_gpu_memory_mb,
        writer_gpu_memory_mb=args.writer_gpu_memory_mb,
        preview_residency_group=args.preview_residency_group,
        final_residency_group=args.final_residency_group,
        preview_residency_policy=args.preview_residency_policy,
        final_residency_policy=args.final_residency_policy,
        writer_residency_policy=args.writer_residency_policy,
        platform_name=args.platform_name,
        preview_device=args.preview_device,
        final_device=args.final_device,
        preview_compute_type=args.preview_compute_type,
        final_compute_type=args.final_compute_type,
    ))
    runner = artifacts.runner
    store = artifacts.store
    if args.context_source == 'workbench':
        runner.context_provider = WorkbenchContextProvider(store)
    recorder = artifacts.recorder
    summary = runner.run(ReplayWavAudioSource(args.wav_path).frames())
    print(summary.to_dict())
    if args.summary_json_out:
        Path(args.summary_json_out).write_text(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2), encoding='utf-8')
    print(store.snapshot())
    print(f'trace_log={recorder.log_path}')


if __name__ == '__main__':
    main()
