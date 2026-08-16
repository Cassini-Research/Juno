from __future__ import annotations

import threading
from pathlib import Path

from juno_v2.contracts.context import TypedContextBundle
from juno_v2.memory.bias import RecognitionBiasEngine
from juno_v2.memory.store import JsonMemoryStore
from juno_v2.modes.defaults import BUILTIN_MODES
from juno_v2.personalization.seed.load_bundle import load_seed_bundle
from juno_v2.personalization.seed.runtime import JunoSeedPersonalizationRuntime
from juno_v2.workbench.server import WorkbenchApp


def _runtime(tmp_path: Path) -> tuple[JunoSeedPersonalizationRuntime, JsonMemoryStore]:
    seed = load_seed_bundle(Path(__file__).resolve().parents[1] / "seed_data")
    memory = JsonMemoryStore(tmp_path / "memory")
    return JunoSeedPersonalizationRuntime(seed, memory_store=memory), memory


def _irl_rule(runtime: JunoSeedPersonalizationRuntime) -> dict:
    return next(
        row
        for row in runtime.list_default_replacements()
        if row["trigger"].casefold() == "in real life"
        and row["replacement"].casefold() == "irl"
    )


def _normalize(
    runtime: JunoSeedPersonalizationRuntime,
    memory: JsonMemoryStore,
    *,
    mode: str,
    app_name: str = "Discord",
) -> str:
    snapshot = memory.snapshot()
    context = TypedContextBundle(app_name=app_name, app_category="messaging")
    attachment = runtime.build_seed_attachment(
        snapshot=snapshot,
        context=context,
        context_plane_suppression=None,
    )
    engine = RecognitionBiasEngine()
    plan = engine.build_plan(
        utterance_id=f"visible-seed-{mode}",
        snapshot=snapshot,
        context=context,
        mode_policy=BUILTIN_MODES[mode],
        effective_mode=mode,
        seed_attachment=attachment,
    )
    return engine.normalize_transcript(
        "in real life",
        snapshot=snapshot,
        plan=plan,
        scope="oneshot",
    ).normalized_text


def _app_for_endpoints(
    runtime: JunoSeedPersonalizationRuntime,
    memory: JsonMemoryStore,
) -> WorkbenchApp:
    app = object.__new__(WorkbenchApp)
    app.memory = memory
    app.juno_seed_runtime = runtime
    app._lock = threading.RLock()
    return app


def test_default_seed_replacements_are_visible_through_memory_endpoint(tmp_path: Path) -> None:
    runtime, memory = _runtime(tmp_path)
    memory.add_replacement(trigger="my email", replacement="me@example.com")
    app = _app_for_endpoints(runtime, memory)

    result = app.broker_memory_replacement_list()

    assert result["ok"] is True
    assert any(row["trigger"] == "my email" and not row["is_builtin"] for row in result["entries"])
    irl = next(row for row in result["entries"] if row.get("seed_rule_id") == _irl_rule(runtime)["seed_rule_id"])
    assert irl["trigger"] == "in real life"
    assert irl["replacement"] == "IRL"
    assert irl["is_builtin"] is True
    assert irl["inactive_in_verbatim"] is True


def test_seed_replacement_is_disabled_automatically_in_verbatim(tmp_path: Path) -> None:
    runtime, memory = _runtime(tmp_path)

    assert _normalize(runtime, memory, mode="default_surface") == "IRL"
    assert _normalize(runtime, memory, mode="verbatim") == "in real life"


def test_textedit_does_not_route_through_the_single_letter_x_rule(tmp_path: Path) -> None:
    runtime, memory = _runtime(tmp_path)

    assert (
        _normalize(runtime, memory, mode="default_surface", app_name="TextEdit")
        == "in real life"
    )


def test_user_can_edit_and_remove_default_replacement_persistently(tmp_path: Path) -> None:
    runtime, memory = _runtime(tmp_path)
    app = _app_for_endpoints(runtime, memory)
    rule = _irl_rule(runtime)

    edited = app.broker_memory_replacement_upsert(
        {
            "seed_rule_id": rule["seed_rule_id"],
            "trigger": "in real life",
            "replacement": "IRL-custom",
        }
    )
    assert edited["ok"] is True

    reloaded = JunoSeedPersonalizationRuntime(runtime.seed_layer, memory_store=memory)
    edited_rule = next(
        row
        for row in reloaded.list_default_replacements()
        if row["seed_rule_id"] == rule["seed_rule_id"]
    )
    assert edited_rule["replacement"] == "IRL-custom"
    assert edited_rule["source"] == "builtin_seed_override"
    assert _normalize(reloaded, memory, mode="default_surface") == "IRL-custom"

    reloaded_app = _app_for_endpoints(reloaded, memory)
    removed = reloaded_app.broker_memory_replacement_remove(
        {
            "seed_rule_id": rule["seed_rule_id"],
            "trigger": edited_rule["trigger"],
            "scope": edited_rule["scope"],
        }
    )
    assert removed["ok"] is True
    assert removed["removed"] is True

    after_restart = JunoSeedPersonalizationRuntime(runtime.seed_layer, memory_store=memory)
    assert all(
        row["seed_rule_id"] != rule["seed_rule_id"]
        for row in after_restart.list_default_replacements()
    )
    assert _normalize(after_restart, memory, mode="default_surface") == "in real life"
