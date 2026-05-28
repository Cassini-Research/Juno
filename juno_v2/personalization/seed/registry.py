from __future__ import annotations

from juno_v2.personalization.seed.models import JunoPersonalizationSeedLayer, SeedTermEntry


class PackRegistry:
    """Read-only registry over a loaded :class:`JunoPersonalizationSeedLayer`."""

    def __init__(self, seed: JunoPersonalizationSeedLayer) -> None:
        self._seed = seed

    @property
    def seed(self) -> JunoPersonalizationSeedLayer:
        return self._seed

    def default_enabled_packs(self) -> tuple[str, ...]:
        return self._seed.manifest.default_enabled_packs

    def optional_packs(self) -> tuple[str, ...]:
        return self._seed.manifest.optional_packs

    def specialized_pack_name(self) -> str:
        return "specialized_distributed_protocols"

    def terms_for_pack(self, pack_name: str) -> tuple[SeedTermEntry, ...]:
        if pack_name == "personal_entities":
            return self._seed.personal_entities
        return self._seed.packs.get(pack_name, ())

    def terms_for_active_packs(self, pack_ids: tuple[str, ...]) -> tuple[SeedTermEntry, ...]:
        """Stable order: pack_ids order, then canonical within each pack."""
        out: list[SeedTermEntry] = []
        for pid in pack_ids:
            for row in sorted(self.terms_for_pack(pid), key=lambda e: e.canonical.casefold()):
                out.append(row)
        return tuple(out)
