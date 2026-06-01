from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol


class SupportsBackendName(Protocol):
    @property
    def backend_name(self) -> str: ...


@dataclass(slots=True)
class ManagedComponentState:
    role: str
    name: str
    metadata: dict = field(default_factory=dict)
    warmed: bool = False
    warm_count: int = 0
    last_warm_ms: float | None = None
    total_warm_ms: float = 0.0
    last_error: str | None = None
    gpu_memory_mb: int = 0
    residency_group: str | None = None
    shared_resident: bool = False
    health_ok: bool | None = None
    restart_count: int = 0
    residency_policy: str = 'resident'
    # Issue #19: when False, warm_all() (which runs on the main thread
    # at startup) skips this component. The expectation is that a
    # dedicated worker thread will warm it via warm_component() — used
    # for MLX preview backends whose state must bind to the decode
    # worker thread, not the main thread.
    warm_on_main_thread: bool = True
    loaded: bool = False
    acquire_count: int = 0
    release_count: int = 0
    unload_count: int = 0
    last_used_monotonic_ns: int | None = None
    # Idle-TTL unload: when >0 and policy=='on_demand', release() does not
    # unload immediately — instead it records the scheduled-unload time and
    # leaves the model resident until reap_idle() or acquire() fires the
    # lazy check past the TTL boundary.
    idle_unload_ttl_s: float = 0.0
    release_scheduled_at_ns: int | None = None

    def to_dict(self) -> dict:
        return {
            'role': self.role,
            'name': self.name,
            'metadata': dict(self.metadata),
            'warmed': self.warmed,
            'warm_count': self.warm_count,
            'last_warm_ms': self.last_warm_ms,
            'total_warm_ms': self.total_warm_ms,
            'last_error': self.last_error,
            'gpu_memory_mb': self.gpu_memory_mb,
            'residency_group': self.residency_group,
            'shared_resident': self.shared_resident,
            'health_ok': self.health_ok,
            'restart_count': self.restart_count,
            'residency_policy': self.residency_policy,
            'warm_on_main_thread': self.warm_on_main_thread,
            'loaded': self.loaded,
            'acquire_count': self.acquire_count,
            'release_count': self.release_count,
            'unload_count': self.unload_count,
            'last_used_monotonic_ns': self.last_used_monotonic_ns,
            'idle_unload_ttl_s': self.idle_unload_ttl_s,
            'release_scheduled_at_ns': self.release_scheduled_at_ns,
        }


@dataclass(slots=True)
class _ManagedComponent:
    state: ManagedComponentState
    warm_fn: Callable[[], None]
    unload_fn: Callable[[], None] | None = None
    healthcheck_fn: Callable[[], bool] | None = None
    restart_fn: Callable[[], None] | None = None


class BackendLifecycleManager:
    def __init__(self, *, total_gpu_memory_mb: int | None = None) -> None:
        self._components: list[_ManagedComponent] = []
        self._warmed_once = False
        self.total_gpu_memory_mb = total_gpu_memory_mb
        self._lock = threading.RLock()

    def register_backend(
        self,
        role: str,
        backend: SupportsBackendName,
        *,
        metadata: dict | None = None,
        gpu_memory_mb: int = 0,
        residency_group: str | None = None,
        shared_resident: bool | None = None,
        residency_policy: str = 'resident',
        healthcheck_fn: Callable[[], bool] | None = None,
        restart_fn: Callable[[], None] | None = None,
        unload_fn: Callable[[], None] | None = None,
        idle_unload_ttl_s: float = 0.0,
        warm_on_main_thread: bool = True,
    ) -> None:
        backend_health = getattr(backend, 'healthcheck', None)
        backend_restart = getattr(backend, 'restart', None)
        backend_unload = getattr(backend, 'unload', None)
        self.register_component(
            role,
            backend.backend_name,
            backend.warm,
            metadata=metadata,
            gpu_memory_mb=gpu_memory_mb,
            residency_group=residency_group,
            shared_resident=shared_resident,
            residency_policy=residency_policy,
            healthcheck_fn=healthcheck_fn or (backend_health if callable(backend_health) else None),
            restart_fn=restart_fn or (backend_restart if callable(backend_restart) else None),
            unload_fn=unload_fn or (backend_unload if callable(backend_unload) else None),
            idle_unload_ttl_s=idle_unload_ttl_s,
            warm_on_main_thread=warm_on_main_thread,
        )

    def register_component(
        self,
        role: str,
        name: str,
        warm_fn: Callable[[], None],
        *,
        metadata: dict | None = None,
        gpu_memory_mb: int = 0,
        residency_group: str | None = None,
        shared_resident: bool | None = None,
        residency_policy: str = 'resident',
        healthcheck_fn: Callable[[], bool] | None = None,
        restart_fn: Callable[[], None] | None = None,
        unload_fn: Callable[[], None] | None = None,
        idle_unload_ttl_s: float = 0.0,
        warm_on_main_thread: bool = True,
    ) -> None:
        residency_policy = residency_policy.strip().lower()
        if residency_policy not in {'resident', 'on_demand'}:
            raise ValueError(f'Unsupported residency_policy: {residency_policy}')
        if idle_unload_ttl_s < 0:
            raise ValueError(f'idle_unload_ttl_s must be non-negative, got {idle_unload_ttl_s!r}')
        state = ManagedComponentState(
            role=role,
            name=name,
            metadata=dict(metadata or {}),
            gpu_memory_mb=max(0, int(gpu_memory_mb)),
            residency_group=residency_group,
            shared_resident=bool(shared_resident if shared_resident is not None else residency_group is not None),
            residency_policy=residency_policy,
            idle_unload_ttl_s=float(idle_unload_ttl_s),
            warm_on_main_thread=bool(warm_on_main_thread),
        )
        self._components.append(_ManagedComponent(
            state=state,
            warm_fn=warm_fn,
            unload_fn=unload_fn,
            healthcheck_fn=healthcheck_fn,
            restart_fn=restart_fn,
        ))

    def warm_all(self, *, force: bool = False, skip_roles: set[str] | None = None) -> None:
        with self._lock:
            if self._warmed_once and not force:
                return
            skip = skip_roles or set()
            for component in self._components:
                if component.state.role in skip or component.state.name in skip:
                    continue
                if component.state.residency_policy != 'resident':
                    continue
                # Issue #19: components flagged warm_on_main_thread=False
                # must be warmed by their dedicated worker (e.g. the
                # preview decode thread). Skipping them here
                # avoids a wasted main-thread warm followed by an
                # unload+rewarm on the worker.
                if not component.state.warm_on_main_thread:
                    continue
                self._warm_component(component, force=force)
            self._warmed_once = True

    def warm_component(self, role_or_name: str, *, force: bool = False) -> dict | None:
        with self._lock:
            component = self._find_component(role_or_name)
            if component is None:
                return None
            self._warm_component(component, force=force)
            return component.state.to_dict()

    def acquire(self, role_or_name: str) -> dict | None:
        with self._lock:
            # Any acquire triggers a lazy reap of components that have been
            # idle past their TTL. This lets a long-idle writer backend
            # unload when a fresh lane kicks in, without needing a timer.
            self._reap_idle_locked()
            component = self._find_component(role_or_name)
            if component is None:
                return None
            # Re-acquiring a component that was scheduled for idle unload
            # cancels that schedule — the model is still resident and we
            # just need to increment the counter.
            component.state.release_scheduled_at_ns = None
            self._warm_component(component, force=False)
            component.state.acquire_count += 1
            component.state.last_used_monotonic_ns = time.monotonic_ns()
            return component.state.to_dict()

    def release(self, role_or_name: str) -> dict | None:
        with self._lock:
            component = self._find_component(role_or_name)
            if component is None:
                return None
            component.state.release_count += 1
            now_ns = time.monotonic_ns()
            component.state.last_used_monotonic_ns = now_ns
            if component.state.residency_policy == 'on_demand':
                if component.state.idle_unload_ttl_s > 0:
                    # Defer the unload — reap_idle() or the next acquire()
                    # will finalize it once the TTL has elapsed.
                    component.state.release_scheduled_at_ns = now_ns
                else:
                    self._unload_component(component)
            return component.state.to_dict()

    def reap_idle(self, *, now_ns: int | None = None) -> list[str]:
        """Unload any on_demand component whose idle TTL has elapsed.

        Returns the list of roles that were unloaded. Callers can invoke
        this manually (tests, service health ticks) or rely on the
        automatic call from acquire(). Thread-safe.
        """
        with self._lock:
            return self._reap_idle_locked(now_ns=now_ns)

    def _reap_idle_locked(self, *, now_ns: int | None = None) -> list[str]:
        """Internal reaper — must be called with self._lock held."""
        reaped: list[str] = []
        if now_ns is None:
            now_ns = time.monotonic_ns()
        for component in self._components:
            state = component.state
            if state.residency_policy != 'on_demand':
                continue
            scheduled = state.release_scheduled_at_ns
            if scheduled is None:
                continue
            ttl_ns = int(state.idle_unload_ttl_s * 1_000_000_000)
            if now_ns - scheduled < ttl_ns:
                continue
            state.release_scheduled_at_ns = None
            if state.loaded:
                self._unload_component(component)
                reaped.append(state.role)
        return reaped

    def snapshot(self) -> dict:
        with self._lock:
            allocated_gpu_memory_mb = self._allocated_gpu_memory_mb(loaded_only=True)
            configured_gpu_memory_mb = self._allocated_gpu_memory_mb(loaded_only=False)
            return {
                'component_count': len(self._components),
                'warmed_once': self._warmed_once,
                'total_gpu_memory_mb': self.total_gpu_memory_mb,
                'allocated_gpu_memory_mb': allocated_gpu_memory_mb,
                'configured_gpu_memory_mb': configured_gpu_memory_mb,
                'available_gpu_memory_mb': None if self.total_gpu_memory_mb is None else max(0, self.total_gpu_memory_mb - allocated_gpu_memory_mb),
                'residency_groups': self._residency_groups(loaded_only=False),
                'loaded_residency_groups': self._residency_groups(loaded_only=True),
                'components': [item.state.to_dict() for item in self._components],
            }

    def _find_component(self, role_or_name: str) -> _ManagedComponent | None:
        for component in self._components:
            if component.state.role == role_or_name or component.state.name == role_or_name:
                return component
        return None

    def _warm_component(self, component: _ManagedComponent, *, force: bool) -> None:
        if component.state.loaded and not force:
            if component.healthcheck_fn is not None:
                component.state.health_ok = bool(component.healthcheck_fn())
            return
        self._ensure_budget_for_load(component)
        try:
            started = time.perf_counter_ns()
            component.warm_fn()
            warm_ms = (time.perf_counter_ns() - started) / 1_000_000.0
            component.state.loaded = True
            component.state.warmed = True
            component.state.warm_count += 1
            component.state.last_warm_ms = warm_ms
            component.state.total_warm_ms += warm_ms
            component.state.last_error = None
            component.state.last_used_monotonic_ns = time.monotonic_ns()
            if component.healthcheck_fn is not None:
                component.state.health_ok = bool(component.healthcheck_fn())
        except Exception as exc:  # pragma: no cover
            component.state.loaded = False
            component.state.last_error = f"{type(exc).__name__}: {exc}"
            raise

    def _unload_component(self, component: _ManagedComponent) -> None:
        if not component.state.loaded:
            return
        if component.unload_fn is not None:
            try:
                component.unload_fn()
            except Exception as exc:  # pragma: no cover
                component.state.last_error = f"{type(exc).__name__}: {exc}"
                raise
        component.state.loaded = False
        component.state.unload_count += 1
        component.state.health_ok = None

    def _ensure_budget_for_load(self, target: _ManagedComponent) -> None:
        if self.total_gpu_memory_mb is None or target.state.gpu_memory_mb <= 0:
            return
        projected = self._projected_gpu_memory_mb(target)
        if projected <= self.total_gpu_memory_mb:
            return
        for component in self._components:
            if component is target:
                continue
            if not component.state.loaded:
                continue
            if component.state.residency_policy != 'on_demand':
                continue
            self._unload_component(component)
            projected = self._projected_gpu_memory_mb(target)
            if projected <= self.total_gpu_memory_mb:
                return
        raise RuntimeError(
            f'Loading {target.state.role} would require {projected} MiB GPU memory, '
            f'but only {self.total_gpu_memory_mb} MiB is budgeted'
        )

    def _projected_gpu_memory_mb(self, target: _ManagedComponent) -> int:
        total = 0
        grouped_peak: dict[str, int] = {}
        for component in self._components:
            state = component.state
            if component is not target and not state.loaded:
                continue
            if state.gpu_memory_mb <= 0:
                continue
            if state.residency_group and state.shared_resident:
                grouped_peak[state.residency_group] = max(grouped_peak.get(state.residency_group, 0), state.gpu_memory_mb)
            else:
                total += state.gpu_memory_mb
        return total + sum(grouped_peak.values())

    def _allocated_gpu_memory_mb(self, *, loaded_only: bool) -> int:
        total = 0
        grouped_peak: dict[str, int] = {}
        for component in self._components:
            state = component.state
            if loaded_only and not state.loaded:
                continue
            if state.gpu_memory_mb <= 0:
                continue
            if state.residency_group and state.shared_resident:
                grouped_peak[state.residency_group] = max(grouped_peak.get(state.residency_group, 0), state.gpu_memory_mb)
            else:
                total += state.gpu_memory_mb
        return total + sum(grouped_peak.values())

    def _residency_groups(self, *, loaded_only: bool) -> dict[str, dict]:
        groups: dict[str, dict] = {}
        for component in self._components:
            state = component.state
            if loaded_only and not state.loaded:
                continue
            if not state.residency_group:
                continue
            entry = groups.setdefault(state.residency_group, {
                'shared_resident': state.shared_resident,
                'gpu_memory_mb': 0,
                'members': [],
            })
            entry['gpu_memory_mb'] = max(entry['gpu_memory_mb'], state.gpu_memory_mb) if state.shared_resident else entry['gpu_memory_mb'] + state.gpu_memory_mb
            entry['members'].append(state.role)
        return groups
