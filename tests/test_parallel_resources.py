"""CPU quota guards and native thread limits, without changing host settings."""

import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pytest
from scipy import stats
from threadpoolctl import threadpool_info, threadpool_limits

from epydemix.calibration import _worker, parallel
from epydemix.calibration.abc import ABCSampler


def _fake_cgroup(monkeypatch, tmp_path, kind, group, mount_root="/"):
    mount = tmp_path / "cpu mount"
    mount.mkdir()
    controller = "" if kind == "cgroup2" else "cpu,cpuacct"
    escaped_mount = str(mount).replace(" ", r"\040")
    metadata = {
        "/proc/self/cgroup": f"0:{controller}:{group}\n",
        "/proc/self/mountinfo": (
            f"1 0 0:1 {mount_root} {escaped_mount} rw - {kind} cgroup rw,{controller}\n"
        ),
    }
    read_text = Path.read_text

    def read(path, *args, **kwargs):
        return (
            metadata[str(path)]
            if str(path) in metadata
            else read_text(path, *args, **kwargs)
        )

    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(parallel.os, "cpu_count", lambda: 16)
    monkeypatch.setattr(parallel.os, "process_cpu_count", lambda: 16, raising=False)
    monkeypatch.setattr(
        parallel.os, "sched_getaffinity", lambda pid: set(range(8)), raising=False
    )
    return mount


@pytest.mark.skipif(
    os.name != "posix" or not Path("/proc/self/cgroup").exists(), reason="Linux cgroups"
)
@pytest.mark.parametrize("kind", ["cgroup", "cgroup2"])
@pytest.mark.parametrize("quota, expected", [(250000, 2), (50000, 1), (-1, 8)])
def test_cpu_capacity_includes_parent_quota(
    monkeypatch, tmp_path, kind, quota, expected
):
    """Check parent quotas, fractional CPU limits, and affinity in cgroup v1 and v2."""
    mount = _fake_cgroup(monkeypatch, tmp_path, kind, "/parent/child")
    child = mount / "parent/child"
    child.mkdir(parents=True)
    for directory, limit in [(mount, -1), (child.parent, quota), (child, 800000)]:
        if kind == "cgroup2":
            (directory / "cpu.max").write_text(
                f"{'max' if limit < 0 else limit} 100000"
            )
        else:
            (directory / "cpu.cfs_quota_us").write_text(str(limit))
            (directory / "cpu.cfs_period_us").write_text("100000")
    assert parallel._get_available_cpu_count() == expected
    monkeypatch.setattr(parallel.os, "sched_getaffinity", lambda pid: {4})
    assert parallel._get_available_cpu_count() == 1


@pytest.mark.parametrize("group, root", [("/tenant/job", "/tenant"), ("/", "/tenant")])
def test_cgroup_subtree_and_namespace_mounts(monkeypatch, tmp_path, group, root):
    """Read CPU quotas from subtree mounts and namespaced cgroup roots."""
    mount = _fake_cgroup(monkeypatch, tmp_path, "cgroup2", group, root)
    (mount / "cpu.max").write_text("200000 100000")
    assert list(parallel._iter_cgroup_cpu_quotas()) == [2]


@pytest.mark.skipif(
    os.name != "posix" or not Path("/proc/self/cgroup").exists(), reason="Linux cgroups"
)
def test_unreadable_limits_fail_conservatively(monkeypatch):
    """Warn and allow only one worker when CPU limits cannot be read."""

    def unreadable():
        raise PermissionError("CPU quota hidden")

    monkeypatch.setattr(parallel, "_iter_cgroup_cpu_quotas", unreadable)
    with pytest.warns(RuntimeWarning, match="allowing one worker"):
        assert parallel._get_available_cpu_count() == 1


@pytest.mark.parametrize(
    "strategy", ["smc", "rejection", "top_fraction", "projections"]
)
@pytest.mark.parametrize("external", [False, True])
def test_oversized_pool_rejected_before_rng_or_submission(
    monkeypatch, strategy, external
):
    """Reject pools above CPU capacity before task submission or RNG advancement."""
    monkeypatch.setattr(parallel, "_get_available_cpu_count", lambda: 2)
    sampler = ABCSampler(
        _native_simulate, {"beta": stats.uniform()}, {}, np.zeros(1), rng=43
    )
    state = sampler.rng.bit_generator.state
    with ProcessPoolExecutor(max_workers=3) as pool:

        def unexpected(*args, **kwargs):
            pytest.fail("Must reject before submitting tasks")

        monkeypatch.setattr(pool, "submit", unexpected)
        options = {"executor": pool, "n_workers": 1} if external else {"n_workers": 3}
        with pytest.raises(ValueError, match="Requested 3 workers, but only 2"):
            if strategy == "projections":
                sampler.run_projections({}, **options)
            else:
                sampler.calibrate(strategy=strategy, verbose=False, **options)
    assert sampler.rng.bit_generator.state == state


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "2"])
def test_invalid_worker_counts(value):
    """Reject non-positive, Boolean, and non-integer worker counts."""
    with pytest.raises((ValueError, TypeError)):
        parallel.executor_context(n_workers=value)


def _thread_counts():
    return [pool["num_threads"] for pool in threadpool_info()]


def _set_two_threads():
    threadpool_limits(limits=2)


def _native_simulate(params):
    assert _thread_counts() and set(_thread_counts()) == {1}
    if params.get("fail"):
        raise RuntimeError("simulation failed")
    return {"data": np.zeros(1)}


def _native_distance(data, simulation):
    assert set(_thread_counts()) == {1}
    return 0.0


def test_native_limits_apply_and_restore_in_spawn_pool_and_sequential():
    """Apply one native thread per evaluation and restore limits, including on failure."""
    with threadpool_limits(limits=2):
        before = _thread_counts()
        assert before and set(before) == {2}
        with pytest.raises(RuntimeError, match="simulation failed"):
            _worker.run_projection(_native_simulate, {"fail": True})
        assert _thread_counts() == before
        with ProcessPoolExecutor(
            max_workers=1, mp_context=get_context("spawn"), initializer=_set_two_threads
        ) as pool:
            with parallel.executor_context(n_workers=999, executor=pool) as reused:
                assert reused is pool
                pool.submit(_worker.run_projection, _native_simulate, {}).result()
                result = pool.submit(
                    _worker.evaluate_particle,
                    _native_simulate,
                    {},
                    [],
                    [],
                    np.zeros(1),
                    _native_distance,
                ).result()
                assert result["accepted"]
                with pytest.raises(RuntimeError, match="simulation failed"):
                    pool.submit(
                        _worker.run_projection, _native_simulate, {"fail": True}
                    ).result()
            assert set(pool.submit(_thread_counts).result()) == {2}
        assert _thread_counts() == before
    with parallel.executor_context(n_workers=1) as pool:
        pool.submit(_worker.run_projection, _native_simulate, {}).result()
