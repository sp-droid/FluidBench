"""Per test case transient cutoff ("transientTimes" in benchmarks/{id}.json).

The cutoff is in rollout time: the same axis as ``times`` in a submission's rollout-analysis.json (cumulative
``dt`` from the rollout's initial snapshot, so snapshot n of a case sits at sum(dt[:n])). ``fluidbench.run``
computes every summary metric a second time over the snapshots after it ("settled").

Rule, applied to the true trajectory only (fields shaped (T, C, ...)), using the change rate
d_n = rms(u_{n+1} - u_n) / dt_n and its envelope e_n = max(d) over a trailing window of ~10% of the case:
  * steady end (final envelope below STEADY_FRACTION of max d): first time after which the snapshot stays within
    STEADY_DISTANCE of the final one, rms(u_n - u_end) / rms(u_end) (scale free, ignores start-up jumps);
  * unsteady end (e.g. periodic shedding): first time after which e stays within +-ENVELOPE_TOLERANCE of its
    final value.
A case whose settled part would be shorter than two windows never settles: its value is None (null).
"""
import numpy as np

STEADY_FRACTION = 0.05
STEADY_DISTANCE = 0.01
ENVELOPE_TOLERANCE = 0.10


def transient_time(fields, dt):
    """Transient cutoff of one case in rollout time, or None when the case does not settle."""
    fields = np.asarray(fields, dtype=np.float64)
    dt = np.asarray(dt, dtype=np.float64).reshape(-1)[: len(fields) - 1]
    rate = np.sqrt(np.mean(np.diff(fields, axis=0).reshape(len(fields) - 1, -1) ** 2, axis=1)) / dt
    count = len(rate)
    window = max(3, count // 10)
    envelope = np.array([rate[max(0, n - window + 1): n + 1].max() for n in range(count)])
    final = envelope[-1]
    if final < STEADY_FRACTION * rate.max():
        flat = fields.reshape(len(fields), -1)
        distance = np.sqrt(np.mean((flat - flat[-1]) ** 2, axis=1)) / max(np.sqrt(np.mean(flat[-1] ** 2)), 1e-12)
        inside = distance[:count] < STEADY_DISTANCE
    else:
        inside = np.abs(envelope / final - 1) <= ENVELOPE_TOLERANCE
    outside = np.flatnonzero(~inside)
    start = 0 if not len(outside) else outside[-1] + 1  # first step of the settled tail
    if count - start < 2 * window:
        return None
    # rate[n] compares snapshots n and n + 1; settled snapshots start at n = start, i.e. at rollout time sum(dt[:start]).
    return round(float(dt[:start].sum()), 6)


def transient_times(cases):
    """One cutoff per case, for cases given as (fields, dt) pairs in test-split order.

    The same value for every case: the latest cutoff of the cases that settle (None if none does).
    """
    times = [transient_time(fields, dt) for fields, dt in cases]
    settled = [time for time in times if time is not None]
    return [max(settled) if settled else None] * len(times)
