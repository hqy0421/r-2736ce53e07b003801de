"""Watermark-windowed Count Encoder with baseline distribution monitoring.

This module keeps the original :class:`~category_encoders.count.CountEncoder`
encoding interface (``fit`` / ``transform`` / ``fit_transform`` against a fixed
historical baseline) and adds an event-time streaming layer:

* a fixed **baseline frequency** table is fit once from clean historical
  samples with the native ``CountEncoder`` and is never mutated afterwards,
  so unseen online categories cannot pollute it;
* online events are maintained inside a sliding window
  ``(watermark - W, watermark]`` (left-open / right-closed) keyed by *event
  time*, not arrival time -- events may arrive out of order;
* the watermark is advanced explicitly. On every advance only newly effective
  (buffered future) events and newly expired events are touched, in timestamp
  order; no full-window or full-vocabulary scan is ever performed;
* per encoded field the encoder reports the exact squared Hellinger distance
  between the baseline and the in-window distribution, plus the share of
  unseen categories (categories absent from the baseline).

Missing values (``NaN`` / ``None`` / ``NaT`` / ``pd.NA``) are treated as a
dedicated standalone category both in the baseline and online.

Exactness of the Hellinger statistic
------------------------------------

For discrete distributions ``p`` (baseline) and ``q`` (window) over the union
of their supports the squared Hellinger distance is

    H^2(p, q) = 1 - sum_k sqrt(p_k * q_k)

(equivalent to ``1/2 * sum_k (sqrt(p_k) - sqrt(q_k))^2``).  With integer
counts ``c_k`` over a window of ``N`` events and fixed baseline counts
``b_k`` over a baseline of ``B`` events,

    sqrt(p_k q_k) = sqrt(b_k * c_k) / sqrt(B * N).

Each per-column state therefore maintains

    T = sum_{k in baseline support} sqrt(b_k * c_k),

which is updated incrementally: adding / removing one event of category ``k``
only changes the single summand for ``k`` (the window total ``N`` is tracked
separately), giving an O(1)-per-affected-category update and an O(1) query.
Only integer counts enter the square roots; the division by the fixed
``sqrt(B)`` and the current ``sqrt(N)`` happens at query time.  A
:meth:`WatermarkCountEncoder.recompute_metrics` hook is provided for explicit
exact recomputation / drift verification and is covered by the test suite.
"""

from __future__ import annotations

import heapq
import math

import numpy as np
import pandas as pd

from category_encoders.count import CountEncoder

__author__ = 'joshua t. dunn'

# Standalone category label for missing values. A private sentinel object is
# used instead of np.nan because np.nan does not compare equal to itself and
# is awkward as a plain dict key (in particular two different float('nan')
# objects are distinct keys). It can never collide with a real user value.
_MISSING = object()


class WatermarkCountEncoder:
    """Count encoder with a fixed baseline and a watermark-driven time window.

    Parameters
    ----------
    window: int, float or pandas.Timedelta
        Width ``W`` of the event-time window. The active window is
        ``(watermark - W, watermark]``: the right endpoint is included, the
        left endpoint is excluded. Numeric widths pair with numeric event
        timestamps; ``Timedelta`` widths pair with datetime-like timestamps.
    timestamp_col: str
        Name of the column holding the event timestamp. It is never encoded.
    cols: list, optional
        Categorical columns to encode. If None, all non-timestamp columns are
        used (mirrors ``CountEncoder`` behaviour on string / categorical data
        when fit on a frame containing only the monitored fields).
    verbose, drop_invariant, return_df
        Forwarded as-is to the underlying baseline :class:`CountEncoder`.
    handle_missing, handle_unknown
        Must be ``'value'`` (the default): missing values are a standalone
        category and unknown values map to count 0 at transform time. Any
        other strategy would contradict the monitoring contract.

    Notes
    -----
    ``min_group_size`` / lumping and ``normalize`` are intentionally not
    exposed: the monitoring statistic requires exact per-category integer
    counts and a fixed, ungrouped baseline vocabulary.
    """

    def __init__(
        self,
        window,
        timestamp_col,
        cols=None,
        verbose=0,
        drop_invariant=False,
        return_df=True,
        handle_missing='value',
        handle_unknown='value',
    ):
        if handle_missing != 'value' or handle_unknown != 'value':
            raise ValueError(
                "WatermarkCountEncoder requires handle_missing='value' and "
                "handle_unknown='value' so that missing values form a "
                "standalone category and unknown values map to 0."
            )
        if not isinstance(window, (int, float, np.integer, np.floating, pd.Timedelta)):
            raise TypeError(
                'window must be a numeric duration or pandas.Timedelta, '
                f'got {type(window)!r}'
            )
        if isinstance(window, (int, float, np.integer, np.floating)) and not float(window) > 0:
            raise ValueError('window must be strictly positive')
        if isinstance(window, pd.Timedelta) and window <= pd.Timedelta(0):
            raise ValueError('window must be strictly positive')

        self.window = window
        self.timestamp_col = timestamp_col
        # Stored verbatim so get_params() round-trips for sklearn clone;
        # normalized to a list lazily in fit().
        self.cols = cols
        self.verbose = verbose
        self.drop_invariant = drop_invariant
        self.return_df = return_df
        self.handle_missing = handle_missing
        self.handle_unknown = handle_unknown

        # Fitted baseline state (populated by fit) ---------------------------
        self.baseline_encoder_: CountEncoder | None = None
        # col -> {category key: baseline integer count}
        self._baseline_counts: dict[str, dict[object, int]] = {}
        # col -> baseline total B
        self._baseline_totals: dict[str, int] = {}
        # col -> sqrt(baseline probability per baseline category)
        self._baseline_sqrt_p: dict[str, dict[object, float]] = {}
        self._is_fitted = False

        # Streaming state ----------------------------------------------------
        self._reset_stream_state()

    # ------------------------------------------------------------------ #
    # Original CountEncoder interface: fixed baseline, never mutated
    # ------------------------------------------------------------------ #
    def fit(self, X, y=None):
        """Fit the fixed baseline on historical, known-good samples.

        Online events never touch this baseline: unknown online categories
        have no baseline entry and therefore cannot pollute it.

        Parameters
        ----------
        X: pandas.DataFrame
            Historical samples. ``timestamp_col`` is ignored if present.
        y: ignored
        """
        X = self._check_dataframe(X)
        if self.timestamp_col not in X.columns:
            raise ValueError(
                f"timestamp_col {self.timestamp_col!r} not found in fit frame; "
                f'available columns: {list(X.columns)}'
            )
        if self.cols is None:
            self.cols = [c for c in X.columns if c != self.timestamp_col]
        else:
            self.cols = list(self.cols)
            missing = [c for c in self.cols if c not in X.columns]
            if missing:
                raise ValueError(f'cols not found in fit frame: {missing}')

        baseline_frame = X.drop(columns=[self.timestamp_col])
        self.baseline_encoder_ = CountEncoder(
            verbose=self.verbose,
            cols=list(self.cols),
            drop_invariant=self.drop_invariant,
            return_df=True,
            handle_unknown='value',
            handle_missing='value',
            normalize=False,
        ).fit(baseline_frame)

        self._baseline_counts = {}
        self._baseline_totals = {}
        self._baseline_sqrt_p = {}
        for col in self.cols:
            series = self.baseline_encoder_.mapping[col]
            counts: dict[object, int] = {}
            total = 0
            for raw_key, value in series.items():
                key = _MISSING if self._is_missing(raw_key) else raw_key
                count = int(value)
                counts[key] = count
                total += count
            if total <= 0:
                raise ValueError(
                    f'Baseline for column {col!r} is empty; cannot estimate frequencies.'
                )
            self._baseline_counts[col] = counts
            self._baseline_totals[col] = total
            self._baseline_sqrt_p[col] = {
                key: math.sqrt(count / total) for key, count in counts.items()
            }

        self._is_fitted = True
        self._reset_stream_state()
        return self

    def fit_transform(self, X, y=None):
        """Fit the baseline and return baseline count encodings of ``X``."""
        self.fit(X, y)
        return self.transform(X)

    def transform(self, X):
        """Encode ``X`` with the fixed baseline counts (native interface).

        Unknown categories receive count 0 and missing values receive the
        baseline missing-category count, exactly as a vanilla
        ``CountEncoder(handle_missing='value', handle_unknown='value')``
        would do.
        """
        self._check_fitted()
        X = self._check_dataframe(X)
        frame = X.drop(columns=[self.timestamp_col], errors='ignore')
        encoded = self.baseline_encoder_.transform(frame)
        if self.return_df:
            return encoded
        return encoded.to_numpy()

    # ------------------------------------------------------------------ #
    # Streaming ingestion and watermark maintenance
    # ------------------------------------------------------------------ #
    def ingest(self, X):
        """Ingest a batch of (possibly out-of-order) events.

        Events are classified against the current watermark ``wm``:

        * ``t <= wm - W``: the event is already past the left window edge --
          it can never belong to any future window and is dropped as late;
        * ``wm - W < t <= wm``: the event falls inside the current window and
          is incorporated immediately (late but still valid arrivals);
        * ``t > wm``: the event is buffered until the watermark reaches ``t``
          and becomes effective then.

        Before the first watermark advance everything is buffered.

        Returns the number of events dropped as irretrievably late.
        """
        self._check_fitted()
        X = self._check_dataframe(X)
        required = set(self.cols) | {self.timestamp_col}
        absent = required - set(X.columns)
        if absent:
            raise ValueError(f'Ingest frame is missing columns: {sorted(absent)}')

        timestamps = X[self.timestamp_col]
        if bool(timestamps.isna().any()):
            raise ValueError(f'Null timestamps in {self.timestamp_col!r} are not supported.')

        n_dropped = 0
        for row_position in range(len(X)):
            ts = self._coerce_timestamp(timestamps.iloc[row_position])
            eid = self._next_eid
            self._next_eid += 1
            values = {col: self._as_key(X[col].iloc[row_position]) for col in self.cols}
            self._events[eid] = (ts, values)

            if self._watermark is None or ts > self._watermark:
                # Future event: wait for the watermark.
                heapq.heappush(self._pending_heap, (ts, eid))
            elif ts <= self._watermark - self.window:
                # Left-open edge: t == watermark - W is excluded.
                del self._events[eid]
                n_dropped += 1
                self._late_dropped += 1
            else:
                heapq.heappush(self._active_heap, (ts, eid))
                self._add_to_window(values)
        return n_dropped

    def advance_watermark(self, new_watermark):
        """Advance the watermark and update the window incrementally.

        Only two sets of events are processed:

        1. buffered events with ``t <= new_watermark`` become effective
           (those already past the new left edge ``new_watermark - W`` are
           dropped as late instead of entering the window);
        2. active events with ``t <= new_watermark - W`` expire.

        Parameters
        ----------
        new_watermark: numeric or pandas.Timestamp
            Must be greater than or equal to the current watermark.

        Returns
        -------
        dict
            ``{'effective': k, 'expired': m, 'late_dropped': n}`` counts for
            this advance.
        """
        self._check_fitted()
        new_watermark = self._coerce_timestamp(new_watermark)
        if self._watermark is not None and new_watermark < self._watermark:
            raise ValueError(
                f'Watermark cannot move backwards: {new_watermark!r} < '
                f'{self._watermark!r}'
            )
        if self._watermark is None:
            self._watermark = new_watermark
        elif new_watermark == self._watermark:
            return {'effective': 0, 'expired': 0, 'late_dropped': 0}

        effective = expired = late_dropped = 0
        new_left_edge = new_watermark - self.window

        # 1) Buffered future events become effective in (ts, eid) order.
        while self._pending_heap and self._pending_heap[0][0] <= new_watermark:
            ts, eid = heapq.heappop(self._pending_heap)
            if ts <= new_left_edge:
                # Already expired before it ever became effective (left edge
                # itself is excluded).
                del self._events[eid]
                late_dropped += 1
                self._late_dropped += 1
                continue
            heapq.heappush(self._active_heap, (ts, eid))
            self._add_to_window(self._events[eid][1])
            effective += 1

        # 2) Active events past the left edge expire.
        while self._active_heap and self._active_heap[0][0] <= new_left_edge:
            ts, eid = heapq.heappop(self._active_heap)
            self._remove_from_window(self._events[eid][1])
            del self._events[eid]
            expired += 1

        self._watermark = new_watermark
        return {
            'effective': effective,
            'expired': expired,
            'late_dropped': late_dropped,
        }

    # ------------------------------------------------------------------ #
    # Metrics: O(1) query, no window / vocabulary scans
    # ------------------------------------------------------------------ #
    def get_metrics(self):
        """Return per-field monitoring metrics for the current window.

        Returns
        -------
        dict of dict
            ``{column: {'is_empty': bool, 'window_size': int,
            'hellinger_squared': float | None,
            'unseen_share': float | None, 'unseen_count': int,
            'distinct_categories': int}}``.

            An empty window is reported explicitly with ``is_empty=True`` and
            ``None`` statistics instead of a fabricated zero distribution.
        """
        self._check_fitted()
        return {col: self._metrics_for(col) for col in self.cols}

    def metrics_frame(self):
        """Same metrics as :meth:`get_metrics`, returned as a DataFrame."""
        return pd.DataFrame(self.get_metrics()).T

    def recompute_metrics(self):
        """Rebuild every per-column statistic from the current window.

        This is an O(window size) verification / drift-repair utility. The
        normal update path never needs it; tests use it to prove that the
        incremental state equals a brute-force recomputation.
        """
        self._check_fitted()
        rebuilt = {
            col: {'counts': {}, 'total': 0, 'unseen': 0, 'T': 0.0} for col in self.cols
        }
        # Only active-heap events belong to the current window; buffered
        # pending events must not be counted.
        for _ts, eid in self._active_heap:
            values = self._events[eid][1]
            for col in self.cols:
                key = values[col]
                state = rebuilt[col]
                state['counts'][key] = state['counts'].get(key, 0) + 1
                state['total'] += 1
                if key not in self._baseline_counts[col]:
                    state['unseen'] += 1
        for col in self.cols:
            state = rebuilt[col]
            t_value = 0.0
            for key, count in state['counts'].items():
                base = self._baseline_counts[col].get(key)
                if base is not None:
                    t_value += math.sqrt(base * count)
            state['T'] = t_value
            self._column_state[col] = state
        return self.get_metrics()

    # ------------------------------------------------------------------ #
    # Inspection helpers
    # ------------------------------------------------------------------ #
    @property
    def watermark(self):
        """Current watermark, or ``None`` before the first advance."""
        return self._watermark

    def window_bounds(self):
        """Return the current ``(left_edge_exclusive, watermark_inclusive]``.

        Returns ``None`` before the first watermark advance.
        """
        if self._watermark is None:
            return None
        return self._watermark - self.window, self._watermark

    @property
    def active_size(self):
        """Number of events currently inside the window."""
        return len(self._active_heap)

    @property
    def pending_size(self):
        """Number of buffered future events waiting on the watermark."""
        return len(self._pending_heap)

    @property
    def late_dropped(self):
        """Total number of events dropped for arriving past the left edge."""
        return self._late_dropped

    def window_distribution(self, col):
        """Return a copy of the current in-window counts for ``col``.

        Intended for inspection / tests; the monitoring hot path does not use
        it. Missing values appear under the encoder's missing sentinel.
        """
        self._check_fitted()
        if col not in self._column_state:
            raise KeyError(f'Unknown column {col!r}')
        return dict(self._column_state[col]['counts'])

    # ------------------------------------------------------------------ #
    # sklearn-style helpers so the encoder remains usable in pipelines
    # ------------------------------------------------------------------ #
    def get_params(self, deep=True):
        return {
            'window': self.window,
            'timestamp_col': self.timestamp_col,
            'cols': self.cols,
            'verbose': self.verbose,
            'drop_invariant': self.drop_invariant,
            'return_df': self.return_df,
            'handle_missing': self.handle_missing,
            'handle_unknown': self.handle_unknown,
        }

    def set_params(self, **params):
        for key, value in params.items():
            if not hasattr(self, key):
                raise ValueError(f'Invalid parameter {key!r}')
            setattr(self, key, value)
        return self

    # ================================================================== #
    # Internal helpers
    # ================================================================== #
    def _reset_stream_state(self):
        # event id -> (timestamp, {col: category key})
        self._events: dict[int, tuple] = {}
        # min-heaps of (timestamp, event id); tuple ordering breaks ties by id
        self._active_heap: list[tuple] = []
        self._pending_heap: list[tuple] = []
        self._next_eid = 0
        self._watermark = None
        self._late_dropped = 0
        self._column_state: dict[str, dict] = {}
        if self._is_fitted:
            for col in self.cols:
                self._column_state[col] = {
                    'counts': {},       # category key -> in-window integer count
                    'total': 0,         # N
                    'unseen': 0,        # events whose key is not in baseline
                    'T': 0.0,           # sum sqrt(b_k * c_k) over baseline keys
                }

    def _add_to_window(self, values):
        for col, key in values.items():
            state = self._column_state[col]
            base = self._baseline_counts[col].get(key)
            old_count = state['counts'].get(key, 0)
            if base is not None:
                if old_count > 0:
                    state['T'] -= math.sqrt(base * old_count)
                state['T'] += math.sqrt(base * (old_count + 1))
            else:
                state['unseen'] += 1
            state['counts'][key] = old_count + 1
            state['total'] += 1

    def _remove_from_window(self, values):
        for col, key in values.items():
            state = self._column_state[col]
            base = self._baseline_counts[col].get(key)
            old_count = state['counts'][key]
            if base is not None:
                state['T'] -= math.sqrt(base * old_count)
                if old_count > 1:
                    state['T'] += math.sqrt(base * (old_count - 1))
            else:
                state['unseen'] -= 1
            if old_count == 1:
                del state['counts'][key]
            else:
                state['counts'][key] = old_count - 1
            state['total'] -= 1

    def _metrics_for(self, col):
        state = self._column_state[col]
        n_events = state['total']
        if n_events == 0:
            return {
                'is_empty': True,
                'window_size': 0,
                'hellinger_squared': None,
                'unseen_share': None,
                'unseen_count': 0,
                'distinct_categories': 0,
            }
        baseline_total = self._baseline_totals[col]
        # H^2 = 1 - sum_k sqrt(p_k q_k) = 1 - T / sqrt(B * N)
        overlap = state['T'] / math.sqrt(baseline_total * n_events)
        # Clamp the overlap into [0, 1] to absorb floating-point round-off.
        overlap = min(1.0, max(0.0, overlap))
        return {
            'is_empty': False,
            'window_size': n_events,
            'hellinger_squared': 1.0 - overlap,
            'unseen_share': state['unseen'] / n_events,
            'unseen_count': state['unseen'],
            'distinct_categories': len(state['counts']),
        }

    @staticmethod
    def _is_missing(value):
        if value is _MISSING:
            return True
        try:
            return bool(pd.isna(value))
        except (TypeError, ValueError):
            return False

    def _as_key(self, value):
        if self._is_missing(value):
            return _MISSING
        return value

    def _coerce_timestamp(self, value):
        if self._is_missing(value):
            raise ValueError(f'Null timestamp in {self.timestamp_col!r} is not supported.')
        if isinstance(self.window, pd.Timedelta):
            ts = pd.Timestamp(value)
        elif np.isscalar(value) and isinstance(value, (int, float, np.integer, np.floating)):
            ts = float(value)
        else:
            # Fall back to pandas coercion for numpy datetime64 etc.
            ts = pd.Timestamp(value) if not isinstance(value, (int, float)) else float(value)
        return ts

    @staticmethod
    def _check_dataframe(X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f'Expected a pandas.DataFrame, got {type(X)!r}')
        return X

    def _check_fitted(self):
        if not self._is_fitted:
            raise ValueError(
                'This WatermarkCountEncoder instance is not fitted yet. '
                'Call fit on historical data before using it.'
            )
