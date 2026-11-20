"""Tests for the watermark-windowed WatermarkCountEncoder."""
import math
from unittest import TestCase

import category_encoders as encoders
import numpy as np
import pandas as pd
from category_encoders.watermark_count import _MISSING


def _frame(times, col_a=None, col_b=None):
    data = {'ts': list(times)}
    if col_a is not None:
        data['a'] = list(col_a)
    if col_b is not None:
        data['b'] = list(col_b)
    return pd.DataFrame(data)


def _bruteforce_hellinger(baseline_counts, baseline_total, window_counts, window_total):
    """Reference H^2 = 1 - sum_k sqrt(p_k q_k) over the union of supports."""
    if window_total == 0:
        return None
    overlap = 0.0
    keys = set(baseline_counts) | set(window_counts)
    for key in keys:
        p = baseline_counts.get(key, 0) / baseline_total
        q = window_counts.get(key, 0) / window_total
        overlap += math.sqrt(p * q)
    return 1.0 - overlap


class TestWatermarkCountEncoder(TestCase):
    def setUp(self):
        # Baseline: a in {x:4, y:3, missing:1}, B = 8
        baseline = _frame(
            range(8),
            col_a=['x', 'x', 'x', 'x', 'y', 'y', 'y', None],
        )
        self.enc = encoders.WatermarkCountEncoder(window=5, timestamp_col='ts')
        self.enc.fit(baseline)
        self.baseline_counts = {'x': 4, 'y': 3, _MISSING: 1}
        self.baseline_total = 8

    def test_baseline_fit_fixed_and_encoding_interface(self):
        # Baseline counts extracted from the native CountEncoder mapping.
        self.assertEqual(self.enc._baseline_counts['a'], self.baseline_counts)
        self.assertEqual(self.enc._baseline_totals['a'], 8)

        # transform() preserves the native CountEncoder interface:
        # known values -> baseline counts, unknown -> 0, missing -> missing count.
        online = _frame([100, 101, 102], col_a=['x', 'z', None])
        encoded = self.enc.transform(online)
        self.assertIsInstance(encoded, pd.DataFrame)
        np.testing.assert_array_equal(encoded['a'].to_numpy(), [4, 0, 1])

        # return_df=False yields a numpy array, like CountEncoder.
        enc_np = encoders.WatermarkCountEncoder(
            window=5, timestamp_col='ts', return_df=False
        ).fit(_frame(range(8), col_a=['x'] * 7 + [None]))
        self.assertIsInstance(enc_np.transform(online), np.ndarray)

        # fit_transform returns encoded baseline frame.
        ft = self.enc.fit_transform(
            _frame(range(8), col_a=['x', 'x', 'x', 'x', 'y', 'y', 'y', None])
        )
        np.testing.assert_array_equal(ft['a'].to_numpy(), [4, 4, 4, 4, 3, 3, 3, 1])

    def test_unknown_categories_do_not_pollute_baseline(self):
        events = _frame(range(6), col_a=['z', 'z', 'w', 'x', 'y', None])
        self.enc.ingest(events)
        # wm=5 -> window (0, 5]: keep t=1..5 (5 events), t=0 never becomes active
        result = self.enc.advance_watermark(5)
        self.assertEqual(result['late_dropped'], 1)
        # Baseline mapping and counts are unchanged.
        self.assertEqual(self.enc._baseline_counts['a'], self.baseline_counts)
        self.assertEqual(self.enc.baseline_encoder_.mapping['a'].get('z'), None)
        encoded = self.enc.transform(_frame([0], col_a=['z']))
        self.assertEqual(encoded['a'].iloc[0], 0)

    def test_missing_is_standalone_category_online(self):
        events = _frame(range(4), col_a=[None, None, 'x', 'x'])
        self.enc.ingest(events)
        # wm=4 -> window (-1, 4]: all four events are active
        self.enc.advance_watermark(4)
        dist = self.enc.window_distribution('a')
        self.assertEqual(dist[_MISSING], 2)
        self.assertEqual(dist['x'], 2)
        metrics = self.enc.get_metrics()['a']
        expected = _bruteforce_hellinger(
            self.baseline_counts, 8, dist, 4
        )
        self.assertAlmostEqual(metrics['hellinger_squared'], expected, places=12)

    def test_out_of_order_late_events_included_immediately(self):
        # Advance first, then ingest an in-window late arrival and a future one.
        self.enc.advance_watermark(10)  # window (5, 10]
        dropped = self.enc.ingest(_frame([6, 7, 11], col_a=['x', 'y', 'x']))
        self.assertEqual(dropped, 0)
        # t=6 and t=7 are effective immediately; t=11 buffered.
        self.assertEqual(self.enc.active_size, 2)
        self.assertEqual(self.enc.pending_size, 1)
        dist = self.enc.window_distribution('a')
        self.assertEqual(dist, {'x': 1, 'y': 1})

        # Event at the left edge t=5 must be excluded; t=4 dropped as too late.
        dropped = self.enc.ingest(_frame([5, 4], col_a=['x', 'x']))
        self.assertEqual(dropped, 2)
        self.assertEqual(self.enc.active_size, 2)
        self.assertEqual(self.enc.late_dropped, 2)

    def test_future_events_wait_for_watermark(self):
        self.enc.ingest(_frame([2, 3], col_a=['x', 'x']))
        self.assertIsNone(self.enc.watermark)
        self.assertEqual(self.enc.pending_size, 2)
        self.assertEqual(self.enc.active_size, 0)
        # Empty window reported explicitly.
        metrics = self.enc.get_metrics()['a']
        self.assertTrue(metrics['is_empty'])
        self.assertIsNone(metrics['hellinger_squared'])
        self.assertIsNone(metrics['unseen_share'])

        result = self.enc.advance_watermark(3)  # window (-2, 3]
        self.assertEqual(result, {'effective': 2, 'expired': 0, 'late_dropped': 0})
        self.assertEqual(self.enc.active_size, 2)
        self.assertEqual(self.enc.window_bounds(), (-2, 3))

    def test_watermark_expiry_left_endpoint_excluded(self):
        self.enc.ingest(_frame([0, 1, 2, 3, 4, 5], col_a=['x'] * 6))
        self.enc.advance_watermark(5)  # window (0, 5]: t=0 excluded
        self.assertEqual(self.enc.active_size, 5)
        dist = self.enc.window_distribution('a')
        self.assertEqual(dist, {'x': 5})

        result = self.enc.advance_watermark(6)  # window (1, 6]: t=1 expires
        self.assertEqual(result['expired'], 1)
        self.assertEqual(self.enc.active_size, 4)

        result = self.enc.advance_watermark(10)  # window (5, 10]: all expire
        self.assertEqual(result['expired'], 4)
        self.assertEqual(self.enc.active_size, 0)
        self.assertTrue(self.enc.get_metrics()['a']['is_empty'])

    def test_pending_event_expiring_before_effective_is_dropped(self):
        # A buffered event only becomes effective when the watermark reaches
        # it; if the watermark jumps past its expiry, it is dropped as late.
        self.enc.ingest(_frame([1, 2], col_a=['x', 'y']))
        result = self.enc.advance_watermark(10)  # window (5, 10]
        self.assertEqual(result, {'effective': 0, 'expired': 0, 'late_dropped': 2})
        self.assertEqual(self.enc.pending_size, 0)
        self.assertEqual(self.enc.active_size, 0)
        self.assertEqual(self.enc.late_dropped, 2)

    def test_incremental_metrics_match_bruteforce_through_moves(self):
        rng = np.random.default_rng(42)
        categories = ['x', 'y', 'z', None]
        times = sorted(rng.integers(0, 30, size=40))
        values = [categories[i] for i in rng.integers(0, 4, size=40)]
        self.enc.ingest(_frame(times, col_a=values))

        for wm in range(0, 35, 3):
            self.enc.advance_watermark(wm)
            left = wm - 5
            live = [
                (t, v) for t, v in zip(times, values) if left < t <= wm
            ]
            window_counts: dict = {}
            for _t, v in live:
                key = _MISSING if pd.isna(v) else v
                window_counts[key] = window_counts.get(key, 0) + 1
            n_total = len(live)
            metrics = self.enc.get_metrics()['a']
            self.assertEqual(metrics['window_size'], n_total)
            self.assertEqual(metrics['distinct_categories'], len(window_counts))
            if n_total == 0:
                self.assertTrue(metrics['is_empty'])
                continue
            expected_h2 = _bruteforce_hellinger(
                self.baseline_counts, 8, window_counts, n_total
            )
            self.assertAlmostEqual(
                metrics['hellinger_squared'], expected_h2, places=10
            )
            unseen = sum(c for k, c in window_counts.items() if k not in self.baseline_counts)
            self.assertEqual(metrics['unseen_count'], unseen)
            self.assertAlmostEqual(metrics['unseen_share'], unseen / n_total)

    def test_recompute_metrics_equals_incremental_state(self):
        rng = np.random.default_rng(7)
        categories = ['x', 'y', 'z', None]
        times = rng.integers(0, 20, size=25)
        values = [categories[i] for i in rng.integers(0, 4, size=25)]
        self.enc.ingest(_frame(times, col_a=values))
        for wm in (5, 9, 14, 25):
            self.enc.advance_watermark(wm)
            incremental = self.enc.get_metrics()['a']
            rebuilt = self.enc.recompute_metrics()['a']
            for key in incremental:
                if isinstance(incremental[key], float):
                    self.assertAlmostEqual(incremental[key], rebuilt[key], places=12)
                else:
                    self.assertEqual(incremental[key], rebuilt[key])

    def test_identical_distribution_gives_zero_distance(self):
        # Recreate the baseline proportions inside the window: 4x, 3y, 1 missing
        events = _frame(
            range(8),
            col_a=['x', 'x', 'x', 'x', 'y', 'y', 'y', None],
        )
        self.enc.ingest(events)
        # wm=7 -> window (2, 7]: events t=3..7 stay active (1x, 3y, 1 missing);
        # feed three more x inside the window to restore the baseline
        # proportions 4x / 3y / 1-missing.
        self.enc.advance_watermark(7)
        self.enc.ingest(_frame([4, 5, 6], col_a=['x', 'x', 'x']))
        metrics = self.enc.get_metrics()['a']
        self.assertEqual(metrics['window_size'], 8)
        self.assertAlmostEqual(metrics['hellinger_squared'], 0.0, places=12)
        self.assertEqual(metrics['unseen_share'], 0.0)

    def test_disjoint_distribution_gives_distance_one(self):
        events = _frame(range(3), col_a=['z', 'z', 'w'])
        self.enc.ingest(events)
        # wm=2 -> window (-3, 2]: all three unseen events active
        self.enc.advance_watermark(2)
        metrics = self.enc.get_metrics()['a']
        self.assertAlmostEqual(metrics['hellinger_squared'], 1.0, places=12)
        self.assertAlmostEqual(metrics['unseen_share'], 1.0)

    def test_metrics_frame_and_multiple_columns(self):
        baseline = _frame(
            range(6),
            col_a=['x', 'x', 'y', 'y', 'y', None],
            col_b=['p', 'p', 'q', 'q', None, None],
        )
        enc = encoders.WatermarkCountEncoder(window=4, timestamp_col='ts').fit(baseline)
        enc.ingest(
            _frame(range(4), col_a=['x', 'z', None, 'y'], col_b=['p', 'p', 'r', None])
        )
        # wm=3 -> window (-1, 3]: all four events active
        enc.advance_watermark(3)
        frame = enc.metrics_frame()
        self.assertEqual(set(frame.index), {'a', 'b'})
        self.assertFalse(frame.loc['a', 'is_empty'])
        self.assertEqual(frame.loc['a', 'unseen_count'], 1)  # only 'z'
        self.assertEqual(frame.loc['b', 'unseen_count'], 1)  # only 'r'

    def test_datetime_timestamps_and_timedelta_window(self):
        baseline = _frame(pd.date_range('2026-01-01', periods=4), col_a=['x', 'x', 'y', 'y'])
        enc = encoders.WatermarkCountEncoder(
            window=pd.Timedelta(hours=1), timestamp_col='ts'
        ).fit(baseline)
        events = _frame(
            pd.to_datetime(
                [
                    '2026-02-01 00:00',
                    '2026-02-01 00:20',  # out of order arrival, in window later
                    '2026-02-01 00:40',
                    '2026-02-01 01:30',
                ]
            ),
            col_a=['x', 'y', 'x', 'y'],
        )
        enc.ingest(events)
        enc.advance_watermark(pd.Timestamp('2026-02-01 01:00'))
        # Window (00:00, 01:00]: first event at exactly 00:00 is excluded.
        self.assertEqual(enc.active_size, 2)
        bounds = enc.window_bounds()
        self.assertEqual(bounds[0], pd.Timestamp('2026-02-01 00:00'))
        self.assertEqual(bounds[1], pd.Timestamp('2026-02-01 01:00'))

    def test_watermark_monotonic_and_validation(self):
        self.enc.advance_watermark(5)
        with self.assertRaises(ValueError):
            self.enc.advance_watermark(4)
        with self.assertRaises(ValueError):
            encoders.WatermarkCountEncoder(window=0, timestamp_col='ts')
        with self.assertRaises(ValueError):
            encoders.WatermarkCountEncoder(
                window=5, timestamp_col='ts', handle_missing='return_nan'
            )
        with self.assertRaises(ValueError):
            encoders.WatermarkCountEncoder(
                window=5, timestamp_col='ts', handle_unknown='error'
            )
        with self.assertRaises(TypeError):
            encoders.WatermarkCountEncoder(window='5s', timestamp_col='ts')

    def test_null_timestamps_rejected(self):
        with self.assertRaises(ValueError):
            self.enc.ingest(_frame([1, None], col_a=['x', 'y']))

    def test_operation_before_fit_raises(self):
        enc = encoders.WatermarkCountEncoder(window=5, timestamp_col='ts')
        with self.assertRaises(ValueError):
            enc.get_metrics()
        with self.assertRaises(ValueError):
            enc.ingest(_frame([1], col_a=['x']))

    def test_get_set_params_and_clone(self):
        from sklearn.base import clone

        enc = encoders.WatermarkCountEncoder(
            window=5, timestamp_col='ts', cols=['a'], return_df=False
        )
        params = enc.get_params()
        self.assertEqual(params['window'], 5)
        enc.set_params(window=9)
        self.assertEqual(enc.window, 9)
        cloned = clone(enc)
        self.assertEqual(cloned.window, 9)
        self.assertEqual(cloned.timestamp_col, 'ts')
        self.assertEqual(cloned.cols, ['a'])

    def test_updates_touch_only_affected_categories(self):
        # Structural guarantee: adding/removing an event changes exactly one
        # count entry per column and the O(1) T accumulator accordingly.
        self.enc.advance_watermark(10)
        self.enc.ingest(_frame([6], col_a=['x']))
        state = self.enc._column_state['a']
        self.assertEqual(state['counts'], {'x': 1})
        t_after_add = state['T']
        self.assertAlmostEqual(t_after_add, math.sqrt(4 * 1))

        self.enc.ingest(_frame([7], col_a=['x']))
        self.assertAlmostEqual(state['T'], math.sqrt(4 * 2))
        self.assertEqual(state['total'], 2)

        self.enc.advance_watermark(11)  # window (6, 11]: t=6 expires, t=7 stays
        self.assertEqual(state['counts'], {'x': 1})
        self.assertAlmostEqual(state['T'], math.sqrt(4 * 1))
        self.assertEqual(state['total'], 1)
