import unittest

import numpy as np

from sparc.sequential_ppg_search import (
    DelayedObserver, audio_hash, corrected_posterior, local_transition_nll, pcm16,
    plan_step, sequential_search, stitch_prediction,
)


def phone_ppg(labels):
    p = np.full((len(labels), 35), .001, dtype=np.float32)
    p[np.arange(len(labels)), labels] = .966
    return p


class ToyEvaluator:
    def __init__(self):
        self.maximum_ends = []
        self.prefixes = []

    def at(self, end, emitted, continuation, crossfade_samples):
        self.maximum_ends.append(end)
        self.prefixes.append(emitted.copy())
        def evaluate(sequences):
            assert sequences.shape[1] == end
            audio = np.repeat(sequences.astype(np.float32)/10, 320, axis=1)
            audio = stitch_prediction(audio, emitted, continuation, crossfade_samples)
            ppg = np.stack([phone_ppg(np.repeat(z, 2)) for z in sequences])
            return audio, ppg
        return evaluate

    def observe(self, audio):
        # A controlled predictor/observation mismatch, used to test feedback availability.
        labels = np.rint(audio[::160]*10).astype(int)
        return phone_ppg((labels+1) % 3)


class SequentialPPGTest(unittest.TestCase):
    def setUp(self):
        self.windows = np.repeat(np.arange(3, dtype=np.float32)[:, None, None], 24, axis=1).reshape(3, 2, 12)
        self.probabilities = np.full((3, 3), 1/3)

    def test_stitch_never_changes_emitted_samples_and_blends_only_future(self):
        emitted = pcm16(np.array([.1, .2, .3], dtype=np.float32))
        forecast = np.full((2, 8), .8, dtype=np.float32)
        continuation = np.full(5, .4, dtype=np.float32)
        result = stitch_prediction(forecast, emitted, continuation, 3)
        np.testing.assert_array_equal(result[:, :3], np.repeat(emitted[None], 2, axis=0))
        self.assertEqual(result[0, 3], pcm16(np.array(.4)))
        self.assertEqual(result[0, 5], pcm16(np.array(.8)))
        self.assertEqual(result[0, 7], pcm16(np.array(.8)))
        self.assertEqual(forecast[0, 0], np.float32(.8))

    def test_observer_delays_then_freezes_estimates_and_flushes_eos(self):
        observer = DelayedObserver(lag_frames=10)
        forecast = phone_ppg(np.zeros(30, dtype=int))
        first = phone_ppg(np.ones(10, dtype=int))
        update = observer.update(first, forecast[:10], feedback=True)
        self.assertEqual(update['observed_until_frame'], 0)
        np.testing.assert_array_equal(observer.bias, np.zeros(35))
        actual = phone_ppg(np.ones(20, dtype=int))
        update = observer.update(actual, forecast[:20], feedback=True)
        self.assertEqual(update['observed_until_frame'], 10)
        frozen = observer.observed.copy()
        self.assertGreater(np.linalg.norm(observer.bias), 0)
        changed_past = phone_ppg(np.full(30, 2, dtype=int))
        update = observer.update(changed_past, forecast, feedback=True, final=True)
        np.testing.assert_array_equal(observer.observed[:10], frozen)
        self.assertEqual(len(observer.observed), 30)
        self.assertTrue(update['eos_flush'])

    def test_disabled_feedback_has_exactly_zero_bias(self):
        observer = DelayedObserver(lag_frames=0)
        observer.update(phone_ppg([1]*20), phone_ppg([0]*20), feedback=False)
        np.testing.assert_array_equal(observer.bias, np.zeros(35))
        ppg = phone_ppg([0, 1])
        np.testing.assert_array_equal(corrected_posterior(ppg, observer.bias, .5), ppg)
        changed = corrected_posterior(ppg, np.arange(35)/35, .5)
        np.testing.assert_allclose(changed.sum(1), 1)

    def test_horizon_nll_includes_incoming_edge_but_excludes_constant_past(self):
        probabilities = np.array([[.8, .15, .05], [.1, .6, .3], [.2, .3, .5]])
        tokens = np.array([0, 1, 2, 0, 1])
        expected = -np.log([probabilities[2, 0], probabilities[0, 1]]).mean()
        self.assertAlmostEqual(local_transition_nll(tokens, 3, probabilities), expected)

    def test_planner_cannot_edit_frozen_tokens_and_accepts_verified_improvements(self):
        tokens = np.array([0, 0, 0, 0, 0])
        target = phone_ppg([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        evaluator = ToyEvaluator().at(5, np.zeros(640, dtype=np.float32), np.empty(0), 0)
        edited, _, _, stats = plan_step(tokens, 2, target, evaluator, self.windows, np.arange(3),
                                       self.probabilities, np.zeros(35), positions=3, passes=1)
        np.testing.assert_array_equal(edited[:2], tokens[:2])
        np.testing.assert_array_equal(edited[2:], np.ones(3, dtype=int))
        self.assertLess(stats['final_objective'], stats['initial_objective'])
        self.assertTrue(all(x['objective'] <= x['previous_objective'] for x in stats['edits']))

    def test_sequential_run_uses_only_horizon_and_available_observations(self):
        initial = np.zeros(17, dtype=np.int64)
        target = phone_ppg(np.ones(34, dtype=int))
        evaluator = ToyEvaluator()
        result = sequential_search(initial, target, self.windows, np.arange(3), self.probabilities,
                                   evaluator, horizon_tokens=10, mode='feedback', positions=2, passes=1)
        self.assertEqual(evaluator.maximum_ends, [10, 15, 17, 17])
        self.assertEqual(len(result['audio']), 17*320)
        self.assertEqual(len(result['observed_ppg']), 34)
        np.testing.assert_array_equal(initial, np.zeros(17, dtype=int))
        for prefix in evaluator.prefixes:
            np.testing.assert_array_equal(result['audio'][:len(prefix)], prefix)
        for step in result['trace']:
            self.assertEqual(step['prefix_sha256_before'], step['retained_prefix_sha256_after'])
            end = step['committed_until_token']
            self.assertLessEqual(step['planning_end_token'], step['start_token']+10)
            self.assertLessEqual(step['observed_until_frame'], end*2-(0 if step['eos_flush'] else 10))
        np.testing.assert_array_equal(result['biases'][:2], np.zeros((2, 35)))
        self.assertGreater(np.linalg.norm(result['biases'][2]), 0)

    def test_first_commit_is_invariant_to_tokens_outside_horizon(self):
        first = np.zeros(20, dtype=np.int64); second = first.copy(); second[10:] = 2
        target = phone_ppg(np.ones(40, dtype=int))
        commits = []
        for initial in [first, second]:
            evaluator = ToyEvaluator()
            output = sequential_search(initial, target, self.windows, np.arange(3), self.probabilities,
                                       evaluator, horizon_tokens=10, mode='no_feedback', positions=1, passes=1)
            commits.append(output['audio'][:1600])
        np.testing.assert_array_equal(*commits)

    def test_unedited_keeps_tokens_and_does_not_run_candidate_search(self):
        initial = np.arange(17, dtype=np.int64) % 3
        result = sequential_search(initial, phone_ppg(np.repeat(initial, 2)), self.windows, np.arange(3),
                                   self.probabilities, ToyEvaluator(), horizon_tokens=10, mode='unedited')
        np.testing.assert_array_equal(result['tokens'], initial)
        self.assertTrue(all(x['evaluations'] == 1 for x in result['trace']))
        np.testing.assert_array_equal(result['biases'], np.zeros_like(result['biases']))

    def test_feedback_bias_can_change_future_token_decisions(self):
        initial = np.zeros(5, dtype=np.int64)
        target = np.zeros((10, 35), dtype=np.float32)
        target[:, :2] = [.65, .35]
        def evaluate(sequences):
            outputs = []
            for z in sequences:
                ppg = np.zeros((len(z)*2, 35), dtype=np.float32)
                for index, token in enumerate(z):
                    ppg[index*2:index*2+2, :2] = [.7, .3] if token == 0 else [.4, .6]
                outputs.append(ppg)
            return np.zeros((len(sequences), 1600), dtype=np.float32), np.stack(outputs)
        bias = np.zeros(35); bias[0] = 2
        free, *_ = plan_step(initial, 0, target, evaluate, self.windows, np.array([0, 1]),
                             self.probabilities, np.zeros(35), weight=0, positions=5, passes=1)
        corrected, *_ = plan_step(initial, 0, target, evaluate, self.windows, np.array([0, 1]),
                                  self.probabilities, bias, gain=.5, weight=0, positions=5, passes=1)
        np.testing.assert_array_equal(free, initial)
        np.testing.assert_array_equal(corrected, np.ones(5, dtype=int))


if __name__ == '__main__':
    unittest.main()
