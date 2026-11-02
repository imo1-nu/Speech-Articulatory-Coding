import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from sparc.japanese_ppg import JapaneseLogMel, JapanesePPGModel
from sparc.local_ppg_search import (
    TokenEvaluator, candidate_codes, posterior_score, priority_positions, search_tokens, write_report,
    transition_nll, transition_statistics,
)
from sparc.sweep_ppg_transition import aggregate


def posterior(labels):
    values = np.full((len(labels), 35), .001, dtype=np.float32)
    values[np.arange(len(labels)), labels] = .966
    return values


class LocalSearchTest(unittest.TestCase):
    def setUp(self):
        self.windows = np.repeat(np.arange(5, dtype=np.float32)[:, None, None], 24, axis=1).reshape(5, 2, 12)

    def test_search_improves_discrete_sequence_with_full_utterance_scores(self):
        initial = np.array([0, 0, 0, 0, 0])
        original = initial.copy()
        desired = np.array([1, 2, 3, 4, 1])
        target = posterior(np.repeat(desired, 2))
        calls = []

        def evaluate(sequences):
            calls.append(sequences.copy())
            return np.stack([posterior(np.repeat(z, 2)) for z in sequences])

        result = search_tokens(initial, target, evaluate, self.windows, np.arange(5),
                               passes=2, positions=5, neighbors=4, global_candidates=0, batch_size=3)
        np.testing.assert_array_equal(initial, original)
        np.testing.assert_array_equal(result['tokens'], desired)
        self.assertEqual(result['final_js'], 0)
        self.assertGreater(result['baseline_js'], .1)
        self.assertTrue(all(x['current_js'] <= x['previous_js'] for x in result['trace']))
        self.assertEqual(result['evaluations'], sum(len(x) for x in calls))
        self.assertTrue(all(np.count_nonzero(x['new_code'] != x['old_code']) == 1 for x in result['trace'] if x['accepted']))

    def test_rejects_batch_only_improvements_and_keeps_incumbent(self):
        initial = np.zeros(3, dtype=np.int64)
        target = posterior(np.ones(6, dtype=int))

        def evaluate(sequences):
            # Simulate a candidate whose batched score improves but verification does not.
            labels = np.ones(6, dtype=int) if len(sequences) > 1 else np.zeros(6, dtype=int)
            return np.repeat(posterior(labels)[None], len(sequences), axis=0)

        result = search_tokens(initial, target, evaluate, self.windows, np.arange(5),
                               passes=3, positions=3, neighbors=4, global_candidates=0, batch_size=4)
        np.testing.assert_array_equal(result['tokens'], initial)
        self.assertEqual(result['baseline_js'], result['final_js'])
        self.assertFalse(any(x['accepted'] for x in result['trace']))
        self.assertEqual(len(result['trace']), 3)  # Stop after an unsuccessful pass.

    def test_candidates_exclude_unused_and_current_codes_and_cover_far_codes(self):
        selected = candidate_codes(0, self.windows, np.array([0, 1, 2, 4]), 1, 5)
        self.assertEqual(selected[0], 1)
        self.assertEqual(selected[1], 4)
        self.assertEqual(set(selected), {1, 2, 4})
        self.assertEqual(len(selected), len(set(selected)))

    def test_priority_uses_20ms_tokens_and_40ms_support_with_context(self):
        distance = np.zeros(40)
        distance[22:26] = 1
        positions = priority_positions(distance, 20, 4)
        self.assertEqual(len(set(positions)), 4)
        self.assertTrue(all(7 <= x <= 15 for x in positions))
        self.assertEqual(len(priority_positions(distance, 20, 999)), 20)

    def test_invalid_budget_and_posteriors_are_rejected(self):
        with self.assertRaises(ValueError):
            search_tokens(np.array([0]), posterior([1, 1]), None, self.windows, np.arange(5), passes=0)
        with self.assertRaises(ValueError):
            posterior_score(posterior([1]), np.full((1, 35), np.nan))

    def test_batched_evaluator_uses_independent_fixed_conditions_and_pcm_audio(self):
        class Generator(torch.nn.Module):
            def forward(self, art, speaker):
                # Like SPARC, the generator is allowed to mutate its input pitch channel.
                art[:, 12] -= 50
                values = art[:, 0] / 20 + speaker[:, 0, None] / 100
                return values.repeat_interleave(320, dim=1)[:, None]

        class Extractor:
            frontend = JapaneseLogMel()
            model = JapanesePPGModel().eval()

        torch.set_num_threads(1)
        conditions = {'pitch': np.full((5, 1), 100, np.float32),
                      'loudness': np.full((5, 1), .1, np.float32),
                      'spk_emb': np.ones(64, np.float32)}
        fixed_pitch = conditions['pitch'].copy()
        evaluator = TokenEvaluator(Generator(), Extractor(), self.windows, conditions, .8, 'cpu')
        sequences = np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])
        batch = evaluator.synthesize(sequences).numpy()
        one = np.concatenate([evaluator.synthesize(z[None]).numpy() for z in sequences])
        np.testing.assert_array_equal(batch, one)
        np.testing.assert_array_equal(conditions['pitch'], fixed_pitch)
        np.testing.assert_array_equal(batch*32768, np.floor(batch*32768))
        ppg = evaluator(sequences)
        self.assertEqual(ppg.shape, (2, 10, 35))
        np.testing.assert_allclose(ppg.sum(-1), 1, atol=1e-6)

    def test_report_contains_target_before_after_audio_and_search_limits(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            write_report(output, [{'directory': 'basic5000/test', 'utterance_id': 'test',
                                   'text': '<unsafe>', 'baseline_js': .2, 'final_js': .1,
                                   'relative_improvement_percent': 50, 'edited_tokens': 3, 'token_count': 10}])
            content = (output/'index.html').read_text()
            self.assertIn('&lt;unsafe&gt;', content)
            self.assertNotIn('<unsafe>', content)
            self.assertEqual(content.count('<audio '), 4)
            self.assertIn('after.wav', content)
            self.assertIn('予算内', content)

    def test_transition_score_counts_both_edges_and_handles_endpoints(self):
        probabilities = np.array([[.8, .15, .05], [.1, .6, .3], [.2, .3, .5]])
        initial = np.array([0, 1, 2, 0])
        for index in range(len(initial)):
            edited = initial.copy(); edited[index] = (edited[index]+1) % 3
            expected_delta = 0.0
            for edge in [index-1, index]:
                if 0 <= edge < len(initial)-1:
                    expected_delta += np.log(probabilities[initial[edge], initial[edge+1]])
                    expected_delta -= np.log(probabilities[edited[edge], edited[edge+1]])
            self.assertAlmostEqual(transition_nll(edited, probabilities)-transition_nll(initial, probabilities),
                                   expected_delta/(len(initial)-1))
        self.assertEqual(transition_nll(np.array([1]), probabilities), 0)

    def test_soft_objective_can_accept_ppg_increase_for_better_transitions(self):
        initial = np.array([0, 1, 0])
        target = posterior(np.repeat(initial, 2))
        probabilities = np.full((5, 5), .0001)
        probabilities[:, 0] = .9996
        def evaluate(sequences):
            return np.stack([posterior(np.repeat(z, 2)) for z in sequences])
        result = search_tokens(initial, target, evaluate, self.windows, np.array([0]),
                               passes=1, positions=3, neighbors=1, global_candidates=0,
                               transition_probs=probabilities, transition_weight=1)
        np.testing.assert_array_equal(result['tokens'], [0, 0, 0])
        self.assertGreater(result['final_js'], result['baseline_js'])
        self.assertLess(result['final_objective'], result['baseline_objective'])
        self.assertLess(result['final_transition_nll'], result['baseline_transition_nll'])
        self.assertTrue(all(x['current_objective'] <= x['previous_objective'] for x in result['trace']))

    def test_zero_weight_reproduces_unconstrained_search_and_strong_weight_blocks_rare_edges(self):
        initial = np.zeros(4, dtype=int)
        target = posterior(np.ones(8, dtype=int))
        probabilities = np.full((5, 5), .0001)
        np.fill_diagonal(probabilities, .9996)
        def evaluate(sequences):
            return np.stack([posterior(np.repeat(z, 2)) for z in sequences])
        kwargs = dict(passes=1, positions=4, neighbors=4, global_candidates=0)
        free = search_tokens(initial, target, evaluate, self.windows, np.arange(5), **kwargs)
        zero = search_tokens(initial, target, evaluate, self.windows, np.arange(5),
                             transition_probs=probabilities, transition_weight=0, **kwargs)
        strong = search_tokens(initial, target, evaluate, self.windows, np.arange(5),
                               transition_probs=probabilities, transition_weight=10, **kwargs)
        np.testing.assert_array_equal(free['tokens'], zero['tokens'])
        self.assertEqual(free['final_js'], zero['final_js'])
        np.testing.assert_array_equal(strong['tokens'], initial)

    def test_invalid_transition_models_are_rejected(self):
        for probabilities, weight in [(None, 1), (np.zeros((5, 5)), .1),
                                       (np.ones((5, 5)), .1), (np.full((5, 5), np.nan), .1),
                                       (np.eye(5), .1), (None, -1)]:
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                search_tokens(np.array([0, 0]), posterior([0, 0]), None, self.windows, np.arange(5),
                              transition_probs=probabilities, transition_weight=weight)

    def test_transition_statistics_and_aggregates_use_edge_and_frame_weights(self):
        probabilities = np.full((5, 5), .2)
        counts = np.zeros((5, 5), dtype=int); counts[0, 0] = 10
        stats = transition_statistics(np.array([0, 0, 1, 1]), probabilities, counts)
        self.assertAlmostEqual(stats['unseen_transition_fraction'], 2/3)
        self.assertAlmostEqual(stats['self_transition_fraction'], 2/3)
        self.assertEqual(stats['max_run_tokens'], 2)
        records = []
        for frames, tokens, value in [(2, 2, 1), (4, 4, 3)]:
            row = {'common_ppg_frames': frames, 'token_count': tokens, 'transition_weight': .1, 'edited_tokens': 1}
            for stage in ['baseline', 'final']:
                for key in ['js', 'transition_nll', 'unseen_transition_fraction', 'self_transition_fraction']:
                    row[f'{stage}_{key}'] = value
            records.append(row)
        result = aggregate(records)
        self.assertAlmostEqual(result['final_js'], 14/6)
        self.assertAlmostEqual(result['final_transition_nll'], 10/4)


if __name__ == '__main__':
    unittest.main()
