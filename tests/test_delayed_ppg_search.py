import unittest

import numpy as np

from sparc.delayed_ppg_search import delayed_search
from test_sequential_ppg_search import ToyEvaluator, phone_ppg


class RecordingEvaluator(ToyEvaluator):
    def __init__(self):
        super().__init__()
        self.observations = []

    def observe(self, audio):
        self.observations.append(audio.copy())
        return super().observe(audio)


class DelayedControlTest(unittest.TestCase):
    def run_case(self, mode='feedback', plant=None):
        initial = np.zeros(12, dtype=np.int64)
        normalized = np.repeat(np.arange(3, dtype=np.float32)[:,None,None],24,axis=1).reshape(3,2,12)
        evaluator = RecordingEvaluator()
        result = delayed_search(initial,phone_ppg(np.ones(24,dtype=int)),normalized,np.arange(3),
                                np.full((3,3),1/3),evaluator,horizon_tokens=5,mode=mode,plant=plant)
        return result,evaluator

    def test_waveform_input_not_just_ppg_times_obeys_sixty_ms_cutoff(self):
        result,evaluator=self.run_case()
        expected_inputs=[]
        for step in result['trace']:
            start=step['start_token']; cutoff=max(0,start*320-960)
            self.assertEqual(step['information_cutoff_samples'],cutoff)
            self.assertLessEqual(step['observed_until_frame'],cutoff//160)
            self.assertEqual(step['observer_input_samples'],cutoff if cutoff>432 else 0)
            if cutoff>432: expected_inputs.append(cutoff)
            self.assertEqual(step['prefix_sha256_before'],step['retained_prefix_sha256_after'])
        self.assertEqual([len(a) for a in evaluator.observations[:-1]],expected_inputs)
        self.assertEqual(len(evaluator.observations[-1]),len(result['audio']))
        self.assertFalse(result['terminal_observation']['used_for_control'])
        self.assertEqual(result['terminal_observation']['observation_time_ms'],300)
        np.testing.assert_array_equal(result['biases'][:5],np.zeros((5,35)))
        self.assertGreater(np.linalg.norm(result['biases'][5]),0)

    def test_blind_context_uses_commands_even_when_actual_output_differs(self):
        result,evaluator=self.run_case(plant=lambda chunk:chunk+.02)
        for start,context in enumerate(evaluator.prefixes):
            cutoff=max(0,start*320-960)
            np.testing.assert_array_equal(context[:cutoff],result['audio'][:cutoff])
            np.testing.assert_array_equal(context[cutoff:],result['commanded_audio'][cutoff:start*320])
            if start:
                self.assertTrue(np.any(context[cutoff:]!=result['audio'][cutoff:start*320]))

    def test_no_feedback_never_updates_bias_and_control_has_no_eos_flush(self):
        result,_=self.run_case(mode='no_feedback')
        np.testing.assert_array_equal(result['biases'],np.zeros_like(result['biases']))
        self.assertFalse(any(x['eos_flush'] for x in result['trace']))
        self.assertEqual(len(result['control_observed_ppg']),16)
        self.assertEqual(len(result['observed_ppg']),24)

    def test_current_token_is_always_in_search_and_past_is_fixed(self):
        result,_=self.run_case()
        for step in result['trace']:
            self.assertEqual(step['edits'][0]['position'],step['start_token'])
            self.assertTrue(all(step['start_token']<=x['position']<step['planning_end_token'] for x in step['edits']))


if __name__=='__main__': unittest.main()
