import unittest

import numpy as np

from sparc.evaluate_generated_audio import (
    ASR_MODEL,ASR_REVISION,acoustic_scores,aggregate,character_errors,normalize_cer,
)


class GeneratedAudioMetricTest(unittest.TestCase):
    def test_whisper_is_large_v3_not_turbo_and_has_pinned_revision(self):
        self.assertEqual(ASR_MODEL,'openai/whisper-large-v3')
        self.assertEqual(len(ASR_REVISION),40)

    def test_japanese_cer_normalization_keeps_orthography_differences(self):
        self.assertEqual(normalize_cer('ＡＢＣ　３回、ｶﾞ。'),'abc3回ガ')
        self.assertEqual(character_errors('日本語。','日本語')['cer'],0)
        errors=character_errors('日本語','日本')
        self.assertEqual((errors['character_errors'],errors['deletions']),(1,1))
        self.assertAlmostEqual(errors['cer'],1/3)
        self.assertEqual(character_errors('三','３')['substitutions'],1)
        self.assertEqual(character_errors('あ','あいう')['insertions'],2)
        self.assertEqual(character_errors('あ','あいう')['cer'],2)

    def test_identical_waveforms_have_zero_mcd_f0_error_and_unit_stoi(self):
        import torch
        torch.set_num_threads(1)
        t=np.arange(16000,dtype=np.float64)/16000
        audio=(.1*np.sin(2*np.pi*220*t)+.03*np.sin(2*np.pi*440*t)).astype(np.float32)
        score=acoustic_scores(audio,audio)
        self.assertAlmostEqual(score['mcd_db'],0,places=10)
        self.assertAlmostEqual(score['stoi'],1,places=6)
        self.assertEqual(score['f0_rmse_hz'],0)
        self.assertEqual(score['vuv_error_rate'],0)
        self.assertGreater(score['both_voiced_frames'],50)

    def test_pitch_difference_is_measured_on_both_voiced_frames(self):
        t=np.arange(16000,dtype=np.float64)/16000
        first=(.1*np.sin(2*np.pi*220*t)).astype(np.float32)
        second=(.1*np.sin(2*np.pi*240*t)).astype(np.float32)
        score=acoustic_scores(first,second)
        self.assertGreater(score['f0_rmse_hz'],15)
        self.assertLess(score['f0_rmse_hz'],25)
        self.assertGreater(score['mcd_db'],0)

    def test_cer_uses_micro_average_and_f0_uses_squared_error_pooling(self):
        rows=[]
        for chars,errors,frames,rmse in [(10,1,1,10),(30,9,3,20)]:
            rows.append({'condition':'test','character_errors':errors,'reference_characters':chars,
                         'mcd_db':2.,'mcd_speech_frames':frames,'stoi':.8,'f0_rmse_hz':rmse,
                         'f0_rmse_cents':rmse*10,'both_voiced_frames':frames,
                         'vuv_error_rate':.1,'acoustic_frames':frames})
        score=aggregate(rows)[0]
        self.assertEqual(score['cer'],.25)
        self.assertAlmostEqual(score['f0_rmse_hz'],np.sqrt((100+3*400)/4))


if __name__=='__main__': unittest.main()
